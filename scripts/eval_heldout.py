#!/usr/bin/env python
"""Held-out loss for a checkpoint that never logged one, on train.py's own terms.

    scripts/eval_heldout.py runs/policy.pt runs/9M-bc-2026-08-01/best.pt

Why this exists. `runs/9M-causal` -- the behavioural-cloning run behind the live
`runs/policy.pt` -- logged 31 puzzle evals and ZERO val_loss. So when the 2026-08-01
retrain finished ~3 puzzle points ahead, the only available comparison was on the
metric PHILOSOPHY forbids selecting on (sigma 1.5 points at n=1000), while the
metric it mandates did not exist for one side. This computes the missing side.

The comparison is only worth anything if it is the SAME measurement, so this
replicates `train.py`'s `validate()` exactly rather than approximately:

  - the same bag (`data/test/<target>_data.bag`) and the first `--val-batches`
    batches of it,
  - `make_loader(..., num_workers=0, shuffle_buffer=1, seed=0, infinite=False)`,
    which is deterministic, so every checkpoint sees identical batches in an
    identical order,
  - `torch.autocast("cuda", bfloat16)` then `.float()` before the loss, matching
    the training objective's dtype path,
  - and the EMA weights, not the raw ones. train.py validates `ema` because that
    is what gets evaluated, promoted and played; scoring `model` here would
    describe a network that never ships.

**Calibration (`--calibration`, value target only).** Is the head correct or
merely confident? MCTS backs Q up from the value head, so a net that is sharper
rather than better concentrates visits and can win a match while playing worse
chess. Larger nets are typically sharper, which makes this exactly the failure
mode a capacity comparison invites, and a score cannot see it.

This measures it against the HELD-OUT LABEL, which is the non-circular way. The
match-archive version of the same idea does not work: `scripts/calibration.py`
found that 90% of an engine's apparent bias there is just the match result, and
the Council finding built on it ("overconfident by +0.021 at parity, +0.126 when
outclassed") was retracted 2026-08-14 because both numbers are the regression
line. Here the label does not depend on who was holding the pieces, so a bias is
a property of the network.

Read `bias` and `ece` together. Bias says which direction the head leans, ECE
says how far off it is per decile regardless of direction, and Brier is the
scoring rule that both decompose out of. Compare candidate against incumbent on
the same batches; the absolute values are meaningless without a partner.

**Do not compare this Brier to `scripts/calibration.py`'s.** The state-value
labels are CONTINUOUS win probabilities, not game outcomes: measured over the
held-out bag they run the full [0, 1] with mean 0.514 and sd 0.274, and only
5.8% sit exactly on 0, 0.5 or 1. So this Brier is a mean squared error against
a soft target and lands near 0.003, while a Brier against realised results
lands near 0.09. Same name, different scale, thirty times apart.

Read-only. Loads checkpoints, prints numbers, writes nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sumofish.data import make_loader  # noqa: E402
from sumofish.hlgauss import HLGauss  # noqa: E402
from sumofish.hlgauss import loss as hl_loss  # noqa: E402
from sumofish.model import PRESETS, ChessTransformer, ModelConfig  # noqa: E402


def held_out_loss(ckpt_path: Path, target: str, val_data: Path, val_batches: int,
                  batch_size: int, device: str,
                  calibration: bool = False) -> tuple[float, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_saved = ckpt.get("cfg")
    args_saved = ckpt.get("args")
    args_saved = vars(args_saved) if hasattr(args_saved, "__dict__") else (args_saved or {})

    preset = args_saved.get("preset", "9M")
    is_value = target == "state_value"
    bins = args_saved.get("value_bins", 64)
    overrides = {"output_size": bins} if is_value else {}
    causal = str(args_saved.get("causal", "1")) in ("1", "True", "true")
    cfg = ModelConfig(**{**PRESETS[preset].__dict__, **overrides, "causal": causal})
    model = ChessTransformer(cfg).to(device)

    # EMA if present, exactly as train.py's validate() does.
    state = ckpt.get("ema") or ckpt["model"]
    model.load_state_dict(state)
    model.eval()

    hl = HLGauss(bins=bins, device=device) if is_value else None
    loader = make_loader(str(val_data), policy=target, batch_size=batch_size,
                         num_workers=0, shuffle_buffer=1, seed=0, infinite=False)
    total, seen = 0.0, 0
    # Calibration accumulators. Kept as running sums rather than stored tensors
    # so this stays O(1) in memory over any number of batches.
    cal = {"n": 0, "sum_p": 0.0, "sum_y": 0.0, "sq": 0.0,
           "bins": [[0.0, 0.0, 0] for _ in range(10)]}
    with torch.no_grad():
        for tokens, targets in loader:
            if seen >= val_batches:
                break
            tokens = tokens.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(tokens)
            total += float(hl_loss(logits.float(), hl.targets(targets)) if is_value
                           else F.cross_entropy(logits.float(), targets))
            if calibration and is_value:
                # The head's own point estimate, which is what the search reads.
                pred = hl.expectation(logits.float()).flatten().double()
                lab = targets.flatten().double()
                cal["n"] += lab.numel()
                cal["sum_p"] += float(pred.sum())
                cal["sum_y"] += float(lab.sum())
                cal["sq"] += float(((pred - lab) ** 2).sum())
                # Decile occupancy, by predicted probability.
                idx = (pred * 10).clamp(0, 9).long()
                for b in range(10):
                    m = idx == b
                    k = int(m.sum())
                    if k:
                        cal["bins"][b][0] += float(pred[m].sum())
                        cal["bins"][b][1] += float(lab[m].sum())
                        cal["bins"][b][2] += k
            seen += 1
    del model
    torch.cuda.empty_cache()
    meta = {"step": ckpt.get("step"), "preset": preset, "causal": causal,
            "used_ema": "ema" in ckpt and ckpt["ema"] is not None,
            "batches": seen}
    if calibration and is_value and cal["n"]:
        n = cal["n"]
        meta["calibration"] = {
            "positions": n,
            "brier": cal["sq"] / n,
            "mean_predicted": cal["sum_p"] / n,
            "mean_label": cal["sum_y"] / n,
            "bias": (cal["sum_p"] - cal["sum_y"]) / n,
            "ece": sum(abs(bp / bn - by / bn) * bn
                       for bp, by, bn in cal["bins"] if bn) / n,
        }
    elif calibration and not is_value:
        # Say so rather than printing nothing, which reads as a clean result.
        meta["calibration_skipped"] = ("policy head: cross-entropy over 1968 "
                                       "moves has no win probability to calibrate")
    return total / max(1, seen), meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--target", default="behavioral_cloning",
                    choices=["behavioral_cloning", "state_value"])
    ap.add_argument("--val-data", default=None)
    ap.add_argument("--val-batches", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--calibration", action="store_true",
                    help="value target only: Brier, bias and ECE against the "
                         "held-out label. Run it on candidate AND incumbent, "
                         "and read the difference, not the absolute number.")
    args = ap.parse_args()

    val_data = Path(args.val_data) if args.val_data else \
        ROOT / f"data/test/{args.target}_data.bag"
    if not val_data.exists():
        sys.exit(f"no held-out bag at {val_data}")
    print(f"held-out: {val_data.name}, first {args.val_batches} batches of "
          f"{args.batch_size}, seed 0 (deterministic: every checkpoint sees the "
          f"same batches in the same order)\n")

    results = []
    for path in args.checkpoints:
        p = Path(path)
        if not p.exists():
            print(f"  {path}: MISSING"); continue
        loss, meta = held_out_loss(p, args.target, val_data, args.val_batches,
                                   args.batch_size, args.device, args.calibration)
        results.append((str(p), loss, meta))
        print(f"  {p.name:<24} val {loss:.5f}   step {meta['step']}  "
              f"{meta['preset']}  ema={meta['used_ema']}")
        c = meta.get("calibration")
        if c:
            print(f"  {'':<24} calib brier {c['brier']:.5f}  "
                  f"pred {c['mean_predicted']:.4f}  label {c['mean_label']:.4f}  "
                  f"bias {c['bias']:+.4f}  ece {c['ece']:.4f}  "
                  f"({c['positions']:,} positions)")
        elif meta.get("calibration_skipped"):
            print(f"  {'':<24} calibration skipped: {meta['calibration_skipped']}")

    if len(results) >= 2:
        best = min(results, key=lambda r: r[1])
        worst = max(results, key=lambda r: r[1])
        print(f"\n  lowest: {Path(best[0]).parent.name}/{Path(best[0]).name} "
              f"at {best[1]:.5f}, better by {worst[1] - best[1]:.5f} than "
              f"{Path(worst[0]).parent.name}/{Path(worst[0]).name}")
        print("  Held-out loss has no sigma quoted here: it is a mean over a FIXED "
              "deterministic set, so repeat runs give the same number. That makes it "
              "comparable, NOT free of the usual caveat -- it is still an instrument, "
              "and PHILOSOPHY ranks Elo above it for deciding what ships.")
        cals = [(Path(r[0]).name, r[2]["calibration"]) for r in results
                if r[2].get("calibration")]
        if len(cals) >= 2:
            (n0, c0), (n1, c1) = cals[0], cals[-1]
            print(f"\n  calibration, {n1} against {n0}: "
                  f"brier {c1['brier'] - c0['brier']:+.5f}, "
                  f"|bias| {abs(c1['bias']) - abs(c0['bias']):+.4f}, "
                  f"ece {c1['ece'] - c0['ece']:+.4f}  (negative is better on all three)")
            print("  A candidate that wins its match while these get worse is the "
                  "sharper-not-better failure mode. Same batches both sides, so "
                  "the difference is the network and nothing else.")


if __name__ == "__main__":
    main()
