#!/usr/bin/env python
"""What `scripts/grow_net.py` preserves, measured where theory pins the answer.

Plan item 3.3. Three checks, in the order they earn their keep:

  1. **The positive control, and it is the reason this file exists.** Growing
     to an integer multiple duplicates every channel the same number of times,
     so LayerNorm sees the same population and the wider net must compute the
     SAME FUNCTION. Theory demands 0.0000 there, which makes it the one case
     where a wrong answer cannot be argued with. It caught the embedding-scale
     bug on 2026-08-15 (`forward` multiplies by sqrt(embedding_dim), so the
     duplicated residual stream came out sqrt(2) too large) while every weight
     tensor verified correct against the Net2Net formula.

  2. **The d=384 approximation, on REAL positions.** 1.5x growth duplicates
     half the channels and leaves the rest alone, so it is approximate by
     construction and the only question is how approximate.

  3. **The step-10k recovery gate**, when pointed at a training run with
     `--run`. Growth's whole argument is that every checkpoint is deployable;
     at 1.5x that argument does not hold until the value head's held-out loss
     comes back to the donor's. No recovery by 10k means the growth scheme is
     broken and the fallback is a cold start.

    tests/identity_grow.py                       # 1 and 2
    tests/identity_grow.py --run 19M-fused       # and 3

**Why real positions and not `torch.randint`.** `grow_net.py`'s own report
measures max|delta| over uniformly random token ids, and both LAB-NOTES and the
script's docstring call those "real token sequences". They are not: a uniform
draw over the 31-token vocabulary is a board that cannot exist, and a
divergence measured off-distribution is not evidence about the positions the
net will actually see. This reads FENs out of the held-out state-value bag and
tokenises them, so the number describes the distribution the recovery gate will
be judged on. Both numbers are printed, because the gap between them is itself
the finding.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sumofish import bagz                                    # noqa: E402
from sumofish.model import ChessTransformer, ModelConfig     # noqa: E402
from sumofish.tokenizer import tokenize                      # noqa: E402
from scripts.grow_net import grow                            # noqa: E402

# The 2x control is exact to float32 noise, so the bound is a float32 bound and
# not a tolerance anyone chose. Measured 0.000000 on 2026-08-15 and again here.
EXACT = 1e-4
# Recovery gate, plan 3.2: the reproducibility floor of the held-out
# instrument. Anything inside this is the same number measured twice.
RECOVERY = 0.014


def real_tokens(n: int) -> torch.Tensor:
    """`n` tokenised positions from the HELD-OUT bag, in bag order.

    Held-out rather than training, because the recovery gate is a held-out
    number and this should describe the same population.
    """
    src = bagz.BagReader(ROOT / "data/test/state_value_data.bag")
    rows = [tokenize(bagz.decode_state_value(src[i])[0])
            for i in range(min(n, len(src)))]
    return torch.tensor(np.stack(rows), dtype=torch.long)


def delta(donor: dict, new_d: int, tokens: torch.Tensor, *, fused: bool,
          bins: int = 64) -> tuple[float, float]:
    """max|delta| and MSE of the grown net's value head against the donor."""
    state, cfg_new, cfg_old, _copied, _cold = grow(
        donor, new_d, fused=fused, bins=bins)
    old = ChessTransformer(cfg_old).eval()
    old.load_state_dict(donor.get("ema") or donor["model"])
    new = ChessTransformer(cfg_new).eval()
    new.load_state_dict(state)
    with torch.no_grad():
        o = old(tokens).float()
        n = new(tokens).float()[:, :cfg_old.output_size]
    return (o - n).abs().max().item(), ((o - n) ** 2).mean().item()


def recovery(run: str, donor_loss: float | None) -> int:
    """Assert the value head recovered the donor's held-out loss by step 10k.

    Reads the run's own `log.jsonl` rather than being told a number, so this
    cannot pass by being handed a figure from somewhere else.
    """
    log = ROOT / "runs" / run / "log.jsonl"
    if not log.exists():
        print(f"  SKIP recovery: {log} does not exist yet")
        return 0
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    # train.py logs the VALUE head as `val_loss` and the policy head as
    # `val_loss_policy`, fused or not, so this is the value head's own number.
    evals = [r for r in rows if r.get("val_loss") is not None
             and r.get("step") is not None]
    if not evals:
        print("  SKIP recovery: no per-head held-out rows logged yet")
        return 0
    if donor_loss is None:
        donor_loss = json.loads((ROOT / "runs/value.pt.json").read_text()) \
            .get("held_out_loss", {}).get("new")
    if donor_loss is None:
        print("  SKIP recovery: no donor held-out loss on record; pass --donor-loss")
        return 0

    at10k = [r for r in evals if r["step"] <= 10_000]
    best = min((r["val_loss"] for r in at10k), default=None)
    latest = evals[-1]
    print(f"  donor held-out (value head):        {donor_loss:.4f}")
    print(f"  best by step 10k:                   "
          f"{'n/a' if best is None else f'{best:.4f}'}")
    print(f"  latest (step {latest['step']:,}):"
          f"{'':>14}{latest['val_loss']:.4f}")
    if best is None:
        print(f"  PENDING: run has not reached step 10k "
              f"(at {latest['step']:,})")
        return 0
    gap = best - donor_loss
    ok = gap <= RECOVERY
    print(f"  gap {gap:+.4f} against a +-{RECOVERY} floor: "
          f"{'RECOVERED' if ok else 'NOT RECOVERED'}")
    if not ok:
        print("  The growth scheme did not return the donor's quality in 10k "
              "steps. Plan 3.2's falsifier: fall back to a cold start and "
              "accept the later first probe.")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--donor", default=str(ROOT / "runs/value.pt"))
    ap.add_argument("--positions", type=int, default=256)
    ap.add_argument("--run", help="training run to check the 10k gate against")
    ap.add_argument("--donor-loss", type=float,
                    help="the donor's held-out value loss; read from "
                         "runs/value.pt.json when omitted")
    a = ap.parse_args()

    donor = torch.load(a.donor, map_location="cpu", weights_only=False)
    tokens = real_tokens(a.positions)
    torch.manual_seed(0)
    noise = torch.randint(0, ModelConfig().vocab_size, tokens.shape)
    fails = 0

    print(f"positions: {tuple(tokens.shape)} real, from the held-out SV bag\n")

    # 1. The positive control. 2x duplicates every channel exactly once, so
    #    LayerNorm's population is unchanged and the function must not move.
    d, mse = delta(donor, 512, tokens, fused=True)
    ok = d < EXACT
    print(f"2x  d=512 fused, REAL   max|delta| {d:.6f}  MSE {mse:.6f}  "
          f"{'PASS' if ok else 'FAIL'} (theory demands 0)")
    if not ok:
        fails += 1
        print("  Growth is NOT function-preserving at an integer multiple, "
              "which is a bug in grow_net.py and not a property of the width.")

    # 2. The width being built. Approximate by construction; the point is to
    #    know by how much, on positions that exist.
    d384_real, mse_real = delta(donor, 384, tokens, fused=True)
    d384_noise, mse_noise = delta(donor, 384, noise, fused=True)
    print(f"1.5x d=384 fused, REAL  max|delta| {d384_real:.6f}  "
          f"MSE {mse_real:.6f}")
    print(f"1.5x d=384 fused, noise max|delta| {d384_noise:.6f}  "
          f"MSE {mse_noise:.6f}   <- what grow_net.py reports")
    print("  Approximate at 1.5x and expected to be: half the channels are "
          "duplicated,\n  so LayerNorm normalises over a different population. "
          "The gate is recovery\n  by step 10k, not this number.")

    if a.run:
        print(f"\nrecovery gate, runs/{a.run}:")
        fails += recovery(a.run, a.donor_loss)

    print(f"\n{'PASS' if not fails else f'FAIL ({fails})'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
