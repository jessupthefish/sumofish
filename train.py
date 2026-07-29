#!/usr/bin/env python
"""Train SumoFish.

    .venv/bin/python train.py --preset 9M --steps 200000

Objective is exactly upstream's: cross-entropy on Stockfish's chosen move given
the position, at the final sequence position only. Everything else in the 77
tokens is context, never a prediction target.

Progress is measured on puzzle accuracy, not loss, because loss is not
comparable to anything published and puzzle accuracy is.

Compare against the right reference: the paper's ablation at fixed architecture
gives 83.3% for action-value, 77.5% for state-value, and 65.7% for behavioral
cloning. This script trains behavioral cloning, so 65.7% is the ceiling, not the
88.9%/2054 Elo figures, which belong to the action-value lineage.
"""

from __future__ import annotations

import argparse
import json
import os
import math
import signal
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from chessgpu.data import make_loader
from chessgpu.hlgauss import HLGauss
from chessgpu.hlgauss import loss as hl_loss
from chessgpu.evaluate import evaluate_puzzles, load_puzzles
from chessgpu.model import PRESETS, ChessTransformer, build
from chessgpu.policy import NeuralPolicy
from chessgpu.value_policy import ValuePolicy

ROOT = Path(__file__).resolve().parent


def lr_at(step: int, *, base_lr: float, warmup: int, total: int, min_frac: float = 0.1) -> float:
    """Linear warmup then cosine decay. Standard, and robust to a bad guess."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, progress)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (min_frac + (1 - min_frac) * cosine)


class EMA:
    """Exponential moving average of weights. Upstream keeps one at decay 0.99.

    Averaged weights are usually worth a little accuracy for free, and cost one
    extra copy of a model that is 9M parameters, so there is no reason not to.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_into(self, model: torch.nn.Module) -> None:
        model.load_state_dict({k: v for k, v in self.shadow.items()}, strict=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="9M", choices=list(PRESETS))
    ap.add_argument(
        "--target",
        default="behavioral_cloning",
        choices=["behavioral_cloning", "state_value"],
        help="what the model predicts. behavioral_cloning = which move Stockfish "
        "played (1968-way softmax). state_value = P(side to move wins), as an "
        "HL-Gauss histogram. The paper's ablation at identical architecture puts "
        "state-value at 77.5%% puzzle accuracy vs 65.7%% for behavioral cloning.",
    )
    ap.add_argument("--value-bins", type=int, default=64,
                    help="HL-Gauss bins; the paper's ablation is flat above 32")
    ap.add_argument("--data", default=None,
                    help="defaults to the bag matching --target")
    ap.add_argument("--val-data", default=None,
                    help="held-out bag; defaults to data/test/<target>_data.bag")
    ap.add_argument("--val-batches", type=int, default=32,
                    help="batches of held-out data per eval; 0 disables")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--causal", type=int, default=1, help="1 = upstream, 0 = bidirectional")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--eval-puzzles", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--run", default=None, help="run name; defaults to preset+causal")
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234,
                    help="weight init and data order; hold it fixed across the "
                         "arms of an ablation or the arms differ by more than "
                         "the thing being tested")
    ap.add_argument("--resume", default=None)
    ap.add_argument(
        "--init-from",
        default=None,
        help="warm start: copy every shape-compatible layer from another "
        "checkpoint, leaving the rest at random init. Changing the prediction "
        "target only changes the output layer -- 91 of 93 tensors are identical "
        "in shape between the 1968-way move head and the 64-bin value head -- so "
        "the entire transformer body transfers and does not need relearning.",
    )
    ap.add_argument(
        "--auto-resume",
        action="store_true",
        help="resume from runs/<run>/latest.pt if it exists; makes the process "
        "restartable by a supervisor without losing the run",
    )
    args = ap.parse_args()

    is_value = args.target == "state_value"
    if args.data is None:
        args.data = str(ROOT / f"data/train/{args.target}_data.bag")
    if args.val_data is None:
        args.val_data = str(ROOT / f"data/test/{args.target}_data.bag")

    run = args.run or f"{args.preset}-{'sv' if is_value else 'bc'}"
    out = ROOT / "runs" / run
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    # The stall watchdog must know WHICH run is live. It used to hardcode
    # 'runs/9M-causal/log.jsonl', which meant the first run with a different
    # --run name would read a stale, frozen log and restart a perfectly healthy
    # process every 30 minutes forever.
    (ROOT / "runs" / "active.json").write_text(
        json.dumps({"run": run, "log": str(out / "log.jsonl"), "pid": os.getpid()})
    )

    device = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Before `build`, so the weights themselves are reproducible and not just
    # the batches. Two runs differing only in --causal should differ only in
    # --causal.
    torch.manual_seed(args.seed)

    # The only architectural difference between the two targets is the width of
    # the output layer: 1968 moves, or `--value-bins` histogram buckets.
    out_size = args.value_bins if is_value else None
    build_kwargs = {"causal": bool(args.causal)}
    if out_size:
        build_kwargs["output_size"] = out_size
    model = build(args.preset, **build_kwargs).to(device)

    hl = HLGauss(bins=args.value_bins, device=device) if is_value else None
    print(f"run {run}: {model.num_parameters():,} parameters")
    print(f"config: {asdict(model.cfg)}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    ema = EMA(model, decay=args.ema_decay)
    start_step = 0
    best = -float("inf")
    # Which metric `best` is a value of. Checkpointed, so a resume across a
    # change of selection rule resets rather than comparing incomparables.
    best_metric = ""

    if args.auto_resume and not args.resume:
        candidate = out / "latest.pt"
        if candidate.exists():
            args.resume = str(candidate)
            print(f"auto-resume: found {candidate}")
        else:
            print("auto-resume: no checkpoint yet, starting fresh")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        ema.shadow = {k: v.to(device) for k, v in ckpt["ema"].items()}
        start_step = ckpt["step"]
        # Older checkpoints predate this field; 0.0 is the old (buggy) behaviour.
        best = ckpt.get("best", -float("inf"))
        best_metric = ckpt.get("best_metric", "puzzle_acc")
        print(f"resumed from {args.resume} at step {start_step:,}, best={best:.4f}")

    if args.init_from and not args.resume:
        donor = torch.load(args.init_from, map_location=device, weights_only=False)
        src = donor.get("ema") or donor["model"]
        own = model.state_dict()
        taken, skipped = [], []
        for k, v in src.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v.to(own[k].dtype)
                taken.append(k)
            else:
                skipped.append(k)
        model.load_state_dict(own)
        ema = EMA(model, decay=args.ema_decay)   # rebuild around the new weights
        print(f"warm start from {args.init_from} (step {donor.get('step')}): "
              f"{len(taken)} tensors copied, {len(skipped)} left random {skipped}")

    train_step = model
    if args.compile:
        train_step = torch.compile(model)

    loader = make_loader(
        args.data,
        policy=args.target,
        batch_size=args.batch_size,
        num_workers=args.workers,
        seed=args.seed + start_step,
    )
    batches = iter(loader)

    puzzles = load_puzzles(ROOT / "data/puzzles.csv", limit=args.eval_puzzles)

    stopping = False

    def on_signal(signum, frame):  # noqa: ARG001
        nonlocal stopping
        print("\nsignal received; will checkpoint and exit after this step")
        stopping = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    def save(step: int, tag: str = "latest") -> Path:
        """Atomic checkpoint write.

        A torn 136MB write is unrecoverable: --auto-resume torch.loads it,
        raises, systemd retries, trips StartLimitBurst, and the unit lands in
        `failed` where the stall watchdog explicitly declines to look. Write to
        a temp file and os.replace, which is atomic on one filesystem and also
        means a concurrent reader keeps the old inode open.
        """
        path = out / f"{tag}.pt"
        tmp = out / f"{tag}.pt.tmp"
        torch.save(
            {
                "step": step,
                "model": model.state_dict(),
                "ema": ema.shadow,
                "opt": opt.state_dict(),
                "cfg": asdict(model.cfg),
                "args": vars(args),
                # Must round-trip, or a resume resets it to 0.0 and the next
                # eval unconditionally overwrites best.pt with a worse model.
                "best": best,
                "best_metric": best_metric,
            },
            tmp,
        )
        os.replace(tmp, path)
        return path

    # Held-out loss. The run logged train loss and puzzle accuracy and nothing
    # else, which cannot separate "the model has stopped learning" from "the
    # model has started memorising" -- and the 9M state-value run needed exactly
    # that distinction when its puzzle curve went flat while its train loss kept
    # falling. (It was capacity: 307M samples is under one epoch of a 530M
    # position bag, so there was nothing to memorise. That was an argument from
    # arithmetic, not a measurement, and it gets weaker at 136M.)
    #
    # Deliberately deterministic: one worker, a shuffle buffer of one, a fresh
    # iterator each time. The same held-out positions in the same order at every
    # eval, so two numbers from different steps differ because the model did.
    val_path = Path(args.val_data)
    if args.val_batches > 0 and not val_path.exists():
        print(f"no held-out bag at {val_path}; validation disabled")

    def validate(eval_model: ChessTransformer) -> float | None:
        if args.val_batches <= 0 or not val_path.exists():
            return None
        eval_model.eval()
        loader = make_loader(
            args.val_data,
            policy=args.target,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle_buffer=1,
            seed=0,
            infinite=False,
        )
        total, batches_seen = 0.0, 0
        with torch.no_grad():
            for tokens, targets in loader:
                if batches_seen >= args.val_batches:
                    break
                tokens = tokens.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = eval_model(tokens)
                # The same objective the run is trained on, so the two numbers
                # are directly comparable. A gap that opens between them is the
                # signal this exists to catch.
                total += float(
                    hl_loss(logits.float(), hl.targets(targets))
                    if is_value
                    else F.cross_entropy(logits.float(), targets)
                )
                batches_seen += 1
        return total / max(1, batches_seen)

    log_path = out / "log.jsonl"
    running = 0.0
    seen = 0
    t0 = time.perf_counter()

    for step in range(start_step, args.steps):
        lr = lr_at(step, base_lr=args.lr, warmup=args.warmup, total=args.steps)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(args.accum):
            tokens, actions = next(batches)
            tokens = tokens.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = train_step(tokens)
            if is_value:
                # Soft cross-entropy against a Gaussian smeared over the bins.
                # Not MSE: Farebrother et al. measure HL-Gauss > C51 > MSE, and
                # plain two-hot binning as WORSE than MSE, so the choice of
                # binning scheme is load-bearing.
                loss = hl_loss(logits.float(), hl.targets(actions)) / args.accum
            else:
                # cross_entropy on raw logits, not nll_loss on log_softmax: same
                # value, fused, and it avoids materializing an fp32 [B, 1968].
                loss = F.cross_entropy(logits.float(), actions) / args.accum
            loss.backward()
            total_loss += loss.item()
            seen += tokens.shape[0]

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        ema.update(model)

        running += total_loss

        if (step + 1) % args.log_every == 0:
            dt = time.perf_counter() - t0
            avg = running / args.log_every
            rec = {
                "step": step + 1,
                "loss": round(avg, 5),
                "ppl": round(math.exp(min(avg, 20)), 2),
                "lr": lr,
                "grad_norm": round(float(grad_norm), 3),
                "samples_per_s": round(seen / dt),
                "positions": seen,
            }
            print(
                f"step {rec['step']:>7,}  loss {rec['loss']:.4f}  ppl {rec['ppl']:>8.1f}  "
                f"lr {lr:.2e}  gn {rec['grad_norm']:>6.2f}  {rec['samples_per_s']:>7,}/s"
            )
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            running = 0.0
            seen = 0
            t0 = time.perf_counter()

        if (step + 1) % args.eval_every == 0 or stopping:
            eval_model = build(args.preset, **build_kwargs).to(device)
            ema.copy_into(eval_model)
            # Validate the EMA weights, not the raw ones: EMA is what gets
            # evaluated, promoted and played, so a held-out number measured on
            # anything else would describe a model that never ships.
            val_loss = validate(eval_model)
            evaluator = (
                ValuePolicy(eval_model, HLGauss(bins=args.value_bins), device=device)
                if is_value
                else NeuralPolicy(eval_model, device=device)
            )
            result = evaluate_puzzles(evaluator, puzzles)
            val_note = f"  val {val_loss:.4f}" if val_loss is not None else ""
            print(f"  [eval] step {step+1:,}  puzzles {result}{val_note}  (BC ceiling ~0.657; 0.889 is the action-value model)")
            rec = {"step": step + 1, "puzzle_acc": result.accuracy}
            if val_loss is not None:
                rec["val_loss"] = round(val_loss, 5)
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            # Select on held-out loss when there is one, and only fall back to
            # puzzle accuracy when there is not.
            #
            # Puzzle accuracy has a sigma of 1.5 points at n=1000, and taking
            # the maximum over twenty evals of a metric that noisy is biased
            # upward by roughly two sigma: best.pt is whichever eval got lucky.
            # Measured on the finished 9M state-value run, final.pt (step 300k)
            # scores 2.1438 held out against best.pt's (step 280k) 2.1459, so
            # the checkpoint this rule promoted -- and which is deployed to the
            # live bot -- is the worse of the two.
            #
            # Higher is better either way, so the loss enters negated, and the
            # metric's name is checkpointed so a resume cannot compare an
            # accuracy against a negative loss and save on the first eval.
            metric = "neg_val_loss" if val_loss is not None else "puzzle_acc"
            score = -val_loss if val_loss is not None else result.accuracy
            if metric != best_metric:
                print(f"  [eval] selection metric is now {metric}; resetting best")
                best, best_metric = -float("inf"), metric
            if score > best:
                best = score
                save(step + 1, "best")
                print(f"  [eval] new best on {metric}, saved")
            del eval_model
            torch.cuda.empty_cache()
            model.train()

        if (step + 1) % args.ckpt_every == 0 or stopping:
            save(step + 1)

        if stopping:
            print(f"stopped at step {step+1:,}")
            break

    if not stopping:
        save(args.steps, "final")
    print("done")


if __name__ == "__main__":
    main()
