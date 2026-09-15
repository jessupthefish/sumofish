#!/usr/bin/env python
"""Search batch size, judged on UNIQUE evaluations per second on the real nets.

LAB-NOTES (the batch-size entry) forbids choosing a batch on raw nps: rows in a
batch that land on the same leaf are the same position evaluated again. With
the live engine the forward pass is ~81% of a cycle and launch-bound, so a
bigger batch buys rows nearly free on the GPU (PLAN-2026-09-15, Phase 2) and
the question is how many of those rows are new.

For each batch size the engine is built as the bot builds it (compile + pad to
the batch, vloss_fix, reuse), warmed up, and then measured two ways:

  fresh  a fresh tree on each fixed position for --seconds
  deep   a --deep-seconds search, then a reroot two plies down the principal
         line and --seconds more. Live moves are 400k-visit searches into a
         reused tree, where collisions behave differently from a shallow one.

Per phase it reports evaluations/s (rows sent), unique/s (distinct positions
per batch, summed), and the duplicate share. A candidate must beat batch 64
on unique/s in BOTH phases before it earns a clock match; it does not ship on
this table alone.

NEEDS THE GPU TO ITSELF.

Usage:
    scripts/batch_sweep.py [--batches 64,96,128,192] [--seconds 8] [--deep-seconds 30]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sumofish.engines.loader import load_nets  # noqa: E402
from sumofish.rust_mcts import RustMCTS  # noqa: E402

POSITIONS = [
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6",
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8 g1f3 b8d7 a1c1 c7c6 f1d3 d5c4 d3c4 f6d5",
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8 h2h3",
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8 f1e2 e7e5 e1g1 b8c6 d4d5 c6e7",
]


class Counting:
    """Wraps the evaluator: rows sent and distinct positions per call."""

    def __init__(self, inner):
        self.inner = inner
        self.fused = getattr(inner, "fused", False)
        self.zero()

    def zero(self):
        self.rows = 0
        self.unique = 0
        self.calls = 0

    def __call__(self, fens, actions):
        self.rows += len(fens)
        self.unique += len(set(fens))
        self.calls += 1
        return self.inner(fens, actions)


def board_of(moves: str) -> chess.Board:
    b = chess.Board()
    for u in moves.split():
        b.push_uci(u)
    return b


def phase(m: RustMCTS, ev: Counting, work) -> dict:
    ev.zero()
    t = time.perf_counter()
    work()
    wall = time.perf_counter() - t
    return {
        "seconds": round(wall, 2),
        "evals_per_s": round(ev.rows / wall),
        "unique_per_s": round(ev.unique / wall),
        "dup_share": round(1 - ev.unique / ev.rows, 4) if ev.rows else None,
        "rows_per_call": round(ev.rows / ev.calls, 1) if ev.calls else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batches", default="64,96,128,192")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--deep-seconds", type=float, default=30.0)
    ap.add_argument("--value", default=str(ROOT / "runs/value.pt"))
    ap.add_argument("--policy", default=str(ROOT / "runs/policy.pt"))
    ap.add_argument("--out", default=str(ROOT / "runs/profiles" / time.strftime("batch-sweep-%Y%m%d-%H%M.json")))
    args = ap.parse_args()

    value, policy, info = load_nets(args.value, args.policy, device="cuda:0")
    results = {"value": args.value, "fused": info.get("fused"), "batches": {}}
    for B in [int(x) for x in args.batches.split(",")]:
        m = RustMCTS(value, policy=policy, simulations=100_000_000, batch=B,
                     compile_nets=True, pad_batches=True, vloss_fix=True, reuse=True)
        ev = Counting(m._evaluate)
        m._evaluate = ev
        # warm-up: compile, capture the CUDA graph at this batch's padded shape
        m.search(board_of(POSITIONS[0]), deadline=time.perf_counter() + 6.0)
        m.reset()

        def fresh():
            for mv in POSITIONS:
                m.reset()
                m.search(board_of(mv), deadline=time.perf_counter() + args.seconds)

        r_fresh = phase(m, ev, fresh)

        m.reset()
        base = board_of(POSITIONS[1])
        _, visits = m.search(base, deadline=time.perf_counter() + args.deep_seconds)
        best = max(visits.items(), key=lambda kv: kv[1])[0]
        after = base.copy(); after.push(best)
        _, v2 = m.search(after, deadline=time.perf_counter() + 1.0)   # one ply, like analysis
        reply = max(v2.items(), key=lambda kv: kv[1])[0]
        deep_board = after.copy(); deep_board.push(reply)

        def deep():
            m.search(deep_board, deadline=time.perf_counter() + args.seconds)

        r_deep = phase(m, ev, deep)
        r_deep["reused"] = m.reused
        results["batches"][B] = {"fresh": r_fresh, "deep": r_deep}
        print(f"batch {B:4}  fresh: {r_fresh['evals_per_s']:6,} ev/s {r_fresh['unique_per_s']:6,} uniq/s "
              f"dup {r_fresh['dup_share']:.1%} | deep (reused {m.reused:,}): {r_deep['evals_per_s']:6,} ev/s "
              f"{r_deep['unique_per_s']:6,} uniq/s dup {r_deep['dup_share']:.1%}", flush=True)
        del m, ev

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
