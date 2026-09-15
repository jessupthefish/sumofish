#!/usr/bin/env python
"""`_softmax_over_legal_batch` against the per-row softmax on REAL logits.

`tests/verify_softmax_batch.py` proves bit-identity on synthetic float32 rows.
This closes the remaining gap before the batched version replaces the per-row
call in `make_evaluator.evaluate`: it runs real searches on the deployed nets
exactly as the engine builds them, records every (logit row, legal actions)
batch the evaluator softmaxes, and compares the batched function bitwise on
those same rows. Real logits are what the prior actually sees: their scale,
their ties and whatever `.float()` after autocast leaves in them.

NEEDS THE GPU: do not run it while a match or the bot is using the card.

Usage:
    tests/verify_softmax_real.py [--seconds 6] [--positions 6]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sumofish.rust_mcts as rm  # noqa: E402
from sumofish.engines.loader import load_nets  # noqa: E402

POSITIONS = [
    "",
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6",
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8 g1f3 b8d7 a1c1 c7c6 f1d3 d5c4 d3c4 f6d5",
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8 h2h3",
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8 f1e2 e7e5 e1g1 b8c6 d4d5 c6e7",
    "e2e4 e7e6 d2d4 d7d5 b1d2 c7c5 e4d5 e6d5 g1f3 b8c6 f1b5 f8d6 d4c5 d6c5 e1g1 g8e7 d2b3 c5d6",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--positions", type=int, default=len(POSITIONS))
    ap.add_argument("--value", default=str(ROOT / "runs/value.pt"))
    ap.add_argument("--policy", default=str(ROOT / "runs/policy.pt"))
    args = ap.parse_args()

    value, policy, info = load_nets(args.value, args.policy, device="cuda:0")
    # Built as the bot builds it: compile + pad, vloss_fix, batch 64.
    m = rm.RustMCTS(value, policy=policy, simulations=100_000_000, batch=64,
                    compile_nets=True, pad_batches=True, vloss_fix=True)

    captured: list[tuple[np.ndarray, list[list[int]]]] = []
    pending: dict[int, tuple[list[np.ndarray], list[list[int]]]] = {}
    real = rm._softmax_over_legal

    def recording(row, actions):
        # Rows arrive one call per row, in batch order, all views of one array.
        base = row.base if row.base is not None else row
        key = id(base)
        rows, acts = pending.setdefault(key, ([], []))
        rows.append(row.copy())
        acts.append(list(actions))
        return real(row, actions)

    rm._softmax_over_legal = recording
    try:
        for moves in POSITIONS[: args.positions]:
            board = chess.Board()
            for u in moves.split():
                board.push_uci(u)
            m.reset()
            t = time.perf_counter()
            m.search(board, deadline=t + args.seconds)
            for rows, acts in pending.values():
                captured.append((np.stack(rows), acts))
            pending.clear()
    finally:
        rm._softmax_over_legal = real

    batches = len(captured)
    rows_total = sum(len(a) for _, a in captured)
    bad = 0
    for rows, acts in captured:
        got = rm._softmax_over_legal_batch(rows, acts)
        want = [real(rows[i], acts[i]) for i in range(len(acts))]
        if any(np.asarray(g, dtype=np.float64).tobytes() != np.asarray(w, dtype=np.float64).tobytes()
               for g, w in zip(got, want, strict=True)):
            bad += 1
    print(f"{batches:,} real batches, {rows_total:,} rows from {args.positions} positions "
          f"({'fused' if info.get('fused') else 'two nets'}): {bad} batches differ")
    print("OK: batched softmax is bit-identical on real logits" if bad == 0 and batches > 0
          else "FAIL")
    return 0 if bad == 0 and batches > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
