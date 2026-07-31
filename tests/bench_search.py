#!/usr/bin/env python
"""Measure Stage C: the whole search loop, Python vs Rust.

Same tree, same PUCT, same batch boundaries, same deterministic evaluator, and
(proved by `identity_search.py`) the same visit vector out. So the ratio here is
a clean measurement of the tree machinery, with the network held constant.

The mock evaluator is intentionally cheap. In the real engine the network is ~9%
of wall clock, so a benchmark that spent most of its time in torch would measure
torch and report it as a search speedup. Isolating the tree is the point; the
network cost is then added back by Amdahl, which is done at the end.
"""

from __future__ import annotations

import argparse
import sys
import time
import zlib
from pathlib import Path

import chess
import numpy as np

import sumofish_core as core

ROOT = Path(__file__).resolve().parent.parent
for candidate in (ROOT, ROOT.parent / "sumofish"):
    if (candidate / "sumofish" / "mcts.py").exists():
        sys.path.insert(0, str(candidate))
        break
from sumofish.mcts import MCTS, _softmax_over_legal  # noqa: E402

from sumofish.tokenizer import ACTION_BY_MOVE_KEY, move_key  # noqa: E402

from identity_search import (  # noqa: E402
    MockPolicy,
    MockValuePolicy,
    logit_row,
    mock_value,
    rust_evaluate,
)

POSITIONS = [
    ("startpos", []),
    ("open middlegame", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]),
    (
        "sharp middlegame",
        ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "d7d5", "b1c3", "c7c5", "c4d5", "f6d5"],
    ),
]


def callback_cost(moves: list[str], sims: int) -> float:
    """Time the evaluator alone, on the positions a search of this size visits.

    Both implementations call the SAME evaluator, so its cost is common-mode and
    must be subtracted before the tree can be compared. Without this the
    measurement is dominated by the mock and reports whatever ratio the mock
    happens to imply -- which is how the first version of this benchmark
    concluded 1.4x for a tree that is an order of magnitude faster.
    """
    board = chess.Board()
    for u in moves:
        board.push(chess.Move.from_uci(u))
    fens, actions = [], []
    seen = board.copy()
    import itertools
    for _ in range(sims):
        legal = list(seen.legal_moves)
        if not legal:
            seen = board.copy()
            continue
        fens.append(seen.fen())
        actions.append([ACTION_BY_MOVE_KEY[move_key(m)] for m in legal])
        seen.push(legal[len(fens) % len(legal)])
        if seen.is_game_over(claim_draw=False):
            seen = board.copy()
    t0 = time.perf_counter()
    for i in range(0, len(fens), 64):
        rust_evaluate(fens[i : i + 64], actions[i : i + 64])
    return time.perf_counter() - t0


def bench(moves: list[str], sims: int, batch: int) -> tuple[float, float]:
    py_board = chess.Board()
    for u in moves:
        py_board.push(chess.Move.from_uci(u))
    py_mcts = MCTS(
        MockValuePolicy(), policy=MockPolicy(), c_puct=2.0, c_puct_base=19652.0,
        c_puct_init=1.25, simulations=sims, batch=batch, fpu=-0.2, reuse=False,
    )
    t = time.perf_counter()
    py_mcts.search(py_board)
    py_el = time.perf_counter() - t

    rs_pos = core.Position()
    for u in moves:
        rs_pos.push_uci(u)
    rs_mcts = core.Mcts(
        c_puct=2.0, c_puct_base=19652.0, c_puct_init=1.25, fpu=-0.2, batch=batch
    )
    t = time.perf_counter()
    rs_mcts.search(rs_pos, sims, rust_evaluate)
    rs_el = time.perf_counter() - t

    return py_el, rs_el


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sims", type=int, default=1600)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    print(f"{args.sims} simulations, batch {args.batch}, deterministic mock evaluator\n")
    print("  END TO END (evaluator included in both, so common-mode)")
    ratios, tree_ratios = [], []
    for name, moves in POSITIONS:
        py_el, rs_el = bench(moves, args.sims, args.batch)
        cb = callback_cost(moves, args.sims)
        # Subtract the shared evaluator to isolate the tree. Clamped: if the
        # subtraction goes non-positive the evaluator is swamping the signal and
        # the number would be noise dressed as a result.
        py_tree = max(py_el - cb, 1e-6)
        rs_tree = max(rs_el - cb, 1e-6)
        ratios.append(py_el / rs_el)
        print(
            f"  {name:18s} python {py_el:7.3f}s | rust {rs_el:7.3f}s | "
            f"{py_el/rs_el:5.1f}x   (evaluator {cb:6.3f}s of both)"
        )
        if rs_el - cb > 0.2 * cb:
            tree_ratios.append(py_tree / rs_tree)

    print()
    print("  TREE ONLY (evaluator subtracted)")
    if tree_ratios:
        for (name, _), r in zip(POSITIONS, tree_ratios):
            print(f"  {name:18s} {r:6.0f}x")
        tree = sum(tree_ratios) / len(tree_ratios)
        print(f"\n  tree-only speedup: ~{tree:.0f}x")
    else:
        tree = None
        print("  the evaluator dominates Rust's total; tree ratio not resolvable here")

    mean = sum(ratios) / len(ratios)
    print()
    print("  WHAT THIS IMPLIES FOR THE ENGINE")
    print("  The Python engine profile: network 9% of wall clock, everything else 91%.")
    print("  Under this port, what stays in Python is the network plus the prior")
    print("  softmax (which must, to keep bit-identity). Everything else moves.")
    print()
    print(f"  measured end-to-end here, with a mock of comparable cost: {mean:.1f}x")
    if tree:
        overall = 1.0 / (0.09 + 0.02 + 0.89 / tree)
        print(f"  extrapolated with the real network at 9% and softmax at ~2%:")
        print(f"    1/(0.09 + 0.02 + 0.89/{tree:.0f}) = {overall:.1f}x")
    print()
    print("  NOT Elo. The exchange rate was retracted 2026-07-29 and is unearned.")


if __name__ == "__main__":
    main()
