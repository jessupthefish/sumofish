#!/usr/bin/env python
"""Measure the difference. Two numbers, and they mean different things.

**A. Whole-loop (perft).** Both implementations run the entire
generate/make/recurse loop in their own language. This is the honest ceiling for
the ported portion: it is what Stage C buys, where the tree lives in Rust and
Python is called once per batch of leaves.

**B. Per-call through FFI.** Python drives the loop and crosses the boundary once
per position. This is what a movegen-only PyO3 port would give, i.e. the option
the council priced at an Amdahl ceiling of 1.8x, and it is deliberately measured
separately because quoting A while shipping B is how a speedup evaporates.

Neither number is Elo. Converting speed to Elo needs the exchange rate, which
this project retracted on 2026-07-29 and has not yet re-earned.
"""

from __future__ import annotations

import time

import chess

import sumofish_core as core

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
MIDGAME = "r1bq1r1k/1pp2pbp/p1np1np1/4p3/2P1P3/2NP1NP1/PP3PBP/R1BQR1K1 w - - 0 11"


def py_perft(board: chess.Board, depth: int) -> int:
    if depth == 0:
        return 1
    nodes = 0
    for mv in board.legal_moves:
        board.push(mv)
        nodes += py_perft(board, depth - 1)
        board.pop()
    return nodes


def whole_loop(fen: str, depth: int) -> None:
    b = chess.Board(fen)
    t = time.perf_counter()
    py_nodes = py_perft(b, depth)
    py_el = time.perf_counter() - t

    t = time.perf_counter()
    rs_nodes = core.perft(fen, depth)
    rs_el = time.perf_counter() - t

    assert py_nodes == rs_nodes, f"node count disagreement: {py_nodes} vs {rs_nodes}"
    print(
        f"  depth {depth}: {py_nodes:>9,} nodes | "
        f"python-chess {py_el:7.3f}s ({py_nodes/py_el/1e3:8.1f} knode/s) | "
        f"rust {rs_el:7.3f}s ({rs_nodes/rs_el/1e6:6.2f} Mnode/s) | "
        f"{py_el/rs_el:6.1f}x"
    )


def per_call(fens: list[str], reps: int = 3) -> None:
    py_boards = [chess.Board(f) for f in fens]
    rs_boards = [core.Board(f) for f in fens]

    best_py = min(
        (
            _time(lambda: [[m.uci() for m in b.legal_moves] for b in py_boards])
            for _ in range(reps)
        )
    )
    best_rs = min((_time(lambda: [b.legal_moves() for b in rs_boards]) for _ in range(reps)))

    n = len(fens)
    print(
        f"  {n} positions | "
        f"python-chess {best_py*1e6/n:7.2f} us/pos | "
        f"rust+FFI {best_rs*1e6/n:7.2f} us/pos | "
        f"{best_py/best_rs:6.1f}x"
    )


def _time(fn) -> float:
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def main() -> None:
    print(f"sumofish_core {core.__version__} vs python-chess {chess.__version__}\n")

    print("A. WHOLE LOOP -- generate + make + recurse, each in its own language")
    print("   (the Stage C ceiling: tree in Rust, Python called once per batch)")
    for fen, name, depth in [
        (STARTPOS, "startpos", 4),
        (KIWIPETE, "kiwipete", 4),
        (MIDGAME, "midgame", 4),
    ]:
        print(f"  -- {name}")
        whole_loop(fen, depth)

    print()
    print("B. PER CALL THROUGH FFI -- Python drives, one boundary crossing each")
    print("   (what a movegen-only port gives; Amdahl-capped around 1.8x overall)")
    import random

    rng = random.Random(7)
    fens = []
    b = chess.Board()
    while len(fens) < 2000:
        if b.is_game_over(claim_draw=False):
            b = chess.Board()
            continue
        fens.append(b.fen())
        b.push(rng.choice(list(b.legal_moves)))
    per_call(fens)

    print()
    print("Neither number is Elo. The exchange rate was retracted 2026-07-29.")


if __name__ == "__main__":
    main()
