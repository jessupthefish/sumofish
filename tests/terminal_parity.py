#!/usr/bin/env python
"""Oracle 4: terminal detection parity against `chessgpu/rules.py`.

Stage B ports `rules.terminal_value` and everything under it: mate, stalemate,
insufficient material, the fifty-move rule, and threefold repetition over the
undo stack. This checks the port against the real thing, ply by ply.

Repetition is the part that needs help. Random legal play essentially never
reaches a threefold -- it wanders off instead -- so the corpus here is built in
four deliberate parts:

  1. **Cycles.** Four-move loops (Nf3 Nf6 Ng1 Ng8) that return to the same
     position, repeated until it has occurred three times. This is the only way
     to exercise the backward walk at all.
  2. **Irreversible interrupts.** The same cycles with a pawn move or a capture
     spliced in, because the walk must STOP at an irreversible move rather than
     counting past it. A port that forgets this reports draws that do not exist.
  3. **Castling-rights interrupts.** A king or rook move changes the
     transposition key even when the pieces return, so the position is NOT the
     same position. Easy to get wrong, and invisible without this case.
  4. **Insufficient material**, enumerated: K-K, KN-K, KB-K, KNN-K, KB-KB on
     the same and on opposite colours, and KP-K as the negative control.

Usage:
    tests/terminal_parity.py --games 2000
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import chess

import sumofish_core as core

ROOT = Path(__file__).resolve().parent.parent

# rules.py imports nothing but `chess`, so it loads fine in this worktree's venv
# without torch. Pull it from the main checkout, which is where the engine lives.
for candidate in (ROOT, ROOT.parent / "chess-gpu"):
    if (candidate / "chessgpu" / "rules.py").exists():
        sys.path.insert(0, str(candidate))
        break
from chessgpu.rules import terminal_value as py_terminal_value  # noqa: E402

# --- material cases, with the expected verdict from python-chess itself ------
MATERIAL = [
    "8/8/4k3/8/8/3K4/8/8 w - - 0 1",           # K vs K
    "8/8/4k3/8/8/3K4/8/6N1 w - - 0 1",         # KN vs K
    "8/8/4k3/8/8/3K4/8/6B1 w - - 0 1",         # KB vs K
    "8/8/4k3/8/8/3K4/8/5NN1 w - - 0 1",        # KNN vs K
    "6b1/8/4k3/8/8/3K4/8/6B1 w - - 0 1",       # KB vs KB, check the colours
    "5b2/8/4k3/8/8/3K4/8/6B1 w - - 0 1",       # KB vs KB, other colour
    "8/8/4k3/8/8/3K4/4P3/8 w - - 0 1",         # KP vs K: NOT insufficient
    "8/8/4k3/8/8/3K4/8/6R1 w - - 0 1",         # KR vs K: NOT insufficient
    # Mate and stalemate, so the first branch of terminal_value is exercised.
    "7k/5QK1/8/8/8/8/8/8 b - - 0 1",           # checkmate
    "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",          # stalemate
    # Fifty-move: halfmove_clock at and just below the boundary.
    "8/8/4k3/8/8/3K4/8/6R1 w - - 99 60",
    "8/8/4k3/8/8/3K4/8/6R1 w - - 100 60",
]

CYCLES = [
    # (starting fen, the reversible 4-ply loop)
    (None, ["g1f3", "g8f6", "f3g1", "f6g8"]),
    (None, ["b1c3", "b8c6", "c3b1", "c6b8"]),
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", ["a1b1", "a8b8", "b1a1", "b8a8"]),
]


def compare(pos: core.Position, ref: chess.Board, where: str) -> str | None:
    checks = [
        ("terminal_value", pos.terminal_value(), py_terminal_value(ref)),
        ("is_repetition(3)", pos.is_repetition(3), ref.is_repetition(3)),
        ("is_repetition(2)", pos.is_repetition(2), ref.is_repetition(2)),
        ("insufficient", pos.is_insufficient_material(), ref.is_insufficient_material()),
        ("is_check", pos.is_check(), ref.is_check()),
        ("halfmove", pos.halfmove_clock, ref.halfmove_clock),
        ("fen", pos.fen(), ref.fen()),
    ]
    for name, got, exp in checks:
        if got != exp:
            return f"{where}: {name} rust={got!r} python={exp!r}\n    fen: {ref.fen()}"
    return None


def run_cycles() -> tuple[list[str], int]:
    """Deliberate repetitions, with and without irreversible interrupts.

    Returns (failures, threefolds_reached). The caller FAILS the run if the
    count is zero: a repetition test that never reaches a repetition passes
    trivially and is worse than no test, because it reports coverage it does
    not have.
    """
    failures = []
    threefolds = 0
    for fen, loop in CYCLES:
        for variant in ("plain", "pawn_interrupt", "castle_interrupt"):
            ref = chess.Board(fen) if fen else chess.Board()
            pos = core.Position(fen) if fen else core.Position()

            seq = list(loop) * 3
            if variant == "pawn_interrupt":
                # A pawn move mid-way: the backward walk must stop here.
                if ref.piece_at(chess.E2) or (fen is None):
                    seq = list(loop) + ["e2e4", "e7e5"] + list(loop) * 2
                else:
                    continue
            elif variant == "castle_interrupt":
                if fen is None:
                    continue
                # Rook moves change castling rights, so the key differs even
                # after the rook returns.
                seq = ["h1g1", "h8g8", "g1h1", "g8h8"] * 3

            for i, uci in enumerate(seq):
                if chess.Move.from_uci(uci) not in ref.legal_moves:
                    break
                ref.push(chess.Move.from_uci(uci))
                pos.push_uci(uci)
                if ref.is_repetition(3):
                    threefolds += 1
                problem = compare(pos, ref, f"cycle[{fen or 'startpos'}/{variant}] ply {i+1}")
                if problem:
                    failures.append(problem)
                    break
            # Unwind, checking parity on the way back up: pop() must restore
            # everything the forward walk changed, including the history.
            while pos.ply > 0:
                pos.pop()
                ref.pop()
                problem = compare(pos, ref, f"cycle[{fen or 'startpos'}/{variant}] unwind")
                if problem:
                    failures.append(problem)
                    break
    return failures, threefolds


def run_material() -> list[str]:
    failures = []
    for fen in MATERIAL:
        ref = chess.Board(fen)
        pos = core.Position(fen)
        problem = compare(pos, ref, "material")
        if problem:
            failures.append(problem)
    return failures


def run_random(games: int, seed: int) -> tuple[int, int, list[str]]:
    """Random games, biased toward reversible moves so repetitions occur."""
    rng = random.Random(seed)
    failures = []
    plies = 0
    reps = 0
    for _ in range(games):
        ref = chess.Board()
        pos = core.Position()
        for _ in range(220):
            moves = list(ref.legal_moves)
            if not moves:
                break
            # Prefer quiet moves: pawn moves and captures reset the clock and
            # make repetition impossible, so unbiased play never reaches one.
            quiet = [
                m
                for m in moves
                if not ref.is_capture(m) and ref.piece_type_at(m.from_square) != chess.PAWN
            ]
            pool = quiet if (quiet and rng.random() < 0.985) else moves
            mv = rng.choice(pool)
            ref.push(mv)
            pos.push_uci(mv.uci())
            plies += 1
            if ref.is_repetition(3):
                reps += 1
            problem = compare(pos, ref, "random")
            if problem:
                failures.append(problem)
                break
            if ref.is_game_over(claim_draw=False) or ref.halfmove_clock >= 120:
                break
    return plies, reps, failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-report", type=int, default=5)
    args = ap.parse_args()

    all_failures: list[str] = []

    print("  material and mate/stalemate/fifty-move cases...")
    all_failures += run_material()

    print("  deliberate repetition cycles, with interrupts and unwind...")
    cycle_fails, threefolds = run_cycles()
    all_failures += cycle_fails
    print(f"    reached a threefold {threefolds} times")
    if threefolds == 0:
        print("FAIL: the cycle corpus never reached a threefold -- the "
              "repetition path is untested and this run proves nothing")
        return 1

    print(f"  {args.games} quiet-biased random games...")
    plies, reps, fails = run_random(args.games, args.seed)
    all_failures += fails
    print(f"    {plies:,} plies, {reps:,} of them at a threefold")

    print()
    if all_failures:
        for f in all_failures[: args.max_report]:
            print(f"FAIL {f}")
        print(f"\nFAIL: {len(all_failures)} disagreements")
        return 1
    print("OK: terminal_value, repetition, material, check and FEN all agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
