#!/usr/bin/env python
"""Oracle 3: `push` parity, ply by ply, against `python-chess`.

Perft is a strong check on move generation and a WEAK check on `push`. It
validates the bookkeeping only where it changes legality, so these all pass
perft while being wrong:

  * `halfmove_clock` reset on the wrong condition -- invisible to perft, and it
    decides fifty-move draws, which `rules.terminal_value` reads.
  * `fullmove_number` off by one -- invisible to perft, and it is one of the
    77 tokens the network reads.
  * `ep_square` set when it should not be -- often invisible to perft, because
    an ep capture that no pawn can make changes no node count. It IS visible in
    the FEN, and therefore in the network input.
  * castling rights cleared too late -- perft catches this only at a depth where
    the illegal castle actually appears.

So: walk random legal games, push the SAME move into both implementations, and
compare the full FEN after every single ply. A divergence anywhere fails.

Usage:
    tests/push_parity.py --games 3000
"""

from __future__ import annotations

import argparse
import random
import sys

import chess

import sumofish_core as core


def walk(rng: random.Random, max_plies: int = 300) -> tuple[int, str | None]:
    """Play one random game in lockstep. Returns (plies, failure or None)."""
    ref = chess.Board()
    rs = core.Board()

    plies = 0
    while plies < max_plies:
        # Both must agree on the move list before either moves, or the
        # comparison below is meaningless.
        expected = [m.uci() for m in ref.legal_moves]
        got = rs.legal_moves()
        if got != expected:
            return plies, (
                f"legal moves diverged at ply {plies}\n"
                f"  fen:    {ref.fen()}\n"
                f"  rust:   {' '.join(got)}\n"
                f"  python: {' '.join(expected)}"
            )
        if not expected:
            return plies, None

        uci = rng.choice(expected)
        ref.push(chess.Move.from_uci(uci))
        rs.push_uci(uci)
        plies += 1

        # The whole point: compare every field, every ply.
        if rs.fen() != ref.fen():
            return plies, (
                f"FEN diverged after {uci} at ply {plies}\n"
                f"  rust:   {rs.fen()}\n"
                f"  python: {ref.fen()}\n"
                f"  {_field_diff(rs.fen(), ref.fen())}"
            )

        if ref.is_game_over(claim_draw=False):
            return plies, None

    return plies, None


def _field_diff(a: str, b: str) -> str:
    names = ["board", "turn", "castling", "ep", "halfmove", "fullmove"]
    pa, pb = a.split(), b.split()
    bad = [
        f"{names[i]}: rust={pa[i]!r} python={pb[i]!r}"
        for i in range(min(len(pa), len(pb)))
        if pa[i] != pb[i]
    ]
    return "differing fields -> " + "; ".join(bad)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-report", type=int, default=3)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    total_plies = 0
    failures = 0

    for g in range(args.games):
        plies, problem = walk(rng)
        total_plies += plies
        if problem:
            failures += 1
            if failures <= args.max_report:
                print(f"\nFAIL game {g}: {problem}")
        if g and g % 500 == 0:
            print(f"  {g} games, {total_plies:,} plies, {failures} failing")

    print()
    if failures:
        print(f"FAIL: {failures} of {args.games} games diverged")
        return 1
    print(
        f"OK: {args.games:,} games, {total_plies:,} plies, "
        f"every FEN field identical at every ply"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
