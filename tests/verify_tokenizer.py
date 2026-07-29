#!/usr/bin/env python
"""Oracle 7: the Rust tokenizer is byte-identical to both Python entry points.

Two comparisons, not one, because the two Python functions have different
standing and both matter:

  * `tokenize(fen)` is the one verified byte-exact against DeepMind's published
    implementation over 12,000 real positions. Every published number in the
    project depends on it. It is the ground truth.
  * `tokenize_board(board)` is the fast path the engine actually calls, checked
    against `tokenize` over 100,836 positions. It is what must be matched in
    practice.

They agree, so matching either should mean matching both -- and checking both
anyway is how you find out if that stops being true.

Unlike the prior softmax, bit-exactness here is reachable by construction: it is
integer and character work with no rounding, no summation order, and no vendor
SIMD kernel. So this test should be, and is, green.

Also checks the vocabulary itself, since `CHARACTERS` is transcribed into Rust and
its ORDER is what the model learnt -- a reordering would silently change every
input the network has ever seen.

Usage:
    tests/verify_tokenizer.py --positions 40000
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import chess
import numpy as np

import sumofish_core as core

ROOT = Path(__file__).resolve().parent.parent
for candidate in (ROOT, ROOT.parent / "chess-gpu"):
    if (candidate / "chessgpu" / "tokenizer.py").exists():
        sys.path.insert(0, str(candidate))
        break
from chessgpu.tokenizer import (  # noqa: E402
    CHARACTERS,
    SEQUENCE_LENGTH,
    VOCAB_SIZE,
    tokenize,
    tokenize_board,
)

# Positions that exercise the fields most likely to be wrong.
SUITE = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # ep available, both colours -- the field must be WRITTEN
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "rnbqkbnr/pppp1ppp/8/8/3pP3/8/PPP2PPP/RNBQKBNR b KQkq e3 0 3",
    # ep square SET but not legal -- the field must be BLANK. The Stage A trap.
    "8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1",
    # partial castling rights, in every combination of one
    "r3k2r/8/8/8/8/8/8/R3K2R w K - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w Q - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w k - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w q - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1",
    # move counters at every digit width
    "8/8/4k3/8/8/3K4/8/6R1 w - - 0 1",
    "8/8/4k3/8/8/3K4/8/6R1 w - - 9 9",
    "8/8/4k3/8/8/3K4/8/6R1 w - - 99 99",
    "8/8/4k3/8/8/3K4/8/6R1 w - - 100 999",
    # a black bishop, whose token is the shared 'b'
    "8/8/4k3/5b2/8/3K4/8/8 w - - 0 1",
]


def corpus(limit: int) -> list[str]:
    for root in (ROOT, ROOT.parent / "chess-gpu"):
        bag = root / "data/train/state_value_data.bag"
        if not bag.exists():
            continue
        sys.path.insert(0, str(root))
        try:
            from chessgpu.bagz import BagReader, decode_state_value
        except Exception:
            break
        reader = BagReader(str(bag))
        n = len(reader)
        step = max(1, n // limit)
        out = []
        for i in range(0, min(n, step * limit), step):
            try:
                out.append(decode_state_value(reader[i])[0])
            except Exception:
                continue
            if len(out) >= limit:
                break
        if out:
            print(f"  sampled {len(out)} positions from the ChessBench bag")
            return out
    print("  no bag; using random legal play")
    rng = random.Random(1234)
    out, b = [], chess.Board()
    while len(out) < limit:
        if b.is_game_over(claim_draw=False):
            b = chess.Board()
            continue
        out.append(b.fen())
        b.push(rng.choice(list(b.legal_moves)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", type=int, default=40000)
    ap.add_argument("--max-report", type=int, default=5)
    args = ap.parse_args()

    print("vocabulary")
    problems = []
    if VOCAB_SIZE != 31 or SEQUENCE_LENGTH != 77:
        problems.append(f"shape changed: vocab={VOCAB_SIZE} seq={SEQUENCE_LENGTH}")
    if "".join(CHARACTERS) != "0123456789abcdefghpnrkqPBNRQKw.":
        problems.append(
            "CHARACTERS ORDER CHANGED. The Rust copy is a transcription and the "
            "order is what the model learnt; every network input would shift. "
            f"python={''.join(CHARACTERS)!r}"
        )
    if problems:
        for p in problems:
            print(f"  FAIL {p}")
        return 1
    print("  OK: 31 characters in the expected order, sequence length 77")

    fens = SUITE + corpus(args.positions)
    print(f"checking {len(fens)} positions against BOTH Python entry points")

    bad_tokenize = bad_board = 0
    reported = 0
    for fen in fens:
        board = chess.Board(fen)
        want_fen = tokenize(board.fen())          # ground truth
        want_board = tokenize_board(board)         # what the engine calls
        got = np.frombuffer(core.tokenize(fen), dtype=np.uint8)

        ok_fen = np.array_equal(got, want_fen)
        ok_board = np.array_equal(got, want_board)
        if not ok_fen:
            bad_tokenize += 1
        if not ok_board:
            bad_board += 1
        if (not ok_fen or not ok_board) and reported < args.max_report:
            reported += 1
            diffs = [
                (i, int(g), int(w))
                for i, (g, w) in enumerate(zip(got, want_board))
                if g != w
            ]
            fields = {65: "castling", 69: "ep", 71: "halfmove", 74: "fullmove"}
            print(f"  FAIL {fen}")
            for i, g, w in diffs[:6]:
                where = "board" if 1 <= i <= 64 else (
                    "side" if i == 0 else next(
                        (n for at, n in fields.items() if at <= i < at + 4), "?"
                    )
                )
                print(f"    [{i}] ({where}) rust={g} python={w}")

    print(f"  vs tokenize(fen):        {len(fens)-bad_tokenize}/{len(fens)} identical")
    print(f"  vs tokenize_board(board): {len(fens)-bad_board}/{len(fens)} identical")

    # The batch entry point must agree with the single one, concatenated.
    sample = fens[:512]
    buf = np.frombuffer(core.tokenize_batch(sample), dtype=np.uint8).reshape(-1, 77)
    singles = np.stack([np.frombuffer(core.tokenize(f), dtype=np.uint8) for f in sample])
    batch_ok = np.array_equal(buf, singles)
    print(f"  tokenize_batch == stacked tokenize: {batch_ok}")

    print()
    if bad_tokenize or bad_board or not batch_ok:
        print("FAIL")
        return 1
    print("OK: Rust tokenizer is byte-identical to both Python entry points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
