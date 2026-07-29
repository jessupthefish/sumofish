#!/usr/bin/env python
"""Oracle 6: what is and is not reproducible about the prior softmax.

Moving `_softmax_over_legal` into Rust was the highest-value remaining item on the
port's roadmap. It is **not achievable at bit-identity**, and this file is the
evidence, kept executable so the verdict can be re-checked rather than trusted.

Four layers. Three must pass. The fourth is a **canary on a known blocker**: it
asserts the blocker is STILL THERE, and tells you loudly if it goes away.

  1. `pairwise_sum` vs `np.sum`, f32 and f64. Must pass. numpy's sum is pairwise,
     not sequential, and this is the half of the problem that IS solved.
  2. f64 `exp` vs libm. Must pass.
  3. f32 `exp` vs libm. Must **DIVERGE** -- numpy's float32 exp is its own SIMD
     implementation and is not correctly rounded. If this ever stops diverging,
     numpy changed and the softmax became portable: go read `rust/src/softmax.rs`.
  4. the whole f32 softmax vs `_softmax_over_legal`. Must diverge, for the same
     reason, and the rate is reported so the size of the blocker is on record.

Why it matters that this is precise rather than "close enough": the differences
are ~1 ULP of float32. A 1-ULP prior only changes a move when it lands on a PUCT
tie, so an end-to-end test would pass most of the time while being wrong. That is
why the comparison here is direct and bitwise.

Usage:
    tests/verify_softmax.py --positions 20000
"""

from __future__ import annotations

import argparse
import math
import random
import struct
import sys
import zlib
from pathlib import Path

import chess
import numpy as np

import sumofish_core as core

ROOT = Path(__file__).resolve().parent.parent
for candidate in (ROOT, ROOT.parent / "chess-gpu"):
    if (candidate / "chessgpu" / "mcts.py").exists():
        sys.path.insert(0, str(candidate))
        break
from chessgpu.mcts import _softmax_over_legal  # noqa: E402
from chessgpu.tokenizer import ACTION_BY_MOVE_KEY, move_key  # noqa: E402


def b32(x) -> str:
    return struct.pack("<f", float(x)).hex()


def b64(x) -> str:
    return struct.pack("<d", float(x)).hex()


def logit_row(fen: str) -> np.ndarray:
    """float32, matching `policy._logprobs`, which ends in `.float()`."""
    return (
        np.random.default_rng(zlib.crc32(fen.encode()))
        .standard_normal(1968)
        .astype(np.float32)
    )


def layer1_pairwise(rng: random.Random) -> list[str]:
    bad = []
    for n in list(range(1, 200)) + [256, 512, 1000, 1968]:
        vals = np.array(
            [rng.random() * (10 ** rng.randint(-4, 1)) for _ in range(n)],
            dtype=np.float32,
        )
        if b32(core.pairwise_sum_f32(vals.tolist())) != b32(vals.sum()):
            bad.append(f"f32 n={n}")
        v64 = vals.astype(np.float64)
        if b64(core.pairwise_sum_f64(v64.tolist())) != b64(v64.sum()):
            bad.append(f"f64 n={n}")
    return bad


def layer2_exp_f64(rng: random.Random) -> int:
    vals = [-rng.random() * 10 for _ in range(20000)]
    np_out = np.exp(np.array(vals, dtype=np.float64))
    return sum(1 for v, n in zip(vals, np_out) if b64(math.exp(v)) != b64(n))


def layer3_exp_f32(rng: random.Random) -> tuple[int, int, float]:
    vals = np.array([-rng.random() * 8 for _ in range(20000)], dtype=np.float32)
    np_f32 = np.exp(vals)
    rounded = np.exp(vals.astype(np.float64)).astype(np.float32)
    diff = sum(1 for a, b in zip(np_f32, rounded) if b32(a) != b32(b))
    worst = max(
        abs(float(a) - float(b)) / max(abs(float(b)), 1e-30)
        for a, b in zip(np_f32, rounded)
    )
    return diff, len(vals), worst


def fens(limit: int) -> list[str]:
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


def layer4_softmax(corpus: list[str]) -> tuple[int, int]:
    bad = checked = 0
    for fen in corpus:
        board = chess.Board(fen)
        legal = list(board.legal_moves)
        if not legal:
            continue
        row = logit_row(fen)
        actions = [ACTION_BY_MOVE_KEY[move_key(m)] for m in legal]
        want = _softmax_over_legal(row, legal)
        got = core.softmax_over_legal(row.tolist(), actions)
        checked += 1
        if [b64(x) for x in got] != [b64(x) for x in want]:
            bad += 1
    return bad, checked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", type=int, default=20000)
    args = ap.parse_args()
    rng = random.Random(7)
    failed = False

    print("layer 1: pairwise summation vs np.sum (MUST PASS)")
    bad = layer1_pairwise(rng)
    if bad:
        print(f"  FAIL: {len(bad)} mismatches, first: {bad[:5]}")
        failed = True
    else:
        print("  OK: bit-exact in f32 and f64, lengths 1..1968")

    print("layer 2: f64 exp vs libm (MUST PASS)")
    n_bad = layer2_exp_f64(rng)
    if n_bad:
        print(f"  FAIL: {n_bad}/20000 mismatches")
        failed = True
    else:
        print("  OK: bit-exact over 20,000 values")

    print("layer 3: f32 exp vs libm (MUST DIVERGE -- known blocker)")
    diff, total, worst = layer3_exp_f32(rng)
    pct = 100.0 * diff / total
    print(f"  numpy float32 exp differs on {diff}/{total} = {pct:.1f}%, "
          f"max relative {worst:.2e} (~1 ULP of f32)")
    if diff == 0:
        print("  !! THE BLOCKER IS GONE. numpy's float32 exp now matches libm.")
        print("  !! The prior softmax has become portable to Rust. See")
        print("  !! rust/src/softmax.rs and re-open that roadmap item.")
        failed = True
    else:
        print("  OK: blocker still present, as documented")

    print("layer 4: the whole f32 softmax vs _softmax_over_legal")
    corpus = fens(args.positions)
    bad, checked = layer4_softmax(corpus)
    pct = 100.0 * bad / max(checked, 1)
    print(f"  differs on {bad}/{checked} = {pct:.1f}% of real positions")
    print("  (expected: layer 3 makes bit-identity unreachable)")

    print()
    if failed:
        print("FAIL")
        return 1
    print("OK: the reproducible half is exact; the blocker is where it was left.")
    print("The prior softmax stays in Python until the project accepts an epoch")
    print("boundary for it, at which point identity stops being the requirement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
