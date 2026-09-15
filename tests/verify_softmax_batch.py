#!/usr/bin/env python
"""`_softmax_over_legal_batch` is bit-identical to `_softmax_over_legal`, row for row.

The engine's prior is float32 and a 1-ULP difference only changes a move when it
lands on a PUCT tie, so this compares BITS, not closeness. Four sweeps:

  1. every row length 1-218 (218 is the most legal moves a chess position has),
     many seeds each, so every pairwise-sum block boundary (8, 16, 128, 129...)
     and every SIMD tail length of numpy's float32 exp is covered.
  2. random live-shaped batches of 64 rows, lengths 1-60, with the occasional
     -1 action (a move outside the action space).
  3. logit scales from nearly flat to very peaked, including rows whose max
     is repeated.
  4. a batch of one row, and a batch where every row has one legal move.

Usage:
    tests/verify_softmax_batch.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sumofish.rust_mcts import _softmax_over_legal, _softmax_over_legal_batch  # noqa: E402

NA = 1968
rng = np.random.default_rng(20260915)
results: list[bool] = []


def same(rows, actions) -> bool:
    got = _softmax_over_legal_batch(rows, actions)
    want = [_softmax_over_legal(rows[i], actions[i]) for i in range(len(actions))]
    return all(np.asarray(g, dtype=np.float64).tobytes() == np.asarray(w, dtype=np.float64).tobytes()
               for g, w in zip(got, want, strict=True))


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    results.append(ok)


def rows_of(n, scale=3.0):
    return (rng.standard_normal((n, NA)) * scale).astype(np.float32)


# 1: every length, mixed into one batch with a long row so padding is real
bad = []
for length in range(1, 219):
    for _ in range(20):
        acts = [rng.choice(NA, size=length, replace=False).tolist(),
                rng.choice(NA, size=218, replace=False).tolist(),
                rng.choice(NA, size=int(rng.integers(1, 219)), replace=False).tolist()]
        if not same(rows_of(3), acts):
            bad.append(length)
            break
check("every row length 1-218, 20 batches each, padded against a 218-move row",
      not bad, f"differs at lengths {bad[:10]}" if bad else "4,360 batches")

# 2: live-shaped batches
bad = 0
for _ in range(3000):
    acts = []
    for _ in range(64):
        k = int(rng.integers(1, 61))
        a = rng.choice(NA, size=k, replace=False).tolist()
        if rng.random() < 0.02:
            a[int(rng.integers(0, k))] = -1
        acts.append(a)
    bad += not same(rows_of(64, float(rng.uniform(0.5, 8))), acts)
check("3,000 live-shaped batches of 64 (lengths 1-60, some -1 actions)", bad == 0, f"{bad} differ")

# 3: scales and ties
bad = 0
for scale in (1e-4, 0.1, 1.0, 10.0, 60.0):
    for _ in range(200):
        r = rows_of(8, scale)
        acts = [rng.choice(NA, size=int(rng.integers(1, 80)), replace=False).tolist() for _ in range(8)]
        r[0, acts[0]] = np.float32(1.5)                     # a row that is all one value
        if len(acts[1]) > 1:
            r[1, acts[1][:2]] = r[1, acts[1]].max()         # a repeated maximum
        bad += not same(r, acts)
check("logit scale 1e-4..60, a constant row and a repeated max", bad == 0, f"{bad} differ")

# 4: degenerate batches
check("a batch of one row", same(rows_of(1), [rng.choice(NA, size=31, replace=False).tolist()]))
check("every row has exactly one legal move", same(rows_of(64), [[int(rng.integers(0, NA))] for _ in range(64)]))
check("a lone move outside the action space", same(rows_of(2), [[-1], [5, -1, 7]]))

ok = all(results)
print(f"\n{'OK' if ok else 'FAIL'}: {sum(results)}/{len(results)} checks")
sys.exit(0 if ok else 1)
