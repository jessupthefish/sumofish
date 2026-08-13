#!/usr/bin/env python
"""Is the network's output a function of the ROW COUNT of its forward pass?

This is the gate in front of `sumofish/batching.py`. Cross-game batching changes
how many rows go through each forward pass, so before it can be wired into
`scripts/match.py` somebody has to establish what that does to the numbers.

# The answer, measured 2026-08-13 on an RTX 5070 Ti, 256 random positions

The evaluator is **deterministic given the row count, and not invariant across
row counts.** Three separate findings, and only the first was expected:

  1. **Row count changes the output.** Against a one-row-per-pass reference, the
     PRIOR changes on ~250/256 positions once a pass carries 48 or more rows,
     and the argmax of the prior moves on 3/256 (1.2%). The VALUE is stable to
     one ULP until 128 rows, where it changes on 251/256 by up to 7.8e-04.
     Two kernel switches, at 40->44 and at 127->128.

  2. **WHICH positions share a pass does not matter at all.** Shuffling the
     partners at a fixed 64 rows gives bit-identical priors, 0/256 different,
     over three independent shuffles. The output for row i depends on how many
     rows there are and not on what is in them.

  3. **A repeated shape is bit-identical.** Same row counts, same bits, every
     time.

# Why that is the opposite of what this repository assumed

`sumofish/batching.py` argued that cross-game batching could not be
reproducible because "the grouping is decided by thread scheduling ... Different
grouping, different last-bit values, different games." The mechanism named there
is wrong: grouping is irrelevant (finding 2). What actually varies run to run is
the ROW COUNT of each flush, because that depends on how many games had a
request pending when the timer fired.

**That distinction is the whole design, because a row count is something you can
fix and a thread schedule is not.** `make_evaluator(..., pad_to=N)` already pads
every pass to a constant N. With it, a cross-game batched run issues passes of
exactly one shape, the row count stops depending on the scheduler, and the run
becomes bit-reproducible. The cost is near zero for the same reason batching
pays at all: the engine is launch-bound below ~128 rows, so the padding rows are
nearly free.

What this does NOT buy is comparability with the existing archive. Every result
in `runs/matches` was measured with variable row counts from an unbatched
search, and no amount of padding reproduces those. A batched run is reproducible
against other batched runs at the same `pad_to`, and that is all.

Usage:
    scripts/batch_invariance.py
    scripts/batch_invariance.py --positions 512 --json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import chess
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sumofish.tokenizer import MOVE_TO_ACTION  # noqa: E402

# Where the two kernel switches were found. Swept finely between 40 and 128.
SWEEP = (1, 8, 16, 32, 40, 44, 48, 56, 64, 72, 96, 112, 127, 128, 160, 192, 256)


def build_positions(n: int, seed: int) -> list[tuple[str, list[int]]]:
    """Random legal positions from random playouts.

    Deliberately NOT the opening book: the book is 6-12 plies and every position
    in it has ~30 legal moves, which would sample one narrow region of the row
    shapes the engine actually sends.
    """
    rng = random.Random(seed)
    out: list[tuple[str, list[int]]] = []
    while len(out) < n:
        b = chess.Board()
        for _ in range(rng.randint(4, 60)):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
            if b.is_game_over():
                break
        if b.is_game_over():
            continue
        out.append((b.fen(), [MOVE_TO_ACTION.get(m.uci(), -1) for m in b.legal_moves]))
    return out


def evaluate_in_chunks(ev, fens, acts, rows):
    priors, values = [], []
    for i in range(0, len(fens), rows):
        p, v = ev(fens[i:i + rows], acts[i:i + rows])
        priors.extend(np.array(x) for x in p)
        values.extend(v)
    return priors, values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=256)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--value", default=str(ROOT / "runs/value.pt"))
    ap.add_argument("--policy", default=str(ROOT / "runs/policy.pt"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shuffles", type=int, default=3)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from match import load_prior, load_value  # noqa: E402
    from sumofish.rust_mcts import make_evaluator  # noqa: E402

    ev = make_evaluator(load_prior(args.policy, device=args.device),
                        load_value(args.value, device=args.device),
                        compile_nets=False, pad_to=None)

    pos = build_positions(args.positions, args.seed)
    fens = [p[0] for p in pos]
    acts = [p[1] for p in pos]
    n = len(pos)

    ref_p, ref_v = evaluate_in_chunks(ev, fens, acts, 1)

    report: dict = {"positions": n, "device": args.device, "sweep": {}}
    if not args.json:
        print(f"{n} positions, reference = one row per forward pass\n")
        print(f"{'rows':>6}{'prior differs':>16}{'argmax moved':>14}"
              f"{'value differs':>16}{'max |dv|':>12}")

    for rows in SWEEP:
        bp, bv = evaluate_in_chunks(ev, fens, acts, rows)
        pd = sum(1 for a, b in zip(ref_p, bp) if not np.array_equal(a, b))
        am = sum(1 for a, b in zip(ref_p, bp)
                 if int(np.argmax(a)) != int(np.argmax(b)))
        vd = sum(1 for a, b in zip(ref_v, bv) if a != b)
        mv = max(abs(a - b) for a, b in zip(ref_v, bv))
        report["sweep"][rows] = {"prior_differs": pd, "argmax_moved": am,
                                 "value_differs": vd, "max_abs_dvalue": mv}
        if not args.json:
            print(f"{rows:>6}{pd:>11}/{n:<4}{am:>14}{vd:>11}/{n:<4}{mv:>12.2e}")

    # Finding 2: does WHICH positions share a pass matter, at a fixed count?
    base_p, _ = evaluate_in_chunks(ev, fens, acts, 64)
    shuffle_diffs = []
    for t in range(args.shuffles):
        rng = random.Random(100 + t)
        idx = list(range(n))
        rng.shuffle(idx)
        got: dict[int, np.ndarray] = {}
        for i in range(0, n, 64):
            chunk = idx[i:i + 64]
            p, _v = ev([fens[j] for j in chunk], [acts[j] for j in chunk])
            for k, j in enumerate(chunk):
                got[j] = np.array(p[k])
        # Only rows from a FULL 64-row pass are comparable; a short tail is a
        # different shape and would be counted as a grouping effect when it is
        # a size effect. With n a multiple of 64 there is no tail.
        shuffle_diffs.append(sum(1 for j in range(n)
                                 if not np.array_equal(base_p[j], got[j])))
    report["grouping_differs_at_64"] = shuffle_diffs

    # Finding 3: is a repeated shape bit-identical?
    repeat_p, _ = evaluate_in_chunks(ev, fens, acts, 64)
    report["repeat_differs_at_64"] = sum(
        1 for a, b in zip(base_p, repeat_p) if not np.array_equal(a, b))

    if args.json:
        print(json.dumps(report, indent=2, default=float))
        return 0

    print(f"\nGROUPING, at a fixed 64 rows with different partners:")
    for t, d in enumerate(shuffle_diffs):
        print(f"  shuffle {t}: prior differs {d}/{n}")
    print(f"\nREPEATABILITY, same shapes again: "
          f"prior differs {report['repeat_differs_at_64']}/{n}")

    print()
    if any(shuffle_diffs) or report["repeat_differs_at_64"]:
        print("GROUPING MATTERS. Padding to a fixed row count is NOT sufficient")
        print("to make a cross-game batched run reproducible, and the design in")
        print("sumofish/batching.py's docstring needs rethinking before use.")
        return 1
    print("Output is a deterministic function of the ROW COUNT alone.")
    print("Padding every pass to a fixed count therefore makes a cross-game")
    print("batched run bit-reproducible, because the count stops depending on")
    print("thread scheduling. It does NOT make it comparable to the unbatched")
    print("archive, which was measured at variable row counts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
