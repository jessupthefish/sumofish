#!/usr/bin/env python
"""Oracle 9: leaf deduplication is identity-preserving, and measure what it saves.

The claim being tested is stronger than "dedup helps". It is that dedup does
**not change the tree at all** -- k paths reaching one leaf still produce k visits
and k copies of the value, and the only difference is how many rows the network
was asked for. If that holds, dedup is a pure efficiency change that needs no Elo
measurement to justify, which matters because the Elo instrument is unreliable.

Two things are checked:

  1. **Identity.** Rust with dedup ON vs the real `chessgpu.mcts` (which has no
     dedup at all). The root visit vectors must be byte-identical. This is the
     whole claim.
  2. **The saving.** `evaluations` vs `unique_evaluations` at several batch sizes,
     which is the project's own mandated metric -- its Lab Notes say to report
     `unique/s`, never raw nps, precisely because of this.

Usage:
    tests/verify_dedup.py --positions 20 --sims 800
"""
from __future__ import annotations
import argparse, sys, random
from pathlib import Path
import chess
import sumofish_core as core

ROOT = Path(__file__).resolve().parent.parent
for c in (ROOT, ROOT.parent / "chess-gpu"):
    if (c / "chessgpu" / "mcts.py").exists():
        sys.path.insert(0, str(c)); break
from chessgpu.mcts import MCTS  # noqa: E402
from identity_search import MockPolicy, MockValuePolicy, corpus, rust_evaluate  # noqa: E402


def one(moves, sims, batch):
    py_board = chess.Board()
    for u in moves: py_board.push(chess.Move.from_uci(u))
    py = MCTS(MockValuePolicy(), policy=MockPolicy(), c_puct=2.0, c_puct_base=19652.0,
              c_puct_init=1.25, simulations=sims, batch=batch, fpu=-0.2, reuse=False)
    _r, pv = py.search(py_board)
    want = [(m.uci(), v) for m, v in pv.items()]

    out = {}
    for dedup in (False, True):
        pos = core.Position()
        for u in moves: pos.push_uci(u)
        m = core.Mcts(c_puct=2.0, c_puct_base=19652.0, c_puct_init=1.25,
                      fpu=-0.2, batch=batch, dedup=dedup)
        got = m.search(pos, sims, rust_evaluate)
        out[dedup] = (got, m.evaluations, m.unique_evaluations)
    return want, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", type=int, default=20)
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    games = corpus(args.positions, args.seed)
    print("1. IDENTITY: rust with dedup ON vs chessgpu.mcts (no dedup)")
    bad_on = bad_off = 0
    for i, moves in enumerate(games):
        want, out = one(moves, args.sims, 64)
        if out[False][0] != want: bad_off += 1
        if out[True][0] != want:
            bad_on += 1
            if bad_on <= 2:
                got = out[True][0]
                diffs = [(m, g, w) for (m, g), (_, w) in zip(got, want) if g != w]
                print(f"  FAIL [{i}] {len(diffs)}/{len(got)} counts differ; first: {diffs[:4]}")
    print(f"  dedup OFF: {len(games)-bad_off}/{len(games)} identical")
    print(f"  dedup ON : {len(games)-bad_on}/{len(games)} identical")

    print()
    print("2. THE SAVING: evaluations vs unique_evaluations")
    print(f"  {'batch':>6} {'sims':>6} {'evals':>8} {'unique':>8} {'duplicated':>11}")
    for batch in (8, 64, 256, 1024):
        _w, out = one(games[1] if len(games) > 1 else [], args.sims, batch)
        ev, un = out[True][1], out[True][2]
        dup = 100.0 * (ev - un) / max(ev, 1)
        print(f"  {batch:>6} {args.sims:>6} {ev:>8,} {un:>8,} {dup:>10.1f}%")

    print()
    if bad_on or bad_off:
        print("FAIL: dedup is NOT identity-preserving as implemented")
        return 1
    print("OK: dedup is identity-preserving. The tree is unchanged; only the")
    print("number of network rows drops. No Elo measurement needed to justify it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
