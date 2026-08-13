#!/usr/bin/env python
"""Oracle 9: leaf deduplication is identity-preserving, and measure what it saves.

The claim being tested is stronger than "dedup helps". It is that dedup does
**not change the tree at all** -- k paths reaching one leaf still produce k visits
and k copies of the value, and the only difference is how many rows the network
was asked for. If that holds, dedup is a pure efficiency change that needs no Elo
measurement to justify, which matters because the Elo instrument is unreliable.

Two things are checked:

  1. **Identity.** Rust with dedup ON vs the real `sumofish.mcts` (which has no
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
for c in (ROOT, ROOT.parent / "sumofish"):
    if (c / "sumofish" / "mcts.py").exists():
        sys.path.insert(0, str(c)); break
from sumofish.mcts import MCTS  # noqa: E402
from identity_search import MockPolicy, MockValuePolicy, corpus, rust_evaluate  # noqa: E402


def deployed(moves, sims, batch):
    """Rust vs Rust at the DEPLOYED search, dedup the only difference.

    `one()` below can only run at pre-v6 constants with every defect off,
    because it compares against `sumofish.mcts`, and that identity holds only
    there. So it says nothing about the search that actually plays. This does:
    v6 constants and vloss_fix on, which is the live configuration.

    The mock evaluator is deliberate and load-bearing. With the real nets dedup
    changes the ROW COUNT of each forward pass, and the network is not
    batch-shape invariant (`scripts/batch_invariance.py`: priors move on ~250 of
    256 positions once a pass carries 44+ rows), so the two arms would differ
    for float reasons and tell you nothing about the tree. The mock is
    batch-shape invariant by construction, which isolates the question.
    """
    out = {}
    for dedup in (False, True):
        pos = core.Position()
        for u in moves:
            pos.push_uci(u)
        m = core.Mcts(c_puct=2.0, c_puct_base=19652.0, c_puct_init=0.875,
                      fpu=-0.05, batch=batch, dedup=dedup, vloss_fix=True)
        out[dedup] = (m.search(pos, sims, rust_evaluate),
                      m.evaluations, m.unique_evaluations)
    return out


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
    print("1. IDENTITY: rust with dedup ON vs sumofish.mcts (no dedup)")
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
    print("3. THE DEPLOYED SEARCH: dedup ON vs OFF at v6 constants, vloss_fix=1")
    print("   (section 1 can only run with every defect OFF, so it cannot see this)")
    bad_dep = 0
    print(f"  {'batch':>6} {'identical':>12} {'rows':>10} {'leaf visits':>12} {'saved':>8}")
    for batch in (32, 64, 256):
        same = 0
        tot_e = tot_u = 0
        for moves in games:
            out = deployed(moves, args.sims, batch)
            if out[False][0] == out[True][0]:
                same += 1
            tot_e += out[True][1]
            tot_u += out[True][2]
        bad_dep += len(games) - same
        saved = 100.0 * (tot_e - tot_u) / max(tot_e, 1)
        print(f"  {batch:>6} {f'{same}/{len(games)}':>12} {tot_u:>10,} "
              f"{tot_e:>12,} {saved:>7.1f}%")

    print()
    if bad_on or bad_off or bad_dep:
        print("FAIL: dedup is NOT identity-preserving as implemented")
        return 1
    print("OK: dedup is identity-preserving, at the pre-v6 settings the Python")
    print("oracle needs AND at the deployed ones. The tree is unchanged; only the")
    print("number of network rows drops.")
    print()
    print("WHAT THIS MEANS FOR THE -168 ELO THAT KEEPS THE FLAG OFF")
    print("  sumofish-bot.service cites -168 Elo at a fixed clock, 20 games,")
    print("  measured 2026-07-29 jointly with CHESSGPU_COMPILE. Its mechanism was")
    print("  that dedup delivered '3,464 UNIQUE evaluations against plain's 4,160',")
    print("  i.e. 17% less knowledge of the position.")
    print()
    print("  That comparison does not hold. `unique_evaluations` counts ROWS SENT")
    print("  (rust/src/tree.rs:558), so with dedup OFF it is equal to the leaf-visit")
    print("  count by construction and is not a distinct-position count at all.")
    print("  Nothing measured plain's duplicates. Section 3 above is the like-for-")
    print("  like version: the trees are byte-identical, so both arms visit the same")
    print("  leaves and hold the same knowledge, and dedup simply pays for about half")
    print("  as many of them.")
    print()
    print("  The other half of that measurement is also stale: it predates")
    print("  vloss_fix, which is exactly the mechanism that stops parallel paths")
    print("  collapsing onto one leaf. Measured 2026-08-13 with the real nets at")
    print("  800 sims and batch 64, the collapsed fraction falls from 72.9% with")
    print("  vloss_fix off to 43.8% with it on.")
    print()
    print("  So the flag's verdict is not trustworthy and dedup deserves a")
    print("  re-measurement, ALONE, at a fixed clock, on the deployed engine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
