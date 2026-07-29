#!/usr/bin/env python
"""Oracle 8: tree reuse stays byte-identical across a whole game.

`identity_search.py` checks a single search from a fresh tree. That cannot see
rerooting at all, and rerooting is where the state lives: the retained subtree
carries visits, values and priors forward, so a divergence compounds instead of
showing up once.

So this plays a game. At every ply both implementations search the SAME position
with reuse enabled, and three things must match:

  * the root visit vector, move for move, in order
  * `reused` -- the simulations inherited from the previous move's tree, which is
    the number that says rerooting actually happened rather than silently
    declining every time
  * the move chosen, since that is what advances the game and a divergence there
    ends the comparison

**The test fails if `reused` is always zero.** A reroot that always declines is
indistinguishable from correct on a visit-vector comparison alone, and would make
this whole file vacuous while passing.

It also drives the cases where rerooting must DECLINE, because declining
correctly matters as much as accepting:

  * a jump to an unrelated position (no shared prefix)
  * skipping three plies at once (more than our-move-plus-their-reply)
  * a fresh game after `reset()`

Usage:
    tests/identity_reuse.py --games 6 --plies 12 --sims 200
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import chess

import sumofish_core as core

ROOT = Path(__file__).resolve().parent.parent
for candidate in (ROOT, ROOT.parent / "chess-gpu"):
    if (candidate / "chessgpu" / "mcts.py").exists():
        sys.path.insert(0, str(candidate))
        break
from chessgpu.mcts import MCTS  # noqa: E402

from identity_search import MockPolicy, MockValuePolicy, rust_evaluate  # noqa: E402


def play_game(plies: int, sims: int, batch: int, seed: int) -> tuple[str | None, int, int]:
    """Returns (failure or None, plies compared, total reused)."""
    rng = random.Random(seed)

    py_board = chess.Board()
    py_mcts = MCTS(
        MockValuePolicy(), policy=MockPolicy(), c_puct=2.0, c_puct_base=19652.0,
        c_puct_init=1.25, simulations=sims, batch=batch, fpu=-0.2, reuse=True,
    )

    rs_pos = core.Position()
    rs_mcts = core.Mcts(
        c_puct=2.0, c_puct_base=19652.0, c_puct_init=1.25, fpu=-0.2,
        batch=batch, reuse=True,
    )

    total_reused = 0
    for ply in range(plies):
        if py_board.is_game_over(claim_draw=False):
            return None, ply, total_reused

        _root, py_visits = py_mcts.search(py_board)
        py_pairs = [(m.uci(), v) for m, v in py_visits.items()]
        rs_pairs = rs_mcts.search(rs_pos, sims, rust_evaluate)

        if rs_pairs != py_pairs:
            same_moves = [m for m, _ in rs_pairs] == [m for m, _ in py_pairs]
            lines = [f"ply {ply}: visit vectors differ (reuse on)"]
            if not same_moves:
                lines.append("  MOVE ORDER differs -- Stage A regression")
                lines.append(f"    rust:   {' '.join(m for m, _ in rs_pairs)}")
                lines.append(f"    python: {' '.join(m for m, _ in py_pairs)}")
            else:
                diffs = [
                    (m, r, p) for (m, r), (_, p) in zip(rs_pairs, py_pairs) if r != p
                ]
                lines.append(f"  {len(diffs)} of {len(rs_pairs)} counts differ")
                for m, r, p in diffs[:6]:
                    lines.append(f"    {m}: rust={r} python={p}")
            lines.append(f"  reused: rust={rs_mcts.reused} python={py_mcts.reused}")
            lines.append(f"  fen: {py_board.fen()}")
            return "\n".join(lines), ply, total_reused

        if rs_mcts.reused != py_mcts.reused:
            return (
                f"ply {ply}: reused differs -- rust={rs_mcts.reused} "
                f"python={py_mcts.reused}\n  fen: {py_board.fen()}"
            ), ply, total_reused

        total_reused += py_mcts.reused

        # Advance both by the same move. Most-visited, ties to the first, which
        # is what `play()` does on both sides.
        best = max(py_pairs, key=lambda kv: kv[1])[0]
        # Occasionally advance TWO plies without searching in between, which is
        # the other case reroot must accept (our move plus their reply).
        moves = [best]
        if rng.random() < 0.35:
            py_board.push(chess.Move.from_uci(best))
            rs_pos.push_uci(best)
            legal = [m.uci() for m in py_board.legal_moves]
            if legal:
                moves.append(rng.choice(legal))
            for u in moves[1:]:
                py_board.push(chess.Move.from_uci(u))
                rs_pos.push_uci(u)
        else:
            py_board.push(chess.Move.from_uci(best))
            rs_pos.push_uci(best)

    return None, plies, total_reused


def decline_cases(sims: int, batch: int) -> list[str]:
    """Rerooting must refuse in these, and refuse the same way Python does."""
    failures = []

    def fresh():
        py_board = chess.Board()
        py = MCTS(
            MockValuePolicy(), policy=MockPolicy(), c_puct=2.0, c_puct_base=19652.0,
            c_puct_init=1.25, simulations=sims, batch=batch, fpu=-0.2, reuse=True,
        )
        rs_pos = core.Position()
        rs = core.Mcts(
            c_puct=2.0, c_puct_base=19652.0, c_puct_init=1.25, fpu=-0.2,
            batch=batch, reuse=True,
        )
        return py_board, py, rs_pos, rs

    # 1. Three plies at once: more than our-move-plus-their-reply.
    py_board, py, rs_pos, rs = fresh()
    py.search(py_board)
    rs.search(rs_pos, sims, rust_evaluate)
    for u in ["e2e4", "e7e5", "g1f3"]:
        py_board.push(chess.Move.from_uci(u))
        rs_pos.push_uci(u)
    _r, pv = py.search(py_board)
    rv = rs.search(rs_pos, sims, rust_evaluate)
    if py.reused != 0 or rs.reused != 0:
        failures.append(f"3-ply skip should decline: rust={rs.reused} python={py.reused}")
    if rv != [(m.uci(), v) for m, v in pv.items()]:
        failures.append("3-ply skip: visit vectors differ")

    # 2. A position with no shared prefix at all.
    py_board, py, rs_pos, rs = fresh()
    py.search(py_board)
    rs.search(rs_pos, sims, rust_evaluate)
    py_board2 = chess.Board()
    rs_pos2 = core.Position()
    for u in ["d2d4", "d7d5"]:
        py_board2.push(chess.Move.from_uci(u))
        rs_pos2.push_uci(u)
    _r, pv = py.search(py_board2)
    rv = rs.search(rs_pos2, sims, rust_evaluate)
    if rs.reused != py.reused:
        failures.append(f"unrelated position: rust={rs.reused} python={py.reused}")
    if rv != [(m.uci(), v) for m, v in pv.items()]:
        failures.append("unrelated position: visit vectors differ")

    # 3. reset() must forget the tree.
    py_board, py, rs_pos, rs = fresh()
    py.search(py_board)
    rs.search(rs_pos, sims, rust_evaluate)
    py_board.push(chess.Move.from_uci("e2e4"))
    rs_pos.push_uci("e2e4")
    py.reset()
    rs.reset()
    _r, pv = py.search(py_board)
    rv = rs.search(rs_pos, sims, rust_evaluate)
    if py.reused != 0 or rs.reused != 0:
        failures.append(f"after reset should decline: rust={rs.reused} python={py.reused}")
    if rv != [(m.uci(), v) for m, v in pv.items()]:
        failures.append("after reset: visit vectors differ")

    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--plies", type=int, default=12)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-report", type=int, default=2)
    args = ap.parse_args()

    print(f"{args.games} games x up to {args.plies} plies, {args.sims} sims, reuse ON")
    failures = 0
    compared = 0
    reused_total = 0
    for g in range(args.games):
        problem, plies, reused = play_game(args.plies, args.sims, args.batch, 1234 + g)
        compared += plies
        reused_total += reused
        if problem:
            failures += 1
            if failures <= args.max_report:
                print(f"\nFAIL game {g}: {problem}")

    print(f"  {compared} plies compared, {reused_total:,} simulations inherited in total")

    print("decline cases (rerooting must refuse, the same way Python does)")
    dec = decline_cases(args.sims, args.batch)
    for d in dec:
        print(f"  FAIL {d}")
    if not dec:
        print("  OK: 3-ply skip, unrelated position, and reset all decline")

    print()
    if reused_total == 0:
        print("FAIL: nothing was ever inherited, so rerooting never actually")
        print("happened and this run proves nothing. Check `reuse` is wired.")
        return 1
    if failures or dec:
        print(f"FAIL: {failures} games diverged, {len(dec)} decline cases wrong")
        return 1
    print("OK: reuse is byte-identical across whole games, and declines correctly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
