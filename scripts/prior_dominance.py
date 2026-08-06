#!/usr/bin/env python
"""Why is the first rung of the exchange ladder worth almost nothing?

    scripts/prior_dominance.py --positions 40

The re-earned ladder (2026-08-02, 300 games a rung, all four honest) came out
NON-MONOTONIC, and nobody has explained it:

    200 ->  400 sims    +29.0 +-25.1 Elo     A scores 54.2%, 55% draws
    400 ->  800         +240.8 +-36.3        A scores 80.0%, 35% draws
    800 -> 1600         +308.2 +-43.8        A scores 85.5%, 26% draws
   1600 -> 3200         +233.7 +-35.6        A scores 79.3%, 37% draws

A doubling of search is worth 8x more at the second rung than at the first, and
then falls away again at the top. The rung configs are byte-identical apart from
the sim counts, so it is not a harness difference. The shape is an inverted U,
which is the signature of a MECHANISM rather than a bug -- and this script tests
the obvious candidate.

## The hypothesis

The engine's move is the most-visited root child, and MCTS spends its first
visits where the policy prior sends it. At 200 simulations across ~30 legal
moves, the search barely has the budget to overturn the prior's first choice: it
plays the prior's move, and so does the 400-sim arm, and two engines playing the
same move draw. At the other end, both arms are near-converged and agree again
for the opposite reason. Maximum disagreement -- and so maximum Elo -- sits in
the middle, which is exactly where the ladder puts it.

If that is right, the disagreement rate between adjacent sim counts should trace
the same inverted U as the Elo, and the low-sim agreement should be agreement
WITH THE PRIOR specifically, not merely with each other.

## What it measures

For each position, the chosen move at every rung of the ladder, plus the prior's
own argmax with no search at all. Then, per adjacent pair:

  disagree%      how often doubling the search changes the move played
  prior-locked%  how often BOTH arms just played the prior's top move

This is not an Elo measurement and does not pretend to be one. It explains a
shape; the ladder still owns the numbers.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

RUNGS = [200, 400, 800, 1600, 3200]


def positions(n: int, seed: int) -> list[chess.Board]:
    """Real positions from the ladder's own games, not a hand-picked set.

    Sampled out of the middle of `sims-800-vs-400`'s PGN stream: openings are
    book moves the arms never chose, and the last few plies of a decided game are
    positions where every move is the same move. Neither says anything about
    whether search overrules the prior.
    """
    rows = [json.loads(l) for l in (ROOT / "runs/matches/sims-800-vs-400/games.jsonl").open() if l.strip()]
    rng = random.Random(seed)
    out = []
    for row in rng.sample(rows, min(len(rows), n)):
        board = chess.Board()
        for uci in row["opening"] + row["moves"]:
            try:
                board.push(chess.Move.from_uci(uci))
            except (ValueError, AssertionError):
                break
        plies = len(board.move_stack)
        if plies < 20:
            continue
        # Somewhere in the middlegame: past the book, before the result is in.
        target = rng.randrange(12, max(13, int(plies * 0.75)))
        replay = chess.Board()
        for mv in board.move_stack[:target]:
            replay.push(mv)
        if replay.legal_moves and not replay.is_game_over():
            out.append(replay)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--positions", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--out", default=str(ROOT / "runs/lab/prior-dominance.json"))
    args = ap.parse_args()

    from match import load_prior, load_value
    from sumofish.rust_mcts import select_mcts_class

    cls, core = select_mcts_class()
    value, policy = load_value(str(ROOT / "runs/value.pt")), load_prior(str(ROOT / "runs/policy.pt"))
    boards = positions(args.positions, args.seed)
    print(f"{len(boards)} positions, core={core}, rungs={RUNGS}")

    engines = {
        s: cls(value, policy=policy, simulations=s, batch=64, c_puct=2.0, fpu=-0.2)
        for s in RUNGS
    }

    rows = []
    for i, board in enumerate(boards):
        # The prior's own choice, with no search at all: the thing the search
        # has to overrule to be worth anything. Read the same way `MCTS._priors`
        # reads it -- logprobs over the full action space, then argmax over the
        # LEGAL moves only, which is the masking the whole engine depends on.
        legal = list(board.legal_moves)
        row = policy._logprobs([board])[0].cpu().numpy()
        from sumofish.mcts import _softmax_over_legal
        prior_probs = dict(zip(legal, _softmax_over_legal(row, legal), strict=True))
        top = max(prior_probs, key=prior_probs.get)
        picks = {}
        for s in RUNGS:
            engines[s].reset()
            _root, visits = engines[s].search(board.copy())
            picks[s] = max(visits.items(), key=lambda kv: kv[1])[0].uci()
        rows.append({"fen": board.fen(), "prior": top.uci(), "picks": picks})
        print(f"  {i + 1}/{len(boards)}  prior={top.uci()}  " +
              "  ".join(f"{s}:{picks[s]}" for s in RUNGS), flush=True)

    print(f"\n  {'rung':>14}  {'disagree':>9}  {'prior-locked':>13}   (ladder Elo)")
    ladder = {400: "+29.0", 800: "+240.8", 1600: "+308.2", 3200: "+233.7"}
    summary = []
    for lo, hi in zip(RUNGS, RUNGS[1:]):
        n = len(rows)
        disagree = sum(1 for r in rows if r["picks"][lo] != r["picks"][hi])
        locked = sum(1 for r in rows if r["picks"][lo] == r["prior"] == r["picks"][hi])
        summary.append({"from": lo, "to": hi, "n": n,
                        "disagree": disagree, "prior_locked": locked})
        print(f"  {lo:5} -> {hi:5}  {disagree / n * 100:8.1f}%  {locked / n * 100:12.1f}%"
              f"   {ladder.get(hi, '?'):>8}")

    Path(args.out).write_text(json.dumps(
        {"core": core, "rungs": RUNGS, "positions": len(rows),
         "seed": args.seed, "pairs": summary, "rows": rows}, indent=1))
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
