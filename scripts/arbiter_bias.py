#!/usr/bin/env python
"""How much did the arbiter's warm transposition table actually change?

`Arbiter.agrees()` called `engine.analyse()` with no `game=` token until
2026-08-11, so python-chess emitted `ucinewgame` exactly once per PROCESS
rather than once per game (`chess/engine.py`: it fires only when
`first_game or self.game != game`, and `None != None` is False). The
adjudicating Stockfish therefore accumulated hash across every probe of every
game in a match, at `--arbiter-nodes` 200,000 a probe.

That is the same defect commit 7711bf7 fixed for `Player`, left on the other
Stockfish instance, and it matters because the arbiter DECIDES THE RESULT:
29.8% of the games in the 1600-node anchor ended `adjudicated-arbiter`.

This script sizes it instead of arguing about it. Every adjudicated game
already stores the position it was called on (`final_fen` in `games.jsonl`), so
the verdicts can be recomputed both ways with no games replayed and no GPU:

  COLD  a fresh `game=` token per position, i.e. the fixed behaviour
  WARM  one engine, no token, positions in their original order, i.e. the bug

A verdict is `wp >= 0.97` (white winning) or `wp <= 0.03`, matching
`Arbiter.agrees()` exactly. What we want to know is how often the two conditions
DISAGREE, because a disagreement is a game that would have been adjudicated
under one and played on under the other.

Usage:
    scripts/arbiter_bias.py runs/matches/stockfish-anchor-700nodes
    scripts/arbiter_bias.py <dir> --limit 200      # a quick read
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess
import chess.engine

ROOT = Path(__file__).resolve().parent.parent
STOCKFISH = ROOT / "tools" / "stockfish" / "stockfish-ubuntu-x86-64-bmi2"


def verdict(info, white_winning: bool) -> bool:
    """Exactly `Arbiter.agrees()`: same threshold, same WDL model, same frame."""
    score = info.get("score")
    if score is None:
        return False
    wp = score.white().wdl(model="sf12").expectation()
    return wp >= 0.97 if white_winning else wp <= 0.03


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("match_dir")
    ap.add_argument("--nodes", type=int, default=200_000,
                    help="must match the match's --arbiter-nodes (default 200000)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    games_p = Path(args.match_dir) / "games.jsonl"
    if not games_p.exists():
        print(f"no games.jsonl in {args.match_dir}", file=sys.stderr)
        return 2

    rows = []
    for line in games_p.read_text().splitlines():
        try:
            g = json.loads(line)
        except json.JSONDecodeError:
            continue
        if g.get("reason") != "adjudicated-arbiter" or not g.get("final_fen"):
            continue
        rows.append((g["final_fen"], g["result"] == "1-0"))
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        print("no adjudicated games with a stored final_fen")
        return 1

    print(f"{len(rows)} adjudicated positions from {args.match_dir}")
    print(f"arbiter at {args.nodes} nodes, threshold wp>=0.97 / <=0.03\n")

    eng = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH))
    try:
        cold = [verdict(eng.analyse(chess.Board(f), chess.engine.Limit(nodes=args.nodes),
                                    game=f"cold-{i}"), w)
                for i, (f, w) in enumerate(rows)]
    finally:
        eng.quit()

    eng = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH))
    try:
        warm = [verdict(eng.analyse(chess.Board(f), chess.engine.Limit(nodes=args.nodes),
                                    game=None), w)
                for f, w in rows]
    finally:
        eng.quit()

    disagree = [i for i, (c, w) in enumerate(zip(cold, warm)) if c != w]
    cold_yes, warm_yes = sum(cold), sum(warm)
    n = len(rows)
    print(f"  COLD  upholds the adjudication: {cold_yes}/{n}  ({100*cold_yes/n:.1f}%)")
    print(f"  WARM  upholds the adjudication: {warm_yes}/{n}  ({100*warm_yes/n:.1f}%)")
    print(f"  DISAGREE:                        {len(disagree)}/{n}  "
          f"({100*len(disagree)/n:.1f}%)")
    print()
    if not disagree:
        print("  No verdict changed. The bias is real in mechanism and did not move a")
        print("  single adjudication on this match, which is the expected outcome for a")
        print("  200k-node probe of a position already past 0.97: the threshold is far")
        print("  from where a warm table changes an evaluation. The fix still stands on")
        print("  reproducibility (a match must be a function of its seed), not on Elo.")
    else:
        print(f"  {len(disagree)} adjudications differ between the two conditions. Each is a")
        print("  game that would have been called under one arbiter and played on under")
        print("  the other, so results taken on the warm harness are not reproducible")
        print("  and any rung containing them needs re-running, not just relabelling.")

    out = Path(args.match_dir) / "arbiter-bias.json"
    out.write_text(json.dumps({
        "positions": n, "nodes": args.nodes,
        "cold_upholds": cold_yes, "warm_upholds": warm_yes,
        "disagreements": len(disagree),
        "disagree_index": disagree[:50],
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
