#!/usr/bin/env python
"""What pondering did, game by game, from the engine's own telemetry.

Reads `logs/engine.jsonl` (and its rotated `.1`), groups `move` and `ponder`
records by the `game` id lichess-bot stamps on every record, and prints one
row per game:

    moves      our moves searched in that game
    think s    wall time of our own searches
    ponder s   wall time spent pondering, split by why it stopped:
                 stopped  the opponent replied (the whole of their think)
                 tree     the tree reached CHESSGPU_PONDER_MAX_TREE_NODES
                 cap      one ponder paid CHESSGPU_PONDER_MAX_NODES evals
    ponder ev  evaluations paid on the opponent's clock
    inherit    share of the visits behind our moves that pondering and reuse
               handed over: sum(reused) / sum(reused + nodes). Reuse existed
               before pondering, so this is not pondering's share alone;
               compare against pre-ponder games for that.
    nps        median evals/s of our own searches

A `tree` or `cap` stop means the opponent thought longer than the ponder ran,
so the opponent's think time is only known for `stopped` ponders.

Memory is not in the telemetry. Sample RSS from /proc while a game runs.

Usage:
    scripts/ponder_report.py                 # the last 20 games
    scripts/ponder_report.py --last 50
    scripts/ponder_report.py --game Sy7VT4eU
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = [ROOT / "logs/engine.jsonl.1", ROOT / "logs/engine.jsonl"]


def records():
    for path in LOGS:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                if not line.startswith("{"):
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut by rotation or a crash mid-write


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--last", type=int, default=20)
    ap.add_argument("--game")
    args = ap.parse_args()

    games: dict[str, dict] = defaultdict(lambda: {"moves": [], "ponders": [], "t0": None})
    for r in records():
        g = r.get("game")
        if not g or r.get("ev") not in ("move", "ponder"):
            continue
        rec = games[g]
        rec["t0"] = rec["t0"] or r.get("t")
        rec["moves" if r["ev"] == "move" else "ponders"].append(r)

    order = sorted(games, key=lambda g: games[g]["t0"] or 0)
    if args.game:
        order = [g for g in order if g == args.game]
    else:
        order = order[-args.last:]
    if not order:
        print("no games with a game id in the telemetry")
        return 1

    hdr = (f"{'game':9} {'start':11} {'moves':>5} {'think s':>8} "
           f"{'stopped s':>9} {'tree s':>7} {'cap s':>6} {'ponder ev':>10} "
           f"{'inherit':>7} {'nps':>6}  stops")
    print(hdr)
    print("-" * len(hdr))
    tot = Counter()
    for g in order:
        m, p = games[g]["moves"], games[g]["ponders"]
        secs = Counter()
        why = Counter()
        for x in p:
            secs[x.get("why", "?")] += x.get("elapsed", 0.0)
            why[x.get("why", "?")] += 1
        reused = sum(max(0, x.get("reused") or 0) for x in m)
        nodes = sum(x.get("nodes") or 0 for x in m)
        inherit = reused / (reused + nodes) if reused + nodes else 0.0
        nps = statistics.median([x["nps"] for x in m if x.get("nps")]) if m else 0
        think = sum(x.get("elapsed", 0.0) for x in m)
        pev = sum(x.get("nodes") or 0 for x in p)
        start = datetime.fromtimestamp(games[g]["t0"]).strftime("%m-%d %H:%M") if games[g]["t0"] else "?"
        stops = " ".join(f"{k}:{v}" for k, v in sorted(why.items()))
        print(f"{g:9} {start:11} {len(m):5} {think:8.0f} {secs['stopped']:9.0f} "
              f"{secs['tree']:7.0f} {secs['cap']:6.0f} {pev:10,} {inherit:7.0%} {nps:6.0f}  {stops}")
        tot["moves"] += len(m)
        tot["think"] += think
        tot["pev"] += pev
        tot["reused"] += reused
        tot["nodes"] += nodes
        for k, v in why.items():
            tot["n_" + k] += v
        for k, v in secs.items():
            tot["s_" + k] += v

    if len(order) > 1:
        share = tot["reused"] / (tot["reused"] + tot["nodes"]) if tot["reused"] + tot["nodes"] else 0
        pondered = tot["s_stopped"] + tot["s_tree"] + tot["s_cap"]
        print("-" * len(hdr))
        print(f"{len(order)} games, {tot['moves']} moves: think {tot['think']/3600:.1f} h, "
              f"ponder {pondered/3600:.1f} h ({pondered / tot['think']:.2f}x our own think), "
              f"{tot['pev']:,} ponder evals, inherited {share:.0%} of searched visits")
        print(f"ponder stops: stopped {tot['n_stopped']}, tree {tot['n_tree']}, cap {tot['n_cap']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
