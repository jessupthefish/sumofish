#!/usr/bin/env python
"""What budget does the engine ACTUALLY search at when it plays?

`docs/OPERATING-POINT.md` is the write-up; this regenerates its measured half.

Why this exists: every strength number in this project is measured at
`--sims 400`, and nothing checked what the deployed engine does until
2026-08-13. It is ~234,000 simulations a move, 9.2 doublings higher. The
parity ladder established that the exchange rate between search and strength is
NOT constant over budget (1.15 node-doublings per sims-doubling at 200 sims,
0.49 at 1600), so that gap is not a detail.

# What counts as a move

`logs/engine.jsonl` carries a progressive `"ev": "think"` record many times per
move and one `"ev": "move"` record when the move is played. ONLY the latter is a
completed move. Averaging `think` records instead answers "what does a search
look like partway through", which is a different question with a smaller answer,
and it is the mistake this script exists to stop being repeated: it reports a
median of ~73k nodes against the true ~150k.

Usage:
    scripts/operating_point.py
    scripts/operating_point.py --logs logs/engine.jsonl --json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The lab's fixed budget, and the highest rung the parity ladder ever reached.
LAB_SIMS = 400
LADDER_TOP_SIMS = 3200


def load_moves(paths: list[Path]) -> list[dict]:
    """Completed moves only. A malformed tail line is skipped, not fatal:
    the log is appended to by a live process and may be mid-write."""
    out = []
    for p in paths:
        if not p.exists():
            continue
        with open(p, errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("ev") == "move":
                    out.append(r)
    return out


def quantile(values: list[float], p: float) -> float:
    v = sorted(values)
    if not v:
        return float("nan")
    return v[int(p * (len(v) - 1))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="*", default=None,
                    help="default: logs/engine.jsonl and its .1 rotation")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    paths = ([Path(p) for p in args.logs] if args.logs
             else [ROOT / "logs/engine.jsonl.1", ROOT / "logs/engine.jsonl"])
    moves = load_moves(paths)
    if not moves:
        print("no completed moves found in", ", ".join(str(p) for p in paths))
        print("the bot writes an 'ev':'move' record only when it plays one; "
              "an idle bot logs nothing.")
        return 1

    fields = {
        "budget_s": lambda x: x.get("budget", 0.0),
        "elapsed_s": lambda x: x.get("elapsed", 0.0),
        "nodes": lambda x: x.get("nodes", 0),
        "sims": lambda x: x.get("sims", 0),
        "nps": lambda x: x.get("nps", 0),
    }
    stats = {
        name: {q: quantile([f(m) for m in moves], p)
               for q, p in (("p10", 0.10), ("median", 0.50), ("p90", 0.90))}
        for name, f in fields.items()
    }
    med_sims = stats["sims"]["median"]
    gap = med_sims / LAB_SIMS
    report = {
        "moves": len(moves),
        "games": len({m.get("game") for m in moves}),
        "stats": stats,
        "sims_per_node": st.median([m.get("sims", 0) / max(m.get("nodes", 1), 1)
                                    for m in moves]),
        "budget_spent_fraction": st.median(
            [m.get("elapsed", 0) / max(m.get("budget", 0) or 1e-9, 1e-9) for m in moves]),
        "moves_under_1s": sum(1 for m in moves if m.get("elapsed", 0) < 1),
        "lab_sims": LAB_SIMS,
        "gap_vs_lab": gap,
        "gap_doublings_vs_lab": math.log2(gap) if gap > 0 else float("nan"),
        "gap_doublings_vs_ladder_top": (math.log2(med_sims / LADDER_TOP_SIMS)
                                        if med_sims > 0 else float("nan")),
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"{report['moves']} completed moves, {report['games']} games\n")
    print(f"{'':12}{'p10':>12}{'median':>12}{'p90':>12}")
    for name in fields:
        s = stats[name]
        print(f"{name:12}{s['p10']:12,.0f}{s['median']:12,.0f}{s['p90']:12,.0f}")
    print()
    print(f"sims/node             {report['sims_per_node']:.2f}")
    print(f"budget spent          {report['budget_spent_fraction']:.2f} of allowance")
    print(f"moves under 1s        {report['moves_under_1s']} "
          f"({100 * report['moves_under_1s'] / report['moves']:.0f}%)")
    print()
    print(f"THE GAP: the lab measures at {LAB_SIMS} sims; deployment runs at "
          f"{med_sims:,.0f}.")
    print(f"  {gap:,.0f}x = {report['gap_doublings_vs_lab']:.1f} doublings below "
          f"the operating point,")
    print(f"  and {report['gap_doublings_vs_ladder_top']:.1f} doublings above the "
          f"parity ladder's top rung ({LADDER_TOP_SIMS} sims).")
    print()
    print("The exchange rate is NOT constant over this axis (1.15 node-doublings")
    print("per sims-doubling at 200 sims, 0.49 at 1600), so a result measured at")
    print("400 sims needs a second arm at a higher budget before it justifies a")
    print("deployment decision. See docs/OPERATING-POINT.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
