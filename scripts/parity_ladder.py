#!/usr/bin/env python
"""The ladder as node budgets, which needs no cross-population Elo conversion.

    scripts/parity_ladder.py --report

`sim_ladder.py` reports each rung as an ABSOLUTE Elo by walking a
Stockfish-vs-Stockfish ruler. That walk was measured on 2026-08-11 not to
transfer (`scripts/ruler_transfer.py`, z = 7.1, factor 0.69), which withdrew
every absolute and `scale_D` with them.

This file answers the same question without the walk. Each rung is played NEAR
PARITY against Stockfish at a pinned node budget, and reported as the budget it
is actually equivalent to:

    parity_nodes = N * 2 ** (rung_elo / slope)

where `slope` is Elo per doubling of Stockfish nodes AS SUMOFISH EXPERIENCES
IT, measured by the two anchors rather than by Stockfish against itself. The
correction is small by construction: a rung inside +-60 Elo moves its budget by
less than 0.4 doublings, so the slope only has to be roughly right, and it is
the same short range where the two routes already agree.

The exchange rate then has no Elo in it at all:

    node-doublings bought per doubling of SIMULATIONS

which is what `scale_D` was for and cannot be contaminated by the scale
mismatch, because both axes are budgets and neither is a rating.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

M = ROOT / "runs" / "matches"

# Rungs already at parity keep their existing measurements; only the three that
# were played far from it are re-run. (sims, nodes, dirname)
RUNGS = [
    (200, 360, "ladder-sims200-vs-sf360"),
    (400, 845, "ladder-sims400-vs-sf845"),
    (800, 1259, "parity-sims800-vs-sf1259"),
    (1600, 1754, "parity-sims1600-vs-sf1754"),
    (3200, 3046, "parity-sims3200-vs-sf3046"),
]
NEAR_PARITY = 60.0      # Elo. Outside this, re-aim and replay; do not correct.


def slope() -> tuple[float, float]:
    """Elo per doubling of Stockfish NODES, as SumoFish experiences it.

    From the anchors, never from the Stockfish-vs-Stockfish ruler. That is the
    whole point of this file.
    """
    import io
    import contextlib
    import ruler_transfer
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sys.argv = ["ruler_transfer", "--json"]
        ruler_transfer.main()
    pairs = json.loads(buf.getvalue()).get("pairs", [])
    if not pairs:
        raise SystemExit("no anchor pair on disk; the local slope is unmeasured")
    p = max(pairs, key=lambda q: q["doublings"])
    return (p["gap_through_sumofish"] / p["doublings"],
            p["gap_through_sumofish_err"] / p["doublings"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="read only (the default)")
    ap.parse_args()

    D, Derr = slope()
    print(f"local slope, measured through SumoFish: {D:.1f} +-{Derr:.1f} Elo per "
          f"doubling of Stockfish nodes\n")
    print(f"{'sims':>6} {'vs':>9} {'W/D/L':>14} {'rung elo':>13} "
          f"{'PARITY nodes':>14} {'log2':>7}")

    rows, out = [], {"slope": round(D, 1), "slope_err": round(Derr, 1), "rungs": {}}
    for sims, nodes, name in RUNGS:
        st_p = M / name / "status.json"
        if not st_p.exists():
            print(f"{sims:>6} {('SF@' + str(nodes)):>9} {'NOT RUN':>14}")
            continue
        st = json.loads(st_p.read_text())
        parity = nodes * 2 ** (st["elo"] / D)
        # The rung's own error and the slope's, through the same exponent.
        hi = nodes * 2 ** ((st["elo"] + st["err"]) / D)
        lo = nodes * 2 ** ((st["elo"] - st["err"]) / D)
        flag = "" if abs(st["elo"]) <= NEAR_PARITY else "  <-- RE-AIM, not near parity"
        print(f"{sims:>6} {('SF@' + str(nodes)):>9} "
              f"{st['w']}/{st['d']}/{st['l']:<6} {st['elo']:+8.1f} +-{st['err']:<3.0f} "
              f"{parity:>10.0f} [{lo:.0f}, {hi:.0f}] {math.log2(parity):>7.2f}{flag}")
        rows.append((sims, parity))
        out["rungs"][str(sims)] = {"nodes": nodes, "elo": st["elo"], "err": st["err"],
                                   "parity_nodes": round(parity),
                                   "parity_nodes_ci": [round(lo), round(hi)],
                                   "near_parity": abs(st["elo"]) <= NEAR_PARITY,
                                   "games": st["games"]}

    if len(rows) >= 2:
        print(f"\n{'sims doubling':>22}  {'node-doublings bought':>22}")
        rates = []
        for (s0, p0), (s1, p1) in zip(rows, rows[1:]):
            r = math.log2(p1 / p0) / math.log2(s1 / s0)
            rates.append(r)
            print(f"{f'{s0} -> {s1}':>22}  {r:>22.2f}")
        mean = sum(rates) / len(rates)
        out["node_doublings_per_sims_doubling"] = [round(r, 3) for r in rates]
        out["mean_rate"] = round(mean, 3)
        print(f"\nMEAN: a doubling of SumoFish's search buys {mean:.2f} doublings "
              f"of Stockfish's node budget.")
        print("Below 1.0 means the engine falls behind an opponent given the same "
              "relative increase; it is not an Elo claim and does not need one.")
        if not all(out["rungs"][str(s)]["near_parity"] for s, _ in rows):
            print("\nAt least one rung is NOT near parity, so its budget carries a "
                  "long extrapolation along the slope. Re-aim it and replay.")

    (ROOT / "runs/lab/parity-ladder.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ROOT / 'runs/lab/parity-ladder.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
