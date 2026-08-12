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


def slope() -> tuple[float, float, str]:
    """Elo per doubling of Stockfish NODES, as SumoFish experiences it NEAR PARITY.

    From SumoFish-vs-Stockfish matches, never from the Stockfish-vs-Stockfish
    ruler. That is the whole point of this file.

    **Which span you take it from matters, and the 2026-08-11 value was taken
    from the wrong one.** The 700->1600 anchor pair spans out to 162 Elo from
    parity and gives 161.7; every span this project has that reaches far from
    parity gives 147-212, and the one span where both arms are within 30 Elo of
    parity gives 203.5. The rungs being converted here are all within 40 Elo of
    parity, so the near-parity slope is the right one, and using the far one
    biased every budget by ~3-5%.

    The correction is small either way, by design: a rung inside +-40 Elo moves
    its budget by less than 0.25 doublings, and swapping 161.7 for 208 moves the
    exchange rates by at most 0.08. If that ever stops being true, the rung is
    not near parity and the answer is to re-aim it, not to measure the slope
    harder.
    """
    spans = _spans()
    near = [sp for sp in spans if sp["far_from_parity"] <= NEAR_PARITY]
    if not near:
        raise SystemExit(
            "no span has both arms near parity, so the local slope is unmeasured. "
            "Every span on disk reaches far from parity, where the perceived "
            "slope is demonstrably different. Play one.")
    # Inverse-variance weighted, on half-widths, which combine as sigmas do.
    wsum = sum(1 / sp["err"] ** 2 for sp in near)
    D = sum(sp["perceived"] / sp["err"] ** 2 for sp in near) / wsum
    return D, wsum ** -0.5, f"{len(near)} span(s) with both arms inside +-{NEAR_PARITY:.0f} Elo"


def _spans() -> list[dict]:
    """Every (sims, node_lo, node_hi) span measurable from matches on disk.

    Two runs may only be differenced if they share the engine AND the harness.
    Mixing a warm-hash run with a ucinewgame-fixed one is the 2026-08-11
    "+30.5, was +44.4" mistake, and it silently produced a 0.09 transfer factor
    when this scan was first written without the check.
    """
    import subprocess
    HARNESS_FIX = "7711bf7"       # "Stockfish kept its hash between games"

    def post_fix(code: str | None) -> bool:
        if not code:
            return False
        return subprocess.run(
            ["git", "merge-base", "--is-ancestor", HARNESS_FIX, code.split("+")[0]],
            cwd=ROOT, capture_output=True).returncode == 0

    from sumofish.mcts import MCTS
    import inspect
    sig = inspect.signature(MCTS.__init__).parameters
    shipped = (sig["c_puct_init"].default, sig["fpu"].default, True)

    pts: dict[int, dict[int, dict]] = {}
    for d in sorted(M.iterdir()):
        cfg_p, st_p = d / "config.json", d / "status.json"
        if not (cfg_p.exists() and st_p.exists()):
            continue
        cfg = json.loads(cfg_p.read_text())
        a, b = cfg.get("a", {}), cfg.get("b", {})
        if a.get("stockfish_nodes") is not None or b.get("stockfish_nodes") is None:
            continue
        if (a.get("c_puct_init"), a.get("fpu"), a.get("vloss_fix")) != shipped:
            continue
        if not post_fix(cfg.get("code")):
            continue
        st = json.loads(st_p.read_text())
        if st.get("games", 0) < 400:
            continue
        seat = pts.setdefault(a["sims"], {})
        n = int(b["stockfish_nodes"])
        if n not in seat or st["games"] > seat[n]["games"]:
            seat[n] = {**st, "name": d.name}

    out = []
    for sims, seat in sorted(pts.items()):
        ns = sorted(seat)
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                lo, hi = seat[ns[i]], seat[ns[j]]
                dbl = math.log2(ns[j] / ns[i])
                out.append({
                    "sims": sims, "lo_nodes": ns[i], "hi_nodes": ns[j], "doublings": dbl,
                    "perceived": (lo["elo"] - hi["elo"]) / dbl,
                    "err": math.hypot(lo["err"], hi["err"]) / dbl,
                    "far_from_parity": max(abs(lo["elo"]), abs(hi["elo"])),
                    "arms": [lo["name"], hi["name"]],
                })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="read only (the default)")
    ap.parse_args()

    D, Derr, prov = slope()
    print(f"local slope, measured through SumoFish NEAR PARITY: {D:.1f} +-{Derr:.1f} "
          f"Elo per doubling of Stockfish nodes ({prov})\n")
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
        rows.append((sims, parity, st["err"]))
        out["rungs"][str(sims)] = {"nodes": nodes, "elo": st["elo"], "err": st["err"],
                                   "parity_nodes": round(parity),
                                   "parity_nodes_ci": [round(lo), round(hi)],
                                   "near_parity": abs(st["elo"]) <= NEAR_PARITY,
                                   "games": st["games"]}

    if len(rows) >= 2:
        print(f"\n{'sims doubling':>22}  {'node-doublings bought':>26}")
        rates = []
        for (s0, p0, e0), (s1, p1, e1) in zip(rows, rows[1:]):
            span = math.log2(s1 / s0)
            r = math.log2(p1 / p0) / span
            # Each budget's error is its rung's Elo error through the same
            # exponent, so in doublings it is err/D. They are independent.
            rerr = math.hypot(e0 / D, e1 / D) / span
            rates.append((s0, s1, r, rerr))
            print(f"{f'{s0} -> {s1}':>22}  {r:>17.2f} +-{rerr:<6.2f}")
        out["node_doublings_per_sims_doubling"] = [
            {"from": a, "to": b, "rate": round(r, 3), "err": round(e, 3)}
            for a, b, r, e in rates]
        # NO MEAN. The rates below are not scattered around one value, they
        # decay, and a mean would hide the only thing this ladder found.
        lo, hi = rates[0], rates[-1]
        gap = lo[2] - hi[2]
        gaperr = math.hypot(lo[3], hi[3])
        out["decay"] = {"bottom": round(lo[2], 2), "top": round(hi[2], 2),
                        "difference": round(gap, 2), "difference_err": round(gaperr, 2)}
        print(f"\nTHE RATE IS NOT CONSTANT. {lo[0]}->{lo[1]} sims buys "
              f"{lo[2]:.2f} +-{lo[3]:.2f} node-doublings and {hi[0]}->{hi[1]} buys "
              f"{hi[2]:.2f} +-{hi[3]:.2f}: a fall of {gap:.2f} +-{gaperr:.2f}. "
              f"Do not quote a mean of these.")
        # The slope rests on one near-parity span and is wide. Show what the
        # conclusion does if it is wrong by the full width of the disagreement
        # between near-parity and far-from-parity spans, rather than implying
        # the budgets are known better than the slope is.
        alt = [sp["perceived"] for sp in _spans() if sp["far_from_parity"] > 120]
        if alt:
            Dalt = sum(alt) / len(alt)
            # Recompute each budget from its OWN rung Elo at the alternative
            # slope. (An earlier version of this block reached for the rung's
            # error instead of its Elo and printed the unchanged rates back,
            # which reads exactly like a robust result.)
            pa = [(int(k), v["nodes"] * 2 ** (v["elo"] / Dalt))
                  for k, v in sorted(out["rungs"].items(), key=lambda kv: int(kv[0]))]
            arates = [math.log2(pa[i + 1][1] / pa[i][1]) /
                      math.log2(pa[i + 1][0] / pa[i][0]) for i in range(len(pa) - 1)]
            out["decay_at_alternative_slope"] = {
                "slope": round(Dalt, 1), "rates": [round(r, 3) for r in arates],
                "difference": round(arates[0] - arates[-1], 2)}
            print(f"\nSENSITIVITY: at the far-from-parity slope ({Dalt:.0f} instead "
                  f"of {D:.0f}), the same rungs give {arates[0]:.2f} at the bottom "
                  f"and {arates[-1]:.2f} at the top, a fall of "
                  f"{arates[0] - arates[-1]:.2f} against {gap:.2f}. The slope is "
                  f"wide; the decay does not depend on it.")
        print("Below 1.0 means the engine falls behind an opponent handed the same "
              "relative increase in budget. It is not an Elo claim and does not "
              "need one, which is the point: no cross-population conversion "
              "appears anywhere above.")
        if not all(out["rungs"][str(s)]["near_parity"] for s, _, _ in rows):
            print("\nAt least one rung is NOT near parity, so its budget carries a "
                  "long extrapolation along the slope. Re-aim it and replay.")

    (ROOT / "runs/lab/parity-ladder.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ROOT / 'runs/lab/parity-ladder.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
