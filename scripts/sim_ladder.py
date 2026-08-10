#!/usr/bin/env python
"""How much is a doubling of SEARCH worth, on an absolute scale?

    scripts/sim_ladder.py                # run the ladder
    scripts/sim_ladder.py --report       # re-read what already ran

This replaces the old exchange ladder (`sims-400-200`, `sims-800-400`, ...),
which is WITHDRAWN and should not be re-run in its original form.

The old design was a CHAIN of relative rungs: 200 vs 400, then 400 vs 800, and
so on, each rung a match of the engine against itself at two budgets. Two
problems, and the second is fatal to the thing the ladder is for:

1. **Error accumulates down a chain.** Four rungs at +-35 each, summed to price
   a 16x range, carry +-70. Every number derived from the total inherits that.
2. **The chain has no zero.** It says a doubling is worth D Elo and can never
   say how strong the engine IS, so `scale_bar`-style arguments have to assume
   the chain's own scale is meaningful outside itself.

So: measure each sim count against a FIXED EXTERNAL opponent instead, and let
the differences fall out. Every rung is an absolute measurement against
Stockfish, no rung depends on any other, and the result is a curve rather than
a chain.

**The ruler is measured, not assumed.** Converting a Stockfish node budget into
Elo needs to know what a doubling of Stockfish's own nodes is worth, and that is
measured directly by `ruler-*` matches (Stockfish vs Stockfish, CPU-only, so
they cost no GPU). As of 2026-08-09 the 700->1400 rung is -205.0 +-57, and
700->845 is -45.4 +-46. That second one independently corroborates the anchor:
SumoFish@400 measured +44.4 +-12 over SF@700, and SF@845 measures +45.4 +-46
over SF@700, so the interpolated "parity at ~845 nodes" holds by two
independent routes. Do not hardcode a slope here; read the ruler.

**Each rung is paired with a budget near its own parity**, not with one fixed
budget for all rungs. A rung the engine wins 95% of has a huge interval and
says nothing about the SIZE of anything, which is the same reason the anchor
retired its 40-node point. Predicted parity uses ~200 Elo per sim doubling from
the withdrawn ladder -- good enough to AIM a rung, never quoted as a result.
"""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
LAB = ROOT / "runs" / "lab"
OUT = LAB / "sim-ladder.json"

# (sims, stockfish_nodes, games). Node budgets aim each rung near parity using
# ~200 Elo/sim-doubling against the measured ~170-205 Elo/node-doubling. Game
# counts fall as the rungs get slower; the effect being measured is hundreds of
# Elo, so +-25 is ample and +-15 is waste.
RUNGS = [
    (200, 360, 800),
    (400, 845, 800),
    (800, 1970, 600),
    (1600, 4600, 500),
    (3200, 10700, 400),
]


def shipped() -> dict:
    """The values the LIVE engine uses, read at runtime rather than hardcoded.

    `search_engine.py` constructs its MCTS with only value/policy/simulations/
    batch, so the library defaults ARE the deployment. Reading them here means
    this ladder measures whatever is currently shipped, and keeps measuring the
    right thing across a tuning change instead of silently pinning yesterday's.
    """
    from sumofish.mcts import MCTS
    p = inspect.signature(MCTS.__init__).parameters
    return {"c_puct_init": p["c_puct_init"].default, "fpu": p["fpu"].default}


def run_rung(sims: int, nodes: int, games: int, cfg: dict) -> dict:
    name = f"ladder-sims{sims}-vs-sf{nodes}"
    status = ROOT / "runs" / "matches" / name / "status.json"
    if status.exists():
        st = json.loads(status.read_text())
        if st.get("games", 0) >= games:
            print(f"  {name}: complete ({st['games']}), reusing", flush=True)
            return st
    argv = [
        str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/match.py"),
        "--value", str(ROOT / "runs/value.pt"),
        "--policy", str(ROOT / "runs/policy.pt"),
        "--sims", str(sims), "--core", "rust", "--a-vloss-fix",
        "--a-cpuct-init", str(cfg["c_puct_init"]), "--a-fpu", str(cfg["fpu"]),
        "--a-label", f"sumofish@{sims}sims",
        "--b-stockfish-nodes", str(nodes), "--b-label", f"SF@{nodes}n",
        "--games", str(games), "--no-sprt", "--seed", "4242", "--name", name,
    ]
    t0 = time.time()
    proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  {name}: FAILED rc={proc.returncode}\n"
              + "\n".join((proc.stdout + proc.stderr).splitlines()[-6:]), flush=True)
        return {}
    st = json.loads(status.read_text())
    st["seconds"] = round(time.time() - t0)
    print(f"  {name}: {st['w']}W {st['d']}D {st['l']}L  "
          f"elo {st.get('elo', 0):+.1f} +-{st.get('err', 0):.0f}  "
          f"draws {st['d'] / st['games'] * 100:.0f}%  ({st['seconds']}s)", flush=True)
    return st


def ruler_slope() -> tuple[float, str]:
    """Elo per doubling of Stockfish nodes, from the measured ruler matches."""
    import math
    pts = []
    for d in (ROOT / "runs" / "matches").glob("ruler-*-vs-*"):
        st_p = d / "status.json"
        if not st_p.exists():
            continue
        try:
            lo, hi = d.name.replace("ruler-", "").split("-vs-")
            st = json.loads(st_p.read_text())
            if st.get("games", 0) < 100:
                continue
            doublings = math.log2(int(hi) / int(lo))
            pts.append((-st["elo"] / doublings, st["games"]))
        except Exception:
            continue
    if not pts:
        return 190.0, "DEFAULT 190 (no ruler matches found -- treat as assumed)"
    slope = sum(s * n for s, n in pts) / sum(n for _, n in pts)
    return slope, f"measured from {len(pts)} ruler rungs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    LAB.mkdir(parents=True, exist_ok=True)
    cfg = shipped()
    print(f"shipped config: c_puct_init={cfg['c_puct_init']} fpu={cfg['fpu']}", flush=True)

    rows = []
    for sims, nodes, games in RUNGS:
        name = f"ladder-sims{sims}-vs-sf{nodes}"
        st_p = ROOT / "runs" / "matches" / name / "status.json"
        if args.report:
            st = json.loads(st_p.read_text()) if st_p.exists() else {}
        else:
            st = run_rung(sims, nodes, games, cfg)
        rows.append((sims, nodes, st))

    slope, provenance = ruler_slope()
    import math
    print(f"\nruler: {slope:.0f} Elo per doubling of Stockfish nodes ({provenance})")
    print(f"\n{'sims':>6}  {'vs':>10}  {'W/D/L':>14}  {'rung elo':>14}  {'ABSOLUTE (vs SF@700)':>22}")
    out = {"shipped": cfg, "ruler_elo_per_doubling": slope,
           "ruler_provenance": provenance, "rungs": {}}
    for sims, nodes, st in rows:
        if not st:
            print(f"{sims:>6}  {('SF@' + str(nodes)):>10}  {'FAILED':>14}")
            continue
        # Absolute strength on one scale: where SF@nodes sits relative to
        # SF@700, plus how far this rung beat (or lost to) it.
        absolute = slope * math.log2(nodes / 700) + st["elo"]
        wdl = f"{st['w']}/{st['d']}/{st['l']}"
        print(f"{sims:>6}  {('SF@' + str(nodes)):>10}  {wdl:>14}  "
              f"{st['elo']:+8.1f} +-{st['err']:<4.0f}  {absolute:+15.1f}")
        out["rungs"][str(sims)] = {"nodes": nodes, "elo": st["elo"], "err": st["err"],
                                   "w": st["w"], "d": st["d"], "l": st["l"],
                                   "absolute_vs_sf700": round(absolute, 1)}
    done = [(s, r["absolute_vs_sf700"]) for s, r in
            ((s, out["rungs"][s]) for s in out["rungs"])]
    if len(done) >= 2:
        done.sort(key=lambda x: int(x[0]))
        lo, hi = done[0], done[-1]
        per = (hi[1] - lo[1]) / math.log2(int(hi[0]) / int(lo[0]))
        out["elo_per_sim_doubling"] = round(per, 1)
        print(f"\nEND TO END: {per:.0f} Elo per doubling of SEARCH, "
              f"{lo[0]} -> {hi[0]} sims. This is the number `scale_D` wanted, "
              f"and unlike the withdrawn chain it is two absolute measurements "
              f"differenced, not four relative rungs summed.")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
