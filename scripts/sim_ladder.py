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


def ruler_curve() -> tuple[dict, str]:
    """Cumulative Elo as a function of Stockfish node budget, MEASURED.

    The five ruler rungs (2026-08-09, 200 games each, Stockfish vs Stockfish):

        350 -> 700    155.5 +-58      2800 -> 5600   240.8 +-55
        700 -> 1400   205.0 +-57      5600 -> 11200  202.6 +-45
        1400 -> 2800  284.9 +-66

    **CORRECTION, same day.** On the first three of these I claimed the slope
    "steepens by about 2x across the range" and justified this function with it.
    That was reading noise: the weighted mean is 213.5 Elo/doubling and
    chi2 = 2.47 on 4 dof, so the data is entirely consistent with a CONSTANT
    slope and no point is more than 1.1 sigma out. A constant would have priced
    these rungs about as well.

    The chain is kept anyway, for the one reason that survives: it uses the
    measured points themselves and `ruler_at` REFUSES to extrapolate past them,
    where a fitted constant would happily price a rung at 50,000 nodes off five
    measurements that stop at 11,200. Interpolating between measurements is
    discipline; extrapolating from a fit is a guess wearing a number.

    Returns {nodes: cumulative_elo} and a provenance string. Extrapolation past
    the measured ends is refused rather than guessed -- see `ruler_at`.
    """
    import math
    edges = []
    for d in sorted((ROOT / "runs" / "matches").glob("ruler-*-vs-*")):
        st_p = d / "status.json"
        if not st_p.exists():
            continue
        try:
            lo, hi = (int(x) for x in d.name.replace("ruler-", "").split("-vs-"))
            st = json.loads(st_p.read_text())
            if st.get("games", 0) < 100:
                continue
            # elo is A's, A is the LOW budget. `err` is a 95% HALF-WIDTH
            # (elo.py returns 1.96*sigma), and half-widths combine in
            # quadrature exactly as sigmas do, because the 1.96 factors out.
            # Everything below stays in half-widths for that reason.
            edges.append((lo, hi, -st["elo"], st.get("err", 0.0)))
        except Exception:
            continue
    if not edges:
        return {}, "NO RULER MATCHES -- ladder cannot be put on an absolute scale"

    origin = min(min(l, h) for l, h, _, _ in edges)
    nodes = {origin: 0.0}
    # Every node's cumulative Elo is a LINEAR COMBINATION of independent edge
    # measurements, so carry the coefficients alongside the value. That is what
    # makes an interval possible at all: the error on a DIFFERENCE of two nodes
    # is not the sum of their errors, because the shared chain segments cancel
    # exactly. Summing them would double-count; ignoring them (what this file
    # did until 2026-08-11) reports no interval whatsoever.
    coefs = {origin: {}}
    for _ in range(len(edges) + 1):           # relax until the chain resolves
        for i, (lo, hi, gain, _err) in enumerate(edges):
            if lo in nodes and hi not in nodes:
                nodes[hi] = nodes[lo] + gain
                coefs[hi] = dict(coefs[lo]); coefs[hi][i] = coefs[hi].get(i, 0) + 1
            elif hi in nodes and lo not in nodes:
                nodes[lo] = nodes[hi] - gain
                coefs[lo] = dict(coefs[hi]); coefs[lo][i] = coefs[lo].get(i, 0) - 1
    errs = [e for _, _, _, e in edges]
    return nodes, coefs, errs, (f"{len(edges)} measured ruler rungs spanning "
                                f"{min(nodes)}-{max(nodes)} nodes")


def ruler_at(nodes_map: dict, n: int, coefs: dict | None = None):
    """Cumulative Elo at `n` nodes, log-linear between measured points.

    Returns None outside the measured range. A ladder rung priced by
    extrapolating a curve that is known to bend is not a measurement, and the
    honest output is a gap in the table rather than a confident wrong number.

    With `coefs`, returns `(value, coefficient_vector)` so the caller can
    propagate the ruler's own error. Without it, returns the bare value, which
    is the old signature.
    """
    import math
    if not nodes_map:
        return None
    xs = sorted(nodes_map)
    if n < xs[0] or n > xs[-1]:
        return None
    if n in nodes_map:
        return nodes_map[n] if coefs is None else (nodes_map[n], dict(coefs[n]))
    lo = max(x for x in xs if x <= n)
    hi = min(x for x in xs if x >= n)
    f = (math.log2(n) - math.log2(lo)) / (math.log2(hi) - math.log2(lo))
    val = nodes_map[lo] + f * (nodes_map[hi] - nodes_map[lo])
    if coefs is None:
        return val
    # The interpolated point is (1-f)*lo + f*hi in the same edge basis.
    c = {}
    for k, v in coefs[lo].items():
        c[k] = c.get(k, 0.0) + (1 - f) * v
    for k, v in coefs[hi].items():
        c[k] = c.get(k, 0.0) + f * v
    return val, c


def _combine(coef: dict, errs: list) -> float:
    """95% half-width of a linear combination of independent edge measurements."""
    import math
    return math.sqrt(sum((c * errs[i]) ** 2 for i, c in coef.items()))


def _sub(a: dict, b: dict) -> dict:
    """Coefficient vector for (a - b), so shared chain segments cancel."""
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0.0) - v
    return out


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

    nodes_map, coefs, ruler_errs, provenance = ruler_curve()
    import math
    _b = ruler_at(nodes_map, 700, coefs)
    base, base_c = _b if _b else (0.0, {})
    print(f"\nruler: {provenance}")
    for x in sorted(nodes_map):
        print(f"    SF@{x:<6} {nodes_map[x] - base:+8.1f}  (vs SF@700)")
    print(f"\n{'sims':>6}  {'vs':>10}  {'W/D/L':>14}  {'rung elo':>14}  {'ABSOLUTE (vs SF@700)':>22}")
    out = {"shipped": cfg, "ruler_nodes_elo": nodes_map,
           "ruler_provenance": provenance, "rungs": {}}
    for sims, nodes, st in rows:
        if not st:
            print(f"{sims:>6}  {('SF@' + str(nodes)):>10}  {'FAILED':>14}")
            continue
        # Absolute strength on one scale: where SF@nodes sits relative to
        # SF@700 on the MEASURED curve, plus how far this rung beat it.
        _a = ruler_at(nodes_map, nodes, coefs)
        anchor, anchor_c = _a if _a else (None, {})
        wdl = f"{st['w']}/{st['d']}/{st['l']}"
        if anchor is None:
            print(f"{sims:>6}  {('SF@' + str(nodes)):>10}  {wdl:>14}  "
                  f"{st['elo']:+8.1f} +-{st['err']:<4.0f}  "
                  f"{'OUTSIDE RULER':>15}")
            out["rungs"][str(sims)] = {"nodes": nodes, "elo": st["elo"],
                                       "err": st["err"], "absolute_vs_sf700": None}
            continue
        absolute = (anchor - base) + st["elo"]
        # The ruler term DOMINATES and was silently dropped until 2026-08-11.
        # `_sub` is load-bearing: the shared chain segments between `nodes` and
        # 700 cancel, so this is smaller than adding the two nodes' errors, and
        # far larger than the rung's own error, which is all that used to be
        # quoted.
        ruler_err = _combine(_sub(anchor_c, base_c), ruler_errs)
        abs_err = math.hypot(ruler_err, st["err"])
        print(f"{sims:>6}  {('SF@' + str(nodes)):>10}  {wdl:>14}  "
              f"{st['elo']:+8.1f} +-{st['err']:<4.0f}  "
              f"{absolute:+10.1f} +-{abs_err:<4.0f}")
        out["rungs"][str(sims)] = {"nodes": nodes, "elo": st["elo"], "err": st["err"],
                                   "w": st["w"], "d": st["d"], "l": st["l"],
                                   "absolute_vs_sf700": round(absolute, 1),
                                   "absolute_err": round(abs_err, 1),
                                   "absolute_err_ruler_term": round(ruler_err, 1),
                                   "_coef": {str(k): round(v, 4)
                                             for k, v in _sub(anchor_c, base_c).items()}}
    done = [(s, r) for s, r in out["rungs"].items()
            if r.get("absolute_vs_sf700") is not None]
    if len(done) >= 2:
        done.sort(key=lambda x: int(x[0]))
        (slo, rlo), (shi, rhi) = done[0], done[-1]
        span = math.log2(int(shi) / int(slo))
        per = (rhi["absolute_vs_sf700"] - rlo["absolute_vs_sf700"]) / span
        # Same basis arithmetic as the absolutes. The two rung errors are
        # independent of each other and of the ruler; the ruler coefficients
        # do NOT cancel between the top and bottom rung (they price opposite
        # ends of the chain), which is why this term dominates.
        dc = _sub({int(k): v for k, v in rhi["_coef"].items()},
                  {int(k): v for k, v in rlo["_coef"].items()})
        ruler_term = _combine(dc, ruler_errs)
        per_err = math.sqrt(ruler_term ** 2 + rhi["err"] ** 2 + rlo["err"] ** 2) / span
        out["elo_per_sim_doubling"] = round(per, 1)
        out["elo_per_sim_doubling_err"] = round(per_err, 1)
        out["elo_per_sim_doubling_err_ruler_term"] = round(ruler_term / span, 1)
        print(f"\nEND TO END: {per:.0f} +-{per_err:.0f} Elo per doubling of SEARCH, "
              f"{slo} -> {shi} sims. This is the number `scale_D` wanted, "
              f"and unlike the withdrawn chain it is two absolute measurements "
              f"differenced, not four relative rungs summed.")
        print(f"  Of that +-{per_err:.1f}, the RULER contributes "
              f"+-{ruler_term / span:.1f} and the two rungs "
              f"+-{math.hypot(rhi['err'], rlo['err']) / span:.1f}. The ruler is "
              f"Stockfish vs Stockfish and therefore CPU-only: more games there "
              f"is the cheapest error reduction available to this project.")
    for r in out["rungs"].values():
        r.pop("_coef", None)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
