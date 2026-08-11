#!/usr/bin/env python
"""Tune the exploration constant and FPU against a FIXED EXTERNAL opponent.

    scripts/tune_search.py                    # the full sweep
    scripts/tune_search.py --games 400        # a faster, wider-interval screen
    scripts/tune_search.py --report           # just re-read what already ran

Why this exists, and why it is shaped this way.

`match-9m-long` spent 9h38m comparing two checkpoints of one lineage and
returned +3.5 +-26.4, with **80 of 100 games drawn by threefold repetition**.
Two configurations that share a value net evaluate balanced positions
identically and neither can make progress: PHILOSOPHY point 4's mirror-match
blindness. Exploration and FPU change move SELECTION but not evaluation, so a
head-to-head sweep would sit in exactly the same trap.

So every arm here plays **Stockfish at a pinned node budget** instead. The
opponent shares no evaluation with us, the draw rate falls from 85% to ~33%,
and each arm gets an ABSOLUTE score that is comparable across arms rather than
only against whatever it happened to be paired with.

Two consequences worth stating plainly:

1. Comparing two arms means differencing two independent intervals, so the
   error on a DIFFERENCE is ~1.4x the error on either arm. This is a screen for
   "is the current value badly wrong", not an instrument for splitting two
   values ten Elo apart. Read the intervals, not the ranking.
2. Every arm runs with the SAME `--seed`, so every arm sees the same openings
   in the same order against the same opponent. That is a real variance
   reduction across arms and costs nothing.

Staged rather than a grid: the full cross is 20 arms and most of a day, and the
two parameters are not strongly coupled at this resolution. Stage 1 sweeps
`c_puct_init` at the current FPU; stage 2 sweeps FPU at stage 1's winner.

Note it is `c_puct_init` and NOT `c_puct`. See the constant below: under the
schedule that ships, `c_puct` is never read, and a sweep of it returns identical
arms that read as "no effect" rather than "not connected".

**This script never changes anything.** It writes `runs/lab/tune-search.json`
and prints a table. Applying a value is a separate, deliberate act, because a
tuning result that clears zero by less than its own interval is not a result.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "runs" / "lab"
OUT = LAB / "tune-search.json"

# The engine's current values, READ FROM THE LIBRARY rather than copied here.
# Both sweeps bracket these rather than starting from a guess, so "the value we
# already ship" is always an arm and always measured under identical conditions
# to its challengers.
#
# NOT `c_puct`. Under AlphaZero's schedule -- the default, and what ships --
# `c_puct_at()` returns `ln((1+N+base)/base) + c_puct_init` and never reads
# `c_puct` at all. A c_puct sweep on 2026-08-09 returned five identical arms
# for values 1.0 through 4.5, with identical move hashes, because every one of
# them was silently running the hardcoded 1.25. `--cpuct-init` was added to
# match.py the same day; `tests/verify_cpuct_binding.py` guards both facts.
#
# These were HARDCODED at 1.25/-0.2 until 2026-08-11, and v6 shipped 0.875/-0.05
# on 08-09 without updating them. A sweep run in that window would have measured
# every arm against a baseline the engine had stopped using two days earlier,
# and the honesty gate below would have "fallen back to shipped" onto a value
# that was not shipped. `sumofish/engines/search_engine.py` passes no search
# constants, so the library defaults ARE the deployment; reading them here is
# the same repair `tests/identity_engine.py` already applies for the same
# reason. A constant that has to be kept in sync by hand will eventually not be.
def _shipped() -> dict:
    """The deployed search constants, from `MCTS.__init__`'s own defaults."""
    import inspect

    sys.path.insert(0, str(ROOT))
    from sumofish.mcts import MCTS

    p = inspect.signature(MCTS.__init__).parameters
    return {k: p[k].default for k in ("c_puct_init", "fpu")}


SHIPPED = _shipped()
CURRENT_CPUCT_INIT = SHIPPED["c_puct_init"]
CURRENT_FPU = SHIPPED["fpu"]

# The parity point, recalibrated 2026-08-09 (8W 8D 8L over 24 games). Parity is
# where a match carries the most information about the SIZE of a difference: a
# budget either side wins outright answers "which is stronger" and nothing else.
ANCHOR_NODES = 700
SIMS = 400
SEED = 4242

# The schedule's own additive term spans ~1.25 to ~2.65 across a 15+10 search
# (N=0 to N~60,000), so these bracket the shipped value by roughly the width the
# schedule itself moves. Wider than that is not a tune, it is a different engine.
CPUCT_INIT_ARMS = [0.5, 0.875, 1.25, 1.75, 2.5]
FPU_ARMS = [-0.5, -0.35, -0.2, -0.05]

# The shipped value has to BE an arm, or the honesty gate below silently loses
# its baseline: `results["stage1"].get(str(CURRENT_CPUCT_INIT))` returns {} and
# the comparison it gates on is skipped rather than failed. Assert it here
# rather than discovering it after the GPU hours are spent.
assert CURRENT_CPUCT_INIT in CPUCT_INIT_ARMS, (
    f"shipped c_puct_init={CURRENT_CPUCT_INIT} is not in CPUCT_INIT_ARMS "
    f"{CPUCT_INIT_ARMS}; the sweep would have no baseline arm")
assert CURRENT_FPU in FPU_ARMS, (
    f"shipped fpu={CURRENT_FPU} is not in FPU_ARMS {FPU_ARMS}; "
    f"the sweep would have no baseline arm")


def run_arm(name: str, games: int, cpuct_init: float, fpu: float) -> dict:
    """One arm against Stockfish. Returns its status dict, or a resumed one."""
    outdir = ROOT / "runs" / "matches" / name
    status = outdir / "status.json"
    if status.exists():
        st = json.loads(status.read_text())
        if st.get("games", 0) >= games:
            print(f"  {name}: already complete ({st['games']} games), reusing")
            return st

    argv = [
        str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/match.py"),
        "--value", str(ROOT / "runs/value.pt"),
        "--policy", str(ROOT / "runs/policy.pt"),
        "--sims", str(SIMS), "--core", "rust",
        # Not optional. store_true defaults to OFF, and without it every arm
        # measures an engine that has not played a rated game since 07-30. This
        # is the error that invalidated the entire exchange ladder.
        "--a-vloss-fix",
        "--a-cpuct-init", str(cpuct_init), "--a-fpu", str(fpu),
        "--a-label", f"ci{cpuct_init}-fpu{fpu}",
        "--b-stockfish-nodes", str(ANCHOR_NODES),
        "--b-label", f"SF@{ANCHOR_NODES}n",
        "--games", str(games), "--seed", str(SEED),
        # An interval, not a verdict. The SPRT stops at the first crossing and
        # hands back the widest interval it will accept; here the interval IS
        # the result.
        "--no-sprt",
        "--name", name,
    ]
    t0 = time.time()
    proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-8:])
        print(f"  {name}: FAILED rc={proc.returncode}\n{tail}", flush=True)
        return {}
    st = json.loads(status.read_text())
    st["seconds"] = round(time.time() - t0)
    print(f"  {name}: {st['w']}W {st['d']}D {st['l']}L  "
          f"elo {st.get('elo', 0):+.1f} +-{st.get('err', 0):.0f}  "
          f"({st['seconds']}s)", flush=True)
    return st


def score(st: dict) -> float:
    n = st.get("games", 0)
    return (st["w"] + 0.5 * st["d"]) / n if n else 0.0


def table(rows: list[tuple[str, dict]], label: str) -> str:
    out = [f"\n{label}", f"{'arm':>22}  {'W/D/L':>14}  {'score':>7}  {'elo':>18}"]
    for name, st in rows:
        if not st:
            out.append(f"{name:>22}  {'FAILED':>14}")
            continue
        wdl = f"{st['w']}/{st['d']}/{st['l']}"
        out.append(f"{name:>22}  {wdl:>14}  {score(st) * 100:6.1f}%  "
                   f"{st.get('elo', 0):+8.1f} +-{st.get('err', 0):<6.0f}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=800,
                    help="games per arm. 800 is ~43min and ~+-25 Elo at this "
                         "draw rate; the error on a DIFFERENCE is ~1.4x that.")
    ap.add_argument("--report", action="store_true",
                    help="re-read completed arms and reprint, running nothing")
    args = ap.parse_args()
    LAB.mkdir(parents=True, exist_ok=True)

    results: dict = {"anchor_nodes": ANCHOR_NODES, "sims": SIMS, "seed": SEED,
                     "games_per_arm": args.games,
                     "current": {"c_puct_init": CURRENT_CPUCT_INIT,
                                 "fpu": CURRENT_FPU}}

    print(f"stage 1: c_puct_init at fpu={CURRENT_FPU}, "
          f"{args.games} games/arm vs Stockfish@{ANCHOR_NODES}n", flush=True)
    stage1 = []
    for c in CPUCT_INIT_ARMS:
        name = f"tune-ci{c}"
        st = {} if args.report and not (ROOT / "runs/matches" / name / "status.json").exists() \
            else (json.loads((ROOT / "runs/matches" / name / "status.json").read_text())
                  if args.report else run_arm(name, args.games, c, CURRENT_FPU))
        stage1.append((f"c_puct_init={c}", st))
        results.setdefault("stage1", {})[str(c)] = st
    print(table(stage1, "stage 1 -- c_puct_init"), flush=True)

    ranked = [(n, s) for n, s in stage1 if s]
    if not ranked:
        print("\nevery stage-1 arm failed; not starting stage 2")
        OUT.write_text(json.dumps(results, indent=2))
        return 1
    best_name, best_st = max(ranked, key=lambda r: score(r[1]))
    best_c = float(best_name.split("=")[1])
    results["stage1_winner"] = {"c_puct_init": best_c, "score": score(best_st),
                                "elo": best_st.get("elo"), "err": best_st.get("err")}

    # Honesty gate: if the winner is not separated from the shipped value by
    # more than the difference's own error, say so rather than let a ranking
    # imply a finding. Stage 2 still runs, at the SHIPPED value, because FPU is
    # worth measuring either way and pretending c_puct moved would poison it.
    cur_st = results["stage1"].get(str(CURRENT_CPUCT_INIT)) or {}
    if cur_st:
        gap = (best_st.get("elo", 0) - cur_st.get("elo", 0))
        differr = 1.41 * max(best_st.get("err", 0), cur_st.get("err", 0))
        results["stage1_separated"] = bool(abs(gap) > differr)
        print(f"\nwinner c_puct_init={best_c} is {gap:+.1f} Elo vs the shipped "
              f"{CURRENT_CPUCT_INIT}, against a difference error of ~{differr:.0f}. "
              f"{'SEPARATED' if abs(gap) > differr else 'NOT separated -- treat as a tie'}",
              flush=True)
        if abs(gap) <= differr:
            best_c = CURRENT_CPUCT_INIT

    print(f"\nstage 2: fpu at c_puct_init={best_c}", flush=True)
    stage2 = []
    for f in FPU_ARMS:
        name = f"tune-fpu{f}-ci{best_c}"
        st = {} if args.report and not (ROOT / "runs/matches" / name / "status.json").exists() \
            else (json.loads((ROOT / "runs/matches" / name / "status.json").read_text())
                  if args.report else run_arm(name, args.games, best_c, f))
        stage2.append((f"fpu={f}", st))
        results.setdefault("stage2", {})[str(f)] = st
    print(table(stage2, "stage 2 -- fpu"), flush=True)

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")
    print("NOTHING WAS APPLIED. Change sumofish/mcts.py deliberately, and only "
          "if an arm clears the shipped value by more than the difference error.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
