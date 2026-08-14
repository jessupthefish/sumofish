#!/bin/bash
# Plan 1.4: price `compile` ALONE, at the clock, on an exclusive box.
#
# Why this is the most valuable single measurement in the plan. `compile`
# (CUDA graphs) has been switched off since 2026-07-30 on the strength of a
# -168 Elo reading whose entire evidence base is TWO arms of 20 games each,
# both with compile bundled with `dedup`, one of them explicitly named
# "contended". Forty games, never isolated. PHILOSOPHY's own discipline section
# prices a 24-game match at +-200. Meanwhile the flag is a measured 6.58 -> 3.53
# ms at batch 64 on a search that Arm C now shows is ~85% forward passes, and
# the diagnosed -168 mechanism is dedup's (fake search via collapsed descents),
# which compile cannot produce: it sends the same rows.
#
# It also commissions the clock instrument. A 1.9x speedup on a network-bound
# search that fails to register at +-14.7 Elo means the instrument or the
# operating-point model is wrong, and Phase 4 depends on that instrument.
#
# Fresh seed: 7, 4242 and 99 are spent across 88 of 104 archived arms.
set -uo pipefail
cd /home/nomad/dev/active/sumofish
PY=.venv/bin/python
SEED=${SEED:-20260814}

echo "########## 0. smoke: does an isolated compile arm even RUN? ##########"
# Never been run alone in this project's history, so it gets two games at 20
# sims before anything commits ten hours. Fixed-sims deliberately: this is an
# existence check, not a measurement, and it must not trip the idle assertion.
rm -rf runs/matches/_compile-smoke
if ! timeout 1800 $PY scripts/match.py --name _compile-smoke --games 2 --sims 20 \
        --a-vloss-fix --b-vloss-fix --a-compile \
        --a-label "compile ON" --b-label "compile OFF" --no-sprt 2>&1 | tail -12; then
    echo "SMOKE FAILED: an isolated compile arm does not run. Not starting the match."
    rm -rf runs/matches/_compile-smoke
    exit 1
fi
if [ ! -s runs/matches/_compile-smoke/games.jsonl ]; then
    echo "SMOKE FAILED: no games recorded. Not starting the match."
    rm -rf runs/matches/_compile-smoke
    exit 1
fi
echo
echo "  smoke: the per-game search block, which is what prices a flag at a clock"
$PY -c "
import json
rows=[json.loads(l) for l in open('runs/matches/_compile-smoke/games.jsonl') if l.strip()]
for i,g in enumerate(rows):
    s=g.get('search',{})
    print(f'    game {i}: A evals {s.get(\"white_evals\" if g[\"a_white\"] else \"black_evals\")}, '
          f'B evals {s.get(\"black_evals\" if g[\"a_white\"] else \"white_evals\")}')
"
rm -rf runs/matches/_compile-smoke
echo "  smoke passed"

echo
echo "########## 1. the gate: compile ALONE, 600 games, --time 0.5 ##########"
echo "  seed $SEED (fresh; 7/4242/99 are spent across 88 of 104 archived arms)"
timeout 86400 $PY scripts/match.py \
    --name compile-time --games 600 --time 0.5 --seed "$SEED" \
    --a-vloss-fix --b-vloss-fix --a-compile \
    --a-label "compile ON" --b-label "compile OFF" \
    --no-sprt
RC=$?

echo
echo "########## 2. what it measured, beyond the Elo ##########"
$PY - <<'PYEOF'
import json, statistics as st
from pathlib import Path
p = Path("runs/matches/compile-time/games.jsonl")
if not p.exists():
    raise SystemExit("no games.jsonl")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
rows = [r for r in rows if r.get("search")]
if not rows:
    raise SystemExit("no search blocks: this match predates the 0.10 recording work")
ra = []
for r in rows:
    s = r["search"]
    a = "white" if r["a_white"] else "black"
    b = "black" if r["a_white"] else "white"
    if s[f"{b}_evals"]:
        ra.append(s[f"{a}_evals"] / s[f"{b}_evals"])
m = st.fmean(ra)
half = 1.96 * st.stdev(ra) / len(ra) ** 0.5 if len(ra) > 1 else float("nan")
print(f"\n  SEARCH: compile ON got {m:.3f} +-{half:.3f} of OFF's evaluations "
      f"in the same time, over {len(ra)} games")
print("  This is the number the flag is actually about, and it closes far faster")
print("  than the Elo. Above 1.0 means compile is buying search. If it sits at")
print("  1.000 the flag is not reaching the engine and the Elo means nothing:")
print("  that is a bug report, not a null result.")
PYEOF
exit $RC
