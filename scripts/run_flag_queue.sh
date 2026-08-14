#!/usr/bin/env bash
# The three flag questions left open on 2026-08-13, in priority order.
#
# Each arm is derived from a finding made that day; the reasoning is in STATE.md
# and LAB-NOTES.md and is summarised per-arm below. Read those before changing an
# arm's parameters, because the parameters ARE the argument.
#
#   1. matedist-3200   mate_distance at 8x the budget.
#      docs/OPERATING-POINT.md: the lab measures at 400 sims and deployment runs
#      at ~231,000, which is 9.2 doublings, and the parity ladder showed the
#      exchange rate is NOT constant along that axis. The rule that earns is
#      that anything justifying a deployment decision needs a second arm at a
#      higher budget. transfer-400/transfer-3200 is the precedent and this
#      mirrors it exactly.
#
#   2. dedup-time      dedup ALONE at a fixed CLOCK.
#      The -168 Elo that keeps CHESSGPU_DEDUP off is 20 games, measured jointly
#      with CHESSGPU_COMPILE, pre-vloss_fix, and its stated mechanism compares
#      unique_evaluations across the flag -- which cannot work, because that
#      counter equals the row count when dedup is off (rust/src/tree.rs:558).
#      tests/verify_dedup.py section 3 now shows byte-identical trees and 54%
#      fewer rows at batch 64 on the deployed engine.
#
#      FIXED CLOCK, not fixed sims, and that is the whole design. At fixed
#      simulations dedup is PROVEN identity-preserving, so such a match would
#      measure nothing but batch-shape float noise. Its entire benefit is that
#      each simulation costs fewer network rows, which only shows up when the
#      clock is what stops the search.
#
#   3. matedist-time   mate_distance at a fixed clock.
#      Same reasoning one step further: verify_mate --real-nets measured the
#      tree 52% smaller on forced mates, and a fixed-simulation match is blind
#      to a per-simulation cost saving by construction.
#
#   4. timemgmt-tc    instamove + early stopping, on a real clock.
#      This arm was impossible until --tc landed on 2026-08-13: --time is
#      seconds per MOVE, and under it time saved on one move goes nowhere,
#      which is the entire thing early stopping exploits.
#
#      READ THE CLOCK COLUMN FIRST, NOT THE ELO. The effect is small -- a
#      4-game smoke test had the flagged arm using 96% of the other's clock --
#      and an Elo that small is not resolvable at any affordable n: a drawish
#      mirror match needs a few thousand games to reach +-10 Elo, which is
#      50+ hours on a real clock because both sides burn wall time. What this
#      arm CAN establish cheaply is the mechanism (clock spent per side is a
#      continuous per-game measurement, so its interval closes fast) and that
#      the Elo is not NEGATIVE. Those two together are the decision: banked
#      time is only worth having if it costs no strength.
#
#      If the clock column does not move, the flags are not reaching the search,
#      and that is a bug report rather than a null result.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
SEED=${SEED:-4242}

# name | games | budget-flag | budget | extra flags for side A
ARMS=(
  "matedist-3200|400|--sims|3200|--a-mate-distance"
  "dedup-time|600|--time|0.5|--a-dedup"
  "matedist-time|600|--time|0.5|--a-mate-distance"
)

# The clock arm does not fit the table above: it needs --tc plus two flags, and
# a different games count. Kept separate rather than generalising the table,
# because a table that can express everything stops documenting anything.
TC_ARM_NAME=${TC_ARM_NAME:-timemgmt-tc}
TC=${TC:-20+0.2}
TC_GAMES=${TC_GAMES:-400}

for spec in "${ARMS[@]}"; do
    IFS='|' read -r NAME GAMES BFLAG BUDGET AFLAG <<< "$spec"
    echo
    echo "=== $NAME: $GAMES games, $BFLAG $BUDGET, A gets $AFLAG ==="

    # An existence check in front of anything that costs GPU hours. On
    # 2026-08-12 a unit was started that had finished three days earlier,
    # because STATE.md said it was queued and nobody read runs/matches.
    if [ -f "runs/matches/$NAME/status.json" ]; then
        DONE=$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['games'])" \
               "runs/matches/$NAME/status.json")
        if [ "$DONE" -ge "$GAMES" ]; then
            echo "    complete already ($DONE games), reusing:"
            $PY -c "import json,sys;s=json.load(open(sys.argv[1]));\
print(f\"    {s['w']}W {s['d']}D {s['l']}L  elo {s['elo']:+.1f} +-{s['err']:.1f}  \
draws {s['d']/s['games']:.0%}  ({s['updated']})\")" "runs/matches/$NAME/status.json"
            continue
        fi
        echo "    partial ($DONE/$GAMES games); match.py decides whether to resume"
    fi

    # --no-sprt deliberately: these are MEASUREMENTS, not ship/no-ship gates, and
    # stopping at a boundary biases the point estimate toward the boundary. The
    # anchors are 2000 games with no SPRT for the same reason.
    #
    # Both arms carry --a-vloss-fix/--b-vloss-fix because that is the
    # deployment (CHESSGPU_VLOSS_FIX=1 since 2026-07-30), not because it is
    # match.py's default, which is off so the identity oracles keep working.
    $PY scripts/match.py \
        --name "$NAME" --games "$GAMES" "$BFLAG" "$BUDGET" --seed "$SEED" \
        --a-vloss-fix --b-vloss-fix $AFLAG \
        --a-label "${AFLAG#--a-} ON" --b-label "${AFLAG#--a-} OFF" \
        --no-sprt
done

echo
echo "=== $TC_ARM_NAME: CANCELLED, see the note below. Set RUN_TIMEMGMT_TC=1 to override ==="
if [ -f "runs/matches/$TC_ARM_NAME/status.json" ]; then
    DONE=$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['games'])" \
           "runs/matches/$TC_ARM_NAME/status.json")
    if [ "$DONE" -ge "$TC_GAMES" ]; then
        echo "    complete already ($DONE games), reusing"
    fi
fi
# CANCELLED 2026-08-14. The arm was queued to see whether instamove + early
# stopping bank clock time, on the theory that the 900+10 time forfeits were a
# time-management problem. They are not. The mechanism was found in source:
# `rust/src/tree.rs` checks the search deadline BETWEEN batches, never mid-batch,
# so a stalled GPU callback means the check never runs. The full ledger is 10
# rated forfeits plus 23 games accepted and never moved in, and this arm fixes
# none of the 33. It costs 12.2 GPU-h of exclusive box, which is most of the
# Phase 1 budget that decides whether to build a bigger net at all.
#
# The fix is plan item 0.9: a GPU mutex (`scripts/gpu_lock.py`, built) plus a
# hard engine-side per-move deadline that can interrupt a stalled batch.
#
# Set RUN_TIMEMGMT_TC=1 to run it anyway. Read docs/PLAN-2026-08-14.md first.
RUN_TC=${RUN_TIMEMGMT_TC:-0}
if [ "${RUN_TC:-0}" = "1" ]; then
    $PY scripts/match.py \
        --name "$TC_ARM_NAME" --games "$TC_GAMES" --tc "$TC" --seed "$SEED" \
        --a-vloss-fix --b-vloss-fix \
        --a-instamove --a-early-stop \
        --a-label "instamove+earlystop ON" --b-label "OFF" \
        --no-sprt
fi

# The clock column is the mechanism check and it closes far faster than the Elo.
$PY - "$TC_ARM_NAME" <<'PYEOF'
import json, sys, statistics as st
from pathlib import Path
path = Path("runs/matches") / sys.argv[1] / "games.jsonl"
if not path.exists():
    raise SystemExit(0)
rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
rows = [r for r in rows if r.get("clock")]
if not rows:
    raise SystemExit("no clock data: was this arm run without --tc?")
ratios = []
for r in rows:
    c = r["clock"]
    a = "white" if r["a_white"] else "black"
    b = "black" if r["a_white"] else "white"
    if c[b + "_spent"] > 0:
        ratios.append(c[a + "_spent"] / c[b + "_spent"])
mean = st.mean(ratios)
half = 1.96 * st.stdev(ratios) / len(ratios) ** 0.5 if len(ratios) > 1 else float("nan")
print(f"\n    CLOCK: A spent {mean:.3f} +-{half:.3f} of B's time over {len(ratios)} games")
print("    Below 1.0 means the flags are banking time. If this sits at 1.000,")
print("    they are not reaching the search and the Elo below means nothing.")
PYEOF

echo
echo "=== all arms done ==="
