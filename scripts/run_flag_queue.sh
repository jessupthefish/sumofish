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
# NOT here, because the harness cannot express it: instamove and early stopping.
# match.py's --time is seconds per MOVE and Player.move never calls
# search_engine.choose, so think_time is not in the loop at all. See the
# game-clock item in STATE.md.
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
echo "=== all arms done ==="
