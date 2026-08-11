#!/usr/bin/env bash
# Re-earn the Stockfish node ruler at high n. CPU ONLY, no GPU.
#
# WHY: every ABSOLUTE in the ladder is a rung plus a walk along this chain, and
# as of 2026-08-11 the chain contributes +-32.1 of scale_D's +-33.5 while the
# rungs contribute +-9.4. Taking these from 200 to 2400 games each drops the
# ruler term to ~+-9.4 and scale_D to ~+-13.6. It is the cheapest error
# reduction available to this project and needs none of the GPU.
#
# Both sides are Stockfish at fixed NODES, so the result is deterministic under
# CPU contention: contention costs wall time, not Elo.
#
# Seed 99 throughout, matching the runs these supersede, so the openings are the
# same and a paired comparison against the old ones stays possible.
set -euo pipefail
cd "$(dirname "$0")/.."

GAMES=${GAMES:-2400}
SEED=${SEED:-99}
PY=.venv/bin/python

# lo hi, in the order sim_ladder.py chains them
RUNGS=("350 700" "700 845" "700 1400" "1400 2800" "2800 5600" "5600 11200")

for pair in "${RUNGS[@]}"; do
    read -r LO HI <<< "$pair"
    NAME="ruler-${LO}-vs-${HI}"
    echo "=== $NAME : $GAMES games, seed $SEED ==="
    $PY scripts/match.py \
        --a-stockfish-nodes "$LO"  --a-label "SF@${LO}" \
        --b-stockfish-nodes "$HI"  --b-label "SF@${HI}" \
        --games "$GAMES" --seed "$SEED" --no-sprt --name "$NAME" \
        2>&1 | tail -2
done

echo
echo "=== regenerating the ladder off the new ruler ==="
$PY scripts/sim_ladder.py --report
