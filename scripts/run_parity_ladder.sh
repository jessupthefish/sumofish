#!/usr/bin/env bash
# The ladder, re-aimed so every rung is played NEAR PARITY.
#
# WHY: the old rungs are real measurements at budgets that are not close to
# parity (-104, -225, -293 Elo), and they were made absolute by walking a
# Stockfish-vs-Stockfish ruler. That ruler was measured 2026-08-11 NOT to
# transfer: SF@700n -> SF@1600n is 280.8 +-16.4 between two Stockfish and
# 192.8 +-18.0 through SumoFish, z = 7.1. So the walk is the problem, and the
# fix is to make the walk short enough not to matter.
#
# A rung played at parity answers "SumoFish at S simulations is worth Stockfish
# at N nodes" with a correction of a few tens of Elo instead of a few hundred,
# and the two routes already AGREE at that range (+30.5 vs +32.3 over 0.27
# doublings). The exchange rate then comes out as node-doublings per
# sims-doubling: dimensionless, and no cross-population Elo conversion is ever
# made at long range.
#
# AIM POINTS come from the existing rungs and the SumoFish-perceived slope
# (161.7 Elo per doubling of Stockfish nodes, from the two anchors):
#
#   800 sims:  -104.4 vs SF@1970  -> parity ~1259 nodes
#   1600 sims: -225.0 vs SF@4600  -> parity ~1754
#   3200 sims: -293.0 vs SF@10700 -> parity ~3046
#
# Aiming is allowed to use a rough number; quoting one is not. If a rung lands
# outside about +-60 Elo, re-aim it and run it again rather than correcting it
# on paper.
#
# 200 and 400 sims are NOT re-run: they are already at -7.4 and -24.8, which is
# parity for this purpose.
#
# The last arm is not a rung. It plays 800 sims against a SECOND budget one
# doubling above its parity point, which measures the transfer factor at a
# different place on the scale than the 700/1600 anchors did. If 0.69 is a
# constant, that is worth knowing; if it is not, that is worth knowing sooner.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
SEED=${SEED:-4242}          # same openings as the rungs these replace

# The shipped search constants, READ FROM THE LIBRARY. match.py carries its own
# hardcoded defaults for these, and a constant kept in sync by hand eventually
# is not: the 08-09 tuning sweep measured every arm against a baseline the
# engine had stopped using two days earlier for exactly this reason.
CI=$($PY -c 'import inspect;from sumofish.mcts import MCTS;print(inspect.signature(MCTS.__init__).parameters["c_puct_init"].default)')
FPU=$($PY -c 'import inspect;from sumofish.mcts import MCTS;print(inspect.signature(MCTS.__init__).parameters["fpu"].default)')
echo "shipped: c_puct_init=$CI fpu=$FPU"

# sims nodes games name
ARMS=(
    "800  1259 800 parity-sims800-vs-sf1259"
    "1600 1754 800 parity-sims1600-vs-sf1754"
    "3200 3046 700 parity-sims3200-vs-sf3046"
    "800  2518 800 transfer-sims800-vs-sf2518"
)

for arm in "${ARMS[@]}"; do
    read -r SIMS NODES GAMES NAME <<< "$arm"
    echo "=== $NAME : $SIMS sims vs Stockfish@${NODES}n, $GAMES games, seed $SEED ==="
    # match.py resumes an interrupted run of the same name and spec, so a
    # reboot costs the game in flight and not the arm. It refuses to resume
    # across a spec change, which is what makes that safe.
    $PY scripts/match.py \
        --value runs/value.pt --policy runs/policy.pt \
        --sims "$SIMS" --core rust --a-vloss-fix \
        --a-cpuct-init "$CI" --a-fpu "$FPU" \
        --a-label "sumofish@${SIMS}sims" \
        --b-stockfish-nodes "$NODES" --b-label "SF@${NODES}n" \
        --games "$GAMES" --seed "$SEED" --no-sprt --name "$NAME" \
        2>&1 | tail -2
done

echo
echo "=== the parity table ==="
$PY scripts/parity_ladder.py --report
