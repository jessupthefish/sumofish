#!/usr/bin/env bash
# Does the v6 search-constant gain SURVIVE A LARGER SEARCH BUDGET?
#
# c_puct_init=0.875 with fpu=-0.05 scored +107.9 +-13 against Stockfish@700n
# where the previous constants scored ~+43. That is +63.5 at 3.6 sigma, and
# strongly superadditive: the marginals were only +16.0 and +13.3. It shipped as
# v6 on 2026-08-09 and the bot has played with it ever since.
#
# It was measured entirely at --sims 400. The bot plays 15+10 at ~60,000 nodes a
# move. `MCTS.c_puct_at`'s own docstring is explicit that this does not transfer
# for free: "the balance between trying something new and pursuing what already
# looks good shifts with N, and the c that balances it shifts too". Tuning
# exploration at 1/150th of the deployment budget and shipping it is precisely
# what that docstring warns about, so it gets tested at a bigger budget.
#
#   transfer-400   600 games at  400 sims -- the CONTROL. Must reproduce ~+63.
#                  If it does not, the disagreement is head-to-head vs
#                  vs-Stockfish and not budget, and the 3200 arm means nothing.
#   transfer-3200  400 games at 3200 sims -- 8x the budget. This is the answer.
#
# Deliberately head to head despite the mirror-match lesson: that collapse needed
# two configurations that play the SAME MOVES. These two demonstrably do not,
# having landed 65 Elo apart. Watch the draw rate anyway; if it comes back near
# 85% the arm is blind and the number should be thrown out, not interpreted.
#
# Side A is read from the library, not hardcoded, because it is whatever the bot
# is actually running. Side B is a fixed historical configuration and is written
# out. The unit that used to hold this inline called B "shipped", which stopped
# being true when v6 was cut.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
SEED=${SEED:-4242}
CI=$($PY -c 'import inspect;from sumofish.mcts import MCTS;print(inspect.signature(MCTS.__init__).parameters["c_puct_init"].default)')
FPU=$($PY -c 'import inspect;from sumofish.mcts import MCTS;print(inspect.signature(MCTS.__init__).parameters["fpu"].default)')
echo "[transfer] A is the LIVE engine: c_puct_init=$CI fpu=$FPU"

for arm in "transfer-400 400 600" "transfer-3200 3200 400"; do
    read -r NAME SIMS GAMES <<< "$arm"
    echo "=== $NAME: $SIMS sims, $GAMES games, live config vs the pre-v6 one ==="
    # A COMPLETE arm is reused. match.py refuses to RESUME across a code change
    # and it is right to: its fingerprint covers the git SHA plus a hash of
    # sumofish/*.py, so it cannot tell a cosmetic commit from an engine change,
    # and half a match played by two engines is how the exchange ladder became
    # four replays. A complete arm is a different case, but only after somebody
    # reads the diff. transfer-400 was played on 2026-08-09 under a1f8a8b and
    # the only engine changes since are the c_puct_init/fpu DEFAULTS (which this
    # match overrides explicitly on both sides) and a new batching.py that
    # nothing in the search path imports. Checked 2026-08-12. Do not extend this
    # exemption to a new arm without doing the same reading.
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
    fi
    $PY scripts/match.py \
        --value runs/value.pt --policy runs/policy.pt --sims "$SIMS" --core rust \
        --a-vloss-fix --a-cpuct-init "$CI"  --a-fpu "$FPU" --a-label "v6-live" \
        --b-vloss-fix --b-cpuct-init 1.25   --b-fpu -0.2   --b-label "pre-v6" \
        --games "$GAMES" --no-sprt --seed "$SEED" --name "$NAME" 2>&1 | tail -2
done
