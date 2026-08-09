#!/usr/bin/env bash
# The first trustworthy external Elo anchor: SumoFish vs Stockfish, playing
# full games (not just adjudicating them). See the "Stockfish as a playing
# side" comment in `Player.__init__` (scripts/match.py) for the mechanism and
# the Arbiter docstring in the same file for why "fixed NODES, full strength"
# is the only axis a reference engine is allowed to be weakened on here.
#
# Two node budgets, run sequentially rather than in parallel.
#
# RECALIBRATED 2026-08-09. The old 40/100-node points are gone. They were set
# 07-30 against a weaker net with the virtual-loss defect still live, and both
# have since become walkovers: today's net scored W3 D1 L0 at 40 nodes. An
# anchor point that one side wins outright has a huge interval and says nothing
# about the SIZE of the gap, which is the whole reason the anchor exists.
#
# Re-measured against the v5 net (9M-sv-long, step 900k) at --sims 400 with
# --a-vloss-fix, 16-24 games per point:
#
#   100 nodes   68.8%      400 nodes   68.8%
#   700 nodes   50.0%   <- 8W 8D 8L over 24 games, elo -0.0 +-125.6. Parity.
#  1600 nodes   28.1%      (an earlier n=8 read of 700n said 31%; it converged
#                           to dead even by n=24, so do not calibrate off n=8)
#
#   700 nodes -- the parity point, and therefore the one that measures the size
#                of the gap rather than its sign. 33% draws here against 85% in
#                the mirror match, which is the entire reason to prefer an
#                external opponent.
#  1600 nodes -- a second rung on the far side of even, so the anchor is not a
#                single possibly-lucky point. 28.1% is a real gap and not a
#                wipeout.
#
# 2000 games per budget, not 600. Measured cost is ~3.2s/game, so 2000 games is
# ~1.8h per rung and ~3.6h for both -- against the 9.6h the mirror match spent
# to earn +-26.4 Elo. The interval scales as 1/sqrt(n): +-125.6 at n=24 becomes
# roughly +-14 at n=2000. That is the first interval this project has ever had
# that is tight enough to price a net change, and it costs a third of what the
# blind test cost.
#
# --no-sprt, and that is deliberate. Fixed 2026-08-09: the sequential test
# exists to answer "is A better than B" as cheaply as possible, and it stops the
# moment the LLR crosses its bound -- which on a clear gap is about eight pairs
# (measured: LLR grows ~0.38/pair on sims-1600-vs-800). An ANCHOR does not want
# that answer. It wants a tight interval around a point estimate, and an early
# stop delivers the widest interval the test will accept. So both budgets play
# their full 600 games and `status.json` carries a real Elo with a 95% interval.
#
# --a-vloss-fix, also fixed 2026-08-09 and the more serious of the two. This
# script was written 07-31, before `match_argv` started passing the flag, and
# `--a-vloss-fix` is `store_true` defaulting to OFF. Without it the anchor would
# have measured an engine that has not played a rated game since 2026-07-30 --
# the exact error that invalidated the entire exchange ladder and cost a rerun.
# B is Stockfish, so B's engine flags are ignored and only A takes one.
#
# The "GPU already carries two training jobs" note above is stale as of
# 2026-08-09: no training is running, only the live bot.
set -euo pipefail
cd /home/nomad/dev/active/sumofish

.venv/bin/python scripts/match.py \
  --value runs/value.pt --policy runs/policy.pt --sims 400 --core rust \
  --a-vloss-fix \
  --a-label SumoFish --b-stockfish-nodes 700 --b-label "Stockfish@700n" \
  --games 2000 --no-sprt --name stockfish-anchor-700nodes

.venv/bin/python scripts/match.py \
  --value runs/value.pt --policy runs/policy.pt --sims 400 --core rust \
  --a-vloss-fix \
  --a-label SumoFish --b-stockfish-nodes 1600 --b-label "Stockfish@1600n" \
  --games 2000 --no-sprt --name stockfish-anchor-1600nodes
