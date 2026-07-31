#!/usr/bin/env bash
# The first trustworthy external Elo anchor: SumoFish vs Stockfish, playing
# full games (not just adjudicating them). See the "Stockfish as a playing
# side" comment in `Player.__init__` (scripts/match.py) for the mechanism and
# the Arbiter docstring in the same file for why "fixed NODES, full strength"
# is the only axis a reference engine is allowed to be weakened on here.
#
# Two node budgets, run sequentially rather than in parallel:
#
#   40 nodes  -- calibrated 2026-07-30 against the CURRENT checkpoint
#                (runs/value.pt / runs/policy.pt) at --sims 400, the
#                project's established convention (see recent
#                runs/matches/*/config.json). An 8-game trial scored
#                SumoFish 43.8% (elo -44 +-166, wide at n=8) -- close to
#                parity, so this is the informative point: neither side
#                wipes the other out, which is exactly what makes a match
#                worth playing out to a real interval.
#   100 nodes -- same calibration, SumoFish scored 31.2% (elo -137 +-309 at
#                n=8): a real gap but not a wipeout either. Gives a second
#                rung so the anchor isn't a single, possibly-lucky data
#                point -- if 40 and 100 nodes both land SumoFish on the same
#                side of even, that is a much stronger claim than either
#                alone.
#
# 1 and 20 nodes favoured SumoFish outright (87.5% and 62.5% at n=4/n=4);
# 500 nodes was a near-wipeout the other way (12.5%). Both were rejected as
# anchor points for the same reason PHILOSOPHY rejects a 0%/100% match: an
# extreme score has a huge confidence interval and answers "which side is
# stronger" without saying anything about the SIZE of the gap.
#
# Sequential, not parallel: the GPU already carries two training jobs
# (~12.6/16.3GB, 100% util) and the live bot. SumoFish's own inference is the
# only GPU consumer in this match (Stockfish is CPU-only); running the two
# node-budget matches one after another keeps that to one match's worth of
# GPU inference at a time, matching the headroom this job was scoped against.
#
# Same SPRT machinery and defaults as tonight's other matches
# (--elo0 0 --elo1 20 --alpha 0.05 --beta 0.05, match.py's own established
# defaults): it will stop early if the evidence is decisive, and either way
# `status.json` always carries a real Elo estimate with a 95% interval, never
# a bare win/loss tally, per PHILOSOPHY's measurement discipline.
set -euo pipefail
cd /home/nomad/dev/active/chess-gpu

.venv/bin/python scripts/match.py \
  --value runs/value.pt --policy runs/policy.pt --sims 400 --core rust \
  --a-label SumoFish --b-stockfish-nodes 40 --b-label "Stockfish@40n" \
  --games 600 --name stockfish-anchor-40nodes

.venv/bin/python scripts/match.py \
  --value runs/value.pt --policy runs/policy.pt --sims 400 --core rust \
  --a-label SumoFish --b-stockfish-nodes 100 --b-label "Stockfish@100n" \
  --games 600 --name stockfish-anchor-100nodes
