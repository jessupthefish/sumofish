#!/usr/bin/env bash
# Wait for tonight's hand-started jobs, land the pending code, start the lab.
#
# Written to run once, unattended, while nobody is awake. Every step that could
# leave the tree in a worse state than it found it is gated on the test suite,
# and any failure stops the script rather than continuing into the next step.
#
#   nohup scripts/handoff.sh > runs/lab/handoff.log 2>&1 &
#
# What it does, in order:
#   1. waits for the hand-started match queue and the causal ablation to finish
#   2. merges the select-loop branch (the hoisted PUCT loop and the curve capture)
#   3. runs BOTH correctness gates and aborts the merge if either fails
#   4. starts chess-gpu-lab, which picks up the finished matches by name and
#      goes on to the scale decision
#
# The one thing it deliberately does NOT do is promote anything. The lab stages
# a candidate; swapping the live bot is a morning decision.
set -uo pipefail

ROOT=/home/nomad/dev/active/chess-gpu
DEV=/home/nomad/dev/active/chess-gpu-dev
PY=$ROOT/.venv/bin/python
cd "$ROOT" || exit 1

say() { printf '[handoff %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { say "STOPPING: $*"; exit 1; }

# --- 1. wait ---------------------------------------------------------------
# Same rule as the lab's own wait: require "python" in the command line, so an
# editor or a `tail` on a log does not look like a running job. The live bot is
# not counted; it plays regardless and every match here is fixed-simulation, so
# contention costs wall clock and cannot bias a result.
busy() {
  ps -eo args= 2>/dev/null | grep -v grep \
    | grep -E 'python.*(train\.py|scripts/match\.py)' | grep -v handoff | head -1
}

say "waiting for the hand-started queue to drain"
WAITED=0
while B=$(busy); [ -n "$B" ]; do
  if [ $((WAITED % 1800)) -eq 0 ]; then say "still busy: ${B:0:90}"; fi
  sleep 60
  WAITED=$((WAITED + 60))
  # 10 hours. Longer than the queue should ever take, short enough that a wedged
  # job does not hold the merge until morning with no explanation.
  [ $WAITED -gt 36000 ] && fail "gave up waiting after 10h"
done
say "card is free after ${WAITED}s"

# --- 2. merge --------------------------------------------------------------
BEFORE=$(git rev-parse HEAD)
say "merging select-loop into $(git branch --show-current), from $BEFORE"
git merge --no-edit select-loop || fail "merge conflict; nothing else attempted"
say "merged to $(git rev-parse --short HEAD)"

# --- 3. gate ---------------------------------------------------------------
# Both suites, because the merge touches the tokenizer path and the search. If
# either fails the merge is rolled back: an unattended script must not leave a
# broken engine in the tree the live bot imports from.
for suite in tests/verify_search.py tests/verify_data.py; do
  say "running $suite"
  if ! $PY "$suite" > "runs/lab/handoff-$(basename "$suite" .py).log" 2>&1; then
    say "$suite FAILED, rolling the merge back to $BEFORE"
    git reset --hard "$BEFORE"
    fail "$suite failed; see runs/lab/handoff-$(basename "$suite" .py).log"
  fi
  say "$suite passed"
done

# A speed change must not move a fixed-simulation result. The selection rewrite
# is proved identical over 4,000 random nodes by verify_search, so this is a
# throughput check rather than a strength check.
say "post-merge throughput:"
$PY scripts/bench_search.py --preset 9M --baseline 9M --seconds 4 2>&1 | sed 's/^/    /'

# --- 4. hand over to the lab ----------------------------------------------
say "starting chess-gpu-lab"
systemctl --user start chess-gpu-lab || fail "could not start the lab"
sleep 10
systemctl --user is-active chess-gpu-lab | sed 's/^/    /'
$PY scripts/lab.py status | sed 's/^/    /'
say "done. sumofish-lab to watch it."
