# Retraction: the exchange-rate ladder

**Date:** 2026-07-29
**Affects:** every claim of the form "N Elo per doubling of search"

## What happened

All four rungs of the exchange-rate ladder were produced by **replaying existing
match logs**, not by playing the matches. `scripts/match.py` resumed on the game
index alone, so a job with different code, a different checkpoint and a different
budget landed on an existing `runs/matches/<name>/`, skipped every game as
"already played", and reported the old numbers as its own. It then rewrote
`config.json` with the new spec over the old games, so each directory actively
asserted a provenance it never had.

## The evidence

The games were real. The per-game timings inside `games.jsonl` are organic. What
is impossible is the elapsed time the lab credited to the job that supposedly
produced them:

| rung | game time in its own log | credited to the job | factor |
|---|---|---|---|
| 400 -> 200 | 2,452s (0.7h) | **5s** | 490x |
| 800 -> 400 | 3,913s (1.1h) | **5s** | 783x |
| 1600 -> 800 | 9,508s (2.6h) | **5s** | 1902x |
| 3200 -> 1600 | 20,733s (5.8h) | **1,070s** | 19x |

Three of the four also carry a code fingerprint on **0 of 300 games**, so they
cannot be attributed to any particular engine even in principle.

## What is withdrawn

* the ladder itself: +2.3 / +218.2 / +283.8 / +237.2 Elo per doubling
* "peaks near 1200 visits"
* "~+50 Elo per doubling" at the deployed budget, extrapolated from it
* `scale_bar: 265.0` and "the 136M must clear +265 Elo to break even"
* **"two independent estimates agreeing is the strongest evidence in the
  project"** -- the +74 from the rating jump is confounded (v1 bundles a 2.5x
  search speedup with the 1+0 -> 15+10 switch under one rating delta), and the
  +50 was chosen so the extrapolation would land near it. One number was fitted
  to the other; they were never independent.
* "3,400 simulations/s deployed" -- it is ~1,600, because `concurrency: 2` halves it

Withdrawn means **deleted at the point of use**, not footnoted. A reader sees
`README.md`; nobody reads a retraction.

## What survives

* **The lichess ratings** (2379 rapid, RD 73, n=34). Measured by someone else on
  an instrument this project cannot forge. The only strength evidence in the repo.
* The cProfile facts: `python-chess` ~44% of search wall clock, `_puct` 29%, the
  network 9%. Single-process, cheap to re-take, never touched the replay path.
* The `games.jsonl` files themselves. The games happened; the attribution did not.

## What prevents recurrence

* `scripts/verify_replays.py` -- checks `sum(game.seconds) <= job.seconds`, an
  inequality that cannot be violated legitimately. No threshold, no duration
  model, no false positives. Currently reports **0 of 8 matches trusted**.
* `scripts/match.py` now hashes the full spec and **refuses to resume** across a
  change, exiting non-zero. See `docs/induced-failures.md` for the three refusals
  induced to prove it works, and the two bugs that exercise found in it.
* Adjudication now requires a third party to agree. The engines under test share
  a value net and were deciding 22-34% of every match on their own word.

## Re-earning it

~10 GPU-hours once the above is merged, with a distinct seed per rung (every match
ever run used seed 7 and the same 150 openings from a pool of 1,761). It buys the
**ordinal only**: at 300 games a rung the standard error is +-35-40 Elo, so
flat-versus-attenuating is not resolvable at that budget. Say so when reporting it.
