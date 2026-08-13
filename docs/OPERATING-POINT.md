# The operating point

> Written 2026-08-13, because STATE.md's "Open, smaller" had been asking for it
> since 2026-07-30 with the note "several arguments have quietly assumed
> different ones." They had. See the headline below.

Every number in this project is an answer to "how strong is the engine", and
that question has no answer without saying *at what budget, against whom, on
what clock*. This file is the single place those are recorded. When a claim
elsewhere assumes something different, this file is what it is wrong against.

Regenerate the measured half with `scripts/operating_point.py`.

---

## THE HEADLINE: the lab measures 9.2 doublings below where the engine plays

| | simulations per move |
|---|---|
| lab default (`match.py --sims`, every rung, anchor, ruler and tuning arm) | **400** |
| top rung of the parity ladder | 3,200 |
| **deployed, median over 336 logged moves** | **231,273** |

That is **578x, or 9.2 doublings**, between where strength is measured and where
it is played. The parity ladder reaches 3,200, so the deployed operating point
sits **6.2 doublings above the highest rung this project has ever measured.**

This would be unremarkable if search behaved the same at every budget. The
parity ladder's central finding is that it does not: the exchange rate falls
from **1.15 node-doublings per sims-doubling at 200 sims to 0.49 at 1600**, a
decay of 0.66 +-0.18 across four doublings. Extrapolating a measured-to-be-
non-constant quantity 6.2 doublings past its last data point is the same species
of assumption as the Stockfish ruler that was withdrawn on 2026-08-11.

**This does not invalidate anything, and it is not a reason to stop using 400
sims.** Cheap arms are what make 2,000-game intervals affordable, and the one
time the project checked transfer directly it came out the *good* way: the v6
constants gained **+35.6 +-26.0 MORE at 3,200 sims than at 400** (z = 2.68). The
rule this file asks for is narrower:

**Any result intended to justify a deployment decision needs a second arm at a
higher budget, and the write-up must state the budget it was measured at.** The
`transfer-400` / `transfer-3200` pair is the pattern; it costs roughly one extra
arm and it is the only evidence this project has that anything transfers upward.

---

## The clock

15+10 rapid, incoming and outgoing, and nothing else. Bullet and blitz starve
the search; classical roughly doubles the cost for little more than rapid buys.

The budget rule is in `sumofish/engines/search_engine.py`:

    budget = remaining/30 + 0.7 * increment
    budget = clamp(budget, 0.05s, min(remaining/3, remaining - 0.02s))

Spending a fixed fraction of what remains is self-correcting: the budget shrinks
with the clock, so the engine cannot flag by arithmetic. The increment is nearly
free to spend because it comes back every move.

At 15+10 the opening move gets `900/30 + 7 = 37s`, and the budget decays from
there as the clock does.

**`CHESSGPU_SIMS=100000000` is deliberately not binding.** The clock stops the
search, never a counter. A search stopped by an arbitrary simulation cap is
throwing away the time control it was just handed.

## What that actually buys, measured

**Snapshot, 2026-08-13 14:20.** 336 completed moves across 6 games, from
`logs/engine.jsonl{,.1}`, with the live bot at `concurrency: 2` (so these are
numbers under self-contention, which is the deployed condition and not a
clean-room figure). The bot is playing continuously and the log rotates, so
these move; re-run the script rather than trusting the table's last digit. The
9.2 doublings is the durable part and it will not move without a change to the
time control.

| | p10 | median | p90 |
|---|---|---|---|
| budget, s | 13 | 19 | 33 |
| elapsed, s | 13 | 19 | 33 |
| nodes | 87,542 | 148,705 | 249,631 |
| simulations | 107,058 | **231,273** | 507,244 |
| nodes/s | 6,069 | 7,761 | 8,672 |

Three things worth keeping:

- **`elapsed/budget` is 1.00 at the median and no move took under a second.**
  The engine spends its entire allowance every move. There is no early stopping
  and no instamove, so every one of those 19 seconds is real search, including
  in positions with one legal reply.
- **simulations/node is 1.43.** Most simulations expand a new node rather than
  revisiting one, which is what a shallow-and-wide MCTS looks like.
- **nodes/s under 2-game contention is ~7,760**, against the ~7,763 measured on
  an idle box at batch 64 in `runs/lab/profile-2026-07-31.json`. Two concurrent
  games are close to free on this GPU, which is the measurement that makes
  cross-game batching worth building rather than just plausible.

## The opponents

- **Rated and casual both accepted; outgoing challenges are rated.** An earlier
  rule of "casual only until the engine genuinely tries to win" was a
  conditional and the condition has been met since 2026-07-31 (real search, real
  evaluation). Casual games never move the rating, so a casual-only bot's
  matchmaking has no signal and drifts into 2500-3500 engines.
- **Matchmaking window: `opponent_min_rating: 2200`, `opponent_max_rating:
  3000`.** Asymmetric on purpose. The bot pool is bottom-heavy relative to us,
  so a symmetric window cannot stop feeding us weaker opponents at any width.
  Measured against a 333-bot online list on 2026-07-31: the old window offered
  150 candidates, 94 of them below us; this one offers 114, only 31 below.
- **These bounds are static and do not track the rating.** Revisit if rapid
  moves more than ~150 points from the 2346 they were set at. It is 2567 as of
  2026-08-13, so this is 71 points from being due.
- `only_bot: false`, so humans who challenge are accepted at 15+10.
- `concurrency: 2`.

## The lab, for contrast

| | value |
|---|---|
| budget | 400 simulations, fixed (not clock) |
| batch | 64 |
| core | rust |
| `c_puct_init` / `fpu` | 0.875 / -0.05 (v6) |
| `vloss_fix` | on, both arms, via `lab.py`'s `match_argv` |
| `dedup` / `compile` / `mate_distance` | off |
| opponent | Stockfish at pinned NODES, or the previous build |
| book | `data/eco_openings.pgn`, plies 6-12, paired colours |
| adjudication | at 0.97 wp for 10 plies, confirmed by a 200k-node arbiter |

The flags agree with deployment. **The budget does not, by 9.2 doublings**, and
the clock/simulation distinction means the two are not even the same *kind* of
limit: deployed play is stopped by time, every lab arm is stopped by a counter.
A change that makes the search cheaper per node is worth real Elo in deployment
and exactly zero in a fixed-simulation match. `mate_distance` shrinking the tree
24% is precisely such a change.
