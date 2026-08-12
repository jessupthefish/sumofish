# What SumoFish is for

Read this before proposing anything. It is the objective function: not *how*
the engine is built, but *why*, and when the two conflict, this wins.

**Rewritten 2026-07-29. The objective changed.** Earlier versions of this file
ranked "fun to play against" above strength and listed Elo as an anti-goal.
That is no longer true and you should not act on any summary of this file
written before this date.

**Amended 2026-08-06.** "Fun to play against" is now CANCELLED, not deferred.
There is one goal and it is strength. See "Cancelled" below for what that
removes and the one narrow constraint that survives.

## The goal

**Build the strongest chess engine one person can build on one consumer GPU,
and understand every mechanism that makes it strong.**

That is one goal, not two. Strength is the scoreboard and learning to train is
the reason for keeping score. A number that goes up for a reason nobody
understands teaches nothing and cannot be repeated; an elegant explanation of a
model that loses is a story about nothing. Both halves have to hold.

Cost and time are not constraints. A 40-hour training run is fine. A week of
GPU time to answer one question is fine. Being wrong for two days and knowing
it is fine. The only real budget is 16 GB of VRAM, one RTX 5070 Ti, and
attention.

### What "strongest" means concretely

Ordered, most trustworthy first:

1. **Elo against fixed external opposition.** Stockfish pinned at known skill
   levels and depths, played on a clock, from book openings. This is the only
   number that survives a rebuild, because everything else here is relative.
2. **Lichess rating at the deployed time control**, as a slow, noisy,
   real-world check on (1).
3. **Elo against the previous SumoFish**, pair-scored with intervals. Fast,
   sensitive, and the daily driver, but it floats.

Everything else (puzzle accuracy, held-out loss, bits-per-move, nodes per
second) is an *instrument*, not the target. Instruments are for deciding what
to try next and for catching a broken run early. They are never the reason to
ship. A change that improves held-out loss and loses the match did not work.

### What "understand every layer" means now

It survives the rewrite, and it is the reason this is worth doing at all, but
its role has changed. It is no longer a veto over strength. It is a *method*.

- Do not hand over a working black box. Prefer the version that can be read,
  modified and broken. Explain the mechanism, not just the result: the actual
  tensor shapes, the actual bytes, the actual profile.
- Hand-written CUDA, custom kernels and from-scratch implementations are goals,
  not premature optimisation, but they get built **when the profile says they
  are the bottleneck**, not before. This project already burned most of a
  session planning kernels for a network that was 5% of its own search. Chasing
  strength is what makes that mistake visible; that is the point of a
  scoreboard.
- Chores stay chores. Deploy plumbing, download scripts, systemd units, config
  files: automate them and move on. The domain (chess, ML, search, GPU) gets
  built by hand.
- When a library would do in one line what forty readable lines would teach,
  take the forty lines *the first time*, then keep whichever one actually wins
  on the clock.

The tension between "build it yourself" and "make it strongest" is real and it
resolves the same way every time: **build it yourself, then measure it, then
keep the version that wins.** Do not defend a hand-written component that
loses. Understanding why it lost is the learning.

## What actually makes an engine strong

The engine is the product of four multipliers and they do not trade off against
one another. Neglecting one caps the others.

1. **Evaluation quality**: how good the network's judgement of a position is.
   Bought with parameters, data, training compute, and target design.
2. **Search**: how much lookahead that judgement gets multiplied by.
   **THE EXCHANGE RATE IS WITHDRAWN AGAIN, on the evening of 2026-08-11, the
   same day this paragraph announced it.** `scale_D` was 199 +-33, then 185.2
   +-12.2 on a better ruler, and it is now no number at all. Five absolute
   rungs against Stockfish at pinned node budgets are real measurements, but
   each is made absolute by walking a Stockfish-vs-Stockfish ruler, and that
   ruler does NOT transfer: the span SF@700n -> SF@1600n is 280.8 +-16.4
   measured Stockfish against Stockfish and 192.8 +-18.0 measured through
   SumoFish, z = 7.1. The walk contributes more of `scale_D` than the rungs do,
   so the bias is several times the interval. `scripts/ruler_transfer.py` runs
   the check; LAB-NOTES 2026-08-11 has the full argument.

   **Do not quote 185.2 x 0.69 either.** The factor is measured at one place on
   the scale, and assuming it is constant is the same assumption that just
   failed. What restores a number: a ladder whose rungs are each played near
   parity and quoted as an equivalent Stockfish NODE BUDGET rather than an Elo,
   so the cross-population conversion is never made at long range.

   **What stands, and it is the strongest external claim this project has:**
   SumoFish@400 sims is **+30.5 +-12 Elo on Stockfish@700 nodes** and
   **-162.4 +-13.4 on Stockfish@1600 nodes**, 2000 games each, no chain.

   The ban this paragraph used to carry is LIFTED, and the history is worth
   keeping because it is why the number is trusted now. The old ~+50 Elo per
   doubling came from "two independent estimates agreeing"; both were retracted
   on 2026-07-29 when all four rungs of that ladder turned out to be *replayed*
   match logs rather than games the jobs credited with them had played, and the
   second estimate had been fitted so the first would agree with it. The
   replacement shares no machinery with it: every rung is an independent
   measurement against an external opponent rather than a link in a chain of
   relative comparisons.

   **The interval was never the problem, and chasing it is what hid the
   problem.** +-32 of the old +-33 was the ruler, so re-earning the ruler at
   2400 games a rung looked like the cheapest possible improvement and was
   duly done. It worked: +-12.2. It also could not have detected the bias,
   because an interval describes repeatability and says nothing about an error
   that points the same way every time. Before narrowing an interval, ask what
   would falsify the point estimate.
3. **Speed**: search per second, which converts directly into (2) on a clock.
   Currently CPU-bound in `python-chess`, not GPU-bound. Speed *is* strength
   here in a way it never was for a searchless engine.
4. **The measurement loop**: how fast a wrong idea can be killed. This is the
   multiplier on all learning and it is the one most often skipped. `match.py`
   and `lab.py` exist for this reason and they are load-bearing.

A proposal should say which of the four it moves and by how much, in Elo,
with the evidence that estimate rests on.

## Measurement discipline, which is not optional

This is the part of the file that most changes what you are allowed to claim.

- **A number without an interval is not a result.** Puzzle accuracy has a
  ±1.5% sigma at n=1000. A 24-game match is worth about ±200 Elo. Most
  historical claims in this repo cannot see a 20-Elo change and that is a fact
  about the instrument, not about the change.
- **Pair-score matches** from book openings; colour-swapping cancels the
  opening's bias. **What it is WORTH depends on the opponent, and the ~2.4x
  once quoted here does not apply to any match this project now runs.** That
  figure came from a MIRROR match (`elo.py`, r = 0.44): pairing pays in
  proportion to how correlated the two arms' results are on the same opening,
  and two builds of one lineage are highly correlated. Against a FIXED external
  opponent that correlation nearly vanishes: measured `pairing_efficiency` is
  0.89 to 1.01 across every vs-Stockfish match now carrying a published number,
  i.e. **1.0x to 1.1x, not 2.4x**. Pair anyway, because it cannot hurt and it
  removes a real bias, but never budget games on the old multiplier. See
  LAB-NOTES 2026-08-11.
- **A mirror match is structurally blind to anything both sides share.** It can
  answer "did this change help". It cannot answer "how does SumoFish play" or
  "what is its draw rate". Those come from real games only.
- **Adjudication by the engine under test is a bug**, because a net that is
  more confident rather than more correct banks more wins. Stockfish is the
  third-party adjudicator and the absolute anchor.
- **Select checkpoints on held-out loss, not on a noisy eval.** Best-of-twenty
  on a ±1.5% metric is biased upward by about two sigma, and it already
  promoted the marginally worse of two checkpoints once.
- Report `unique/s`, never raw nps, for anything touching batched search. This
  rule was already here on 2026-07-29 and was violated the same day: the
  `dedup`+`compile` configuration was adopted on the strength of 1.75x the
  simulations, and cost **-168 Elo**, because at a fixed clock it got 3,464
  unique evaluations where plain got 4,160. More claimed search, 17% less
  knowledge. The gap between `evaluations` and `unique_evaluations` is the part
  that is not search, and both engines have always printed both.
- **An identity proof at a fixed simulation count says nothing about strength at
  a fixed clock.** They are different experiments and only the second one is the
  one that gets played. A change that provably does not alter the tree still
  alters *how much tree you get per second*, and that is a strength change. So:
  byte-identity licenses shipping a pure speedup at matched sims; it never
  licenses a flag that shifts the sims/second ratio. That needs games.
- **A wall-clock match requires an idle machine.** Contention biases a
  time-budgeted experiment and nothing else, so it is the one experiment where
  "the bot was also running" invalidates the result. Stop the bot and the lab,
  and record in the match config that they were down.
- **A number without provenance is not a result either.** On 2026-07-29 it was
  found that all four rungs of the exchange-rate ladder were produced by
  *replaying* existing match logs: `match.py` keys resume on the game index
  alone, so a job with different code and config lands on an existing directory
  and reports it as its own work, and `match.py:458` then rewrites `config.json`
  over it. The replay is invisible in `games.jsonl` (the per-game timings are
  organic) and showed up only as an impossible wall clock (three rungs credited
  5 seconds for 0.7-2.6 hours of play; the fourth 1,070s for 5.8 hours).
  Consequences that are now rules:
  - The cheap, total check is the inequality `sum(game.seconds) <= job.seconds`.
    It is physically impossible to violate legitimately and catches all four.
  - A run directory must be addressed by the **hash of its inputs** (code sha,
    config, args), not by a human label, and must be write-once. Fingerprinting
    detects the failure; content-addressing makes it unrepresentable.
  - Hash file **content**, not paths: `runs/value.pt` is a mutable path that
    promotion overwrites in place, so a match spanning a promotion silently
    changed engines.
  - One confirmed replay is not an isolated incident, it is a demonstrated
    capability of the harness. Every published number is suspect until traced.
  - Retracted claims get **deleted at the point of use**, not annotated. A
    reader sees `README.md`, never a retraction.

## Cancelled

**Fun to play against. Cut 2026-08-06, by Steven, and not deferred this time.**

It was goal two of the original charter, demoted to "deferred, not cancelled" on
2026-07-29, and it kept coming back: it stayed in this file, it held slot 1 on
the roadmap in `STATE.md`, and every session surfaced the unplayed blind test as
outstanding work. That is the whole reason it is being deleted rather than
demoted again. A deferred goal that resurfaces every session is not deferred,
it is an open item, and this one had been open with no progress since the
project began.

The objective is now exactly one thing: **strength, and understanding what
produces it.** Nothing in this repository measures enjoyment, nothing is gated
on it, and no proposal should be argued for on those grounds.

`scripts/acceptance.py` and its drawn session are DELETED, not left lying
around, because a script that exists is a script that gets suggested. Git
history has it if the question is ever reopened.

**One narrow thing survives, and it is not a goal.** If difficulty levels are
ever built, they come from genuinely weaker models: earlier checkpoints, smaller
nets, human-rating training, calibrated "play the move X centipawns worse than
best". Never a strong model told to blunder at random, which produces twelve
immaculate moves and then a hung queen. That is a design constraint on a feature
that does not exist yet, it costs nothing to honour, and it does not compete
with strength for a single GPU-hour.

## The roadmap, ordered by expected Elo per unit of effort

This ordering is a hypothesis and it is meant to be revised by evidence, not
defended. The lab's job is to falsify it.

1. **Search speed.** 44% of wall clock is inside `python-chess` push/pop and
   move generation, the network is 9%, and the selection loop's `_puct` plus
   its `max()` is 29%. Halving that overhead is worth *something*, but how much
   is exactly the unknown above, so speed is no longer automatically slot 1.
   Note that the obvious fix is already falsified: `_select_child`'s
   own measurements show a numpy array-of-children layout **losing at every
   branching factor chess produces**, because per-call overhead exceeds the
   thirty-iteration loop it replaces. It only wins if selections are batched
   across many nodes at once, which is a different and larger change. The
   honest ladder is: eliminate redundant move generation, then escape
   `python-chess` for the hot path, then kernels, and only as far as the
   profile justifies at each rung.
2. **Search quality per simulation.** Leaf deduplication (the batch is up to
   98.6% duplicated work at large batch), `c_puct` and FPU tuning, which have
   never been measured, policy-prior temperature, and a proper handling of
   transpositions.
3. **Evaluation capacity, and the 9M is NOT capacity-bound.** It is
   *underfitting*: held-out loss 2.1438 sits **below** train loss 2.2106, so
   there is no generalisation gap and nothing has been memorised. The evidence
   once cited for "capacity-bound" (train loss falling while puzzle accuracy
   flattened) is the signature of a data/compute-bound model; a capacity-bound
   model has its *train* loss flatten. The puzzle plateau was 0.679 -> 0.675
   against a sigma of 1.5, so the ruler ran out, not the curve. `lab.py` has
   said this in-tree all along. **Nothing in `runs/` measures d(loss)/d(params)
   at all**: no smaller model was ever trained, so the scaling curve has zero
   points, not two. A 3-point width sweep costs ~6 GPU-hours and should precede
   any further scaling.
4. **Better targets and better data, and action-value is NOT the free win it
   was recorded as.** The "65.7% BC vs 88.9% action-value" pairing is
   apples-to-oranges: 65.7% is a small ablation, 88.9% is a full-data run.
   DeepMind's own *data-matched* comparison has state-value and action-value
   statistically tied (+264+-22 vs +252+-22) with BC the only genuinely weaker
   target. Worse, the action-value bag is shuffled per (position, move), so an
   AV net needs ~35 rows per node instead of 2, roughly 17x the GPU rows,
   which on this profile is plausibly Elo-NEGATIVE. Do not port it on the
   strength of the old comparison.
5. **Self-play RL** on top of the supervised net, once search and speed make it
   affordable. AlphaZero's idea starting from a strong prior instead of from
   zero.
6. **Kernels**, when and only when the GPU is the bottleneck. Search makes this
   inevitable eventually; the profile decides when.
7. **Difficulties and personality.** After all of the above.

## Anti-goals

- **Moving the yardstick.** Changing an eval, a time budget or a seed so a
  number improves is the one unforgivable move here. The research harness
  exists to prevent exactly this.
- **Claiming a win the instrument cannot see.** See measurement discipline.
- **Defending a component because it was hand-written.** Build it, measure it,
  and let it lose if it loses.
- **Optimising something before profiling it.** Correctness, then algorithm,
  then kernels, in that order.
- **Weakening the engine on purpose** anywhere on the main line. Difficulty is
  a separate, later, derived artifact.
- **Delivering something finished the author did not participate in building.**
  This one is unchanged and it is why the project exists.
