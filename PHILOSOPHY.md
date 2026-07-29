# What SumoFish is for

Read this before proposing anything. It is the objective function: not *how*
the engine is built, but *why*, and when the two conflict, this wins.

**Rewritten 2026-07-29. The objective changed.** Earlier versions of this file
ranked "fun to play against" above strength and listed Elo as an anti-goal.
That is no longer true and you should not act on any summary of this file
written before this date.

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

Everything else — puzzle accuracy, held-out loss, bits-per-move, nodes per
second — is an *instrument*, not the target. Instruments are for deciding what
to try next and for catching a broken run early. They are never the reason to
ship. A change that improves held-out loss and loses the match did not work.

### What "understand every layer" means now

It survives the rewrite, and it is the reason this is worth doing at all, but
its role has changed. It is no longer a veto over strength. It is a *method*.

- Do not hand over a working black box. Prefer the version that can be read,
  modified and broken. Explain the mechanism, not just the result: the actual
  tensor shapes, the actual bytes, the actual profile.
- Hand-written CUDA, custom kernels and from-scratch implementations are goals,
  not premature optimisation — but they get built **when the profile says they
  are the bottleneck**, not before. This project already burned most of a
  session planning kernels for a network that was 5% of its own search. Chasing
  strength is what makes that mistake visible; that is the point of a
  scoreboard.
- Chores stay chores. Deploy plumbing, download scripts, systemd units, config
  files: automate them and move on. The domain — chess, ML, search, GPU — gets
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

1. **Evaluation quality** — how good the network's judgement of a position is.
   Bought with parameters, data, training compute, and target design.
2. **Search** — how much lookahead that judgement gets multiplied by.
   **The exchange rate is currently UNKNOWN.** It was believed to be ~+50 Elo
   per doubling from "two independent estimates agreeing". Both are retracted
   (2026-07-29): all four rungs of the measured ladder were produced by
   *replayed* match logs rather than by the jobs credited with running them, and
   the second estimate was fitted so the first would agree with it. Until the
   ladder is re-earned under a fingerprinted harness, no proposal may be
   justified by an Elo-per-doubling figure.
3. **Speed** — search per second, which converts directly into (2) on a clock.
   Currently CPU-bound in `python-chess`, not GPU-bound. Speed *is* strength
   here in a way it never was for a searchless engine.
4. **The measurement loop** — how fast a wrong idea can be killed. This is the
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
  opening's bias and is worth ~2.4x in games.
- **A mirror match is structurally blind to anything both sides share.** It can
  answer "did this change help". It cannot answer "how does SumoFish play" or
  "what is its draw rate". Those come from real games only.
- **Adjudication by the engine under test is a bug**, because a net that is
  more confident rather than more correct banks more wins. Stockfish is the
  third-party adjudicator and the absolute anchor.
- **Select checkpoints on held-out loss, not on a noisy eval.** Best-of-twenty
  on a ±1.5% metric is biased upward by about two sigma, and it already
  promoted the marginally worse of two checkpoints once.
- Report `unique/s`, never raw nps, for anything touching batched search.
- **A number without provenance is not a result either.** On 2026-07-29 it was
  found that all four rungs of the exchange-rate ladder were produced by
  *replaying* existing match logs: `match.py` keys resume on the game index
  alone, so a job with different code and config lands on an existing directory
  and reports it as its own work, and `match.py:458` then rewrites `config.json`
  over it. The replay is invisible in `games.jsonl` — the per-game timings are
  organic — and showed up only as an impossible wall clock (three rungs credited
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

## Deferred, not cancelled

**Fun to play against.** It used to be goal two. It is now a thing to build
*after* there is something strong to build it from, and the argument for it is
unchanged and still correct: a handicapped strong engine plays twelve immaculate
moves then hangs its queen for no reason, and that is miserable. Human weakness
clusters where the pattern is hard to see; random weakness does not.

So the constraint that survives is narrow and it is a **design constraint on
the difficulty feature, not on the engine**: when difficulties are built, they
come from genuinely weaker models — earlier checkpoints, smaller nets,
Maia-style human-rating training, calibrated "play the move X centipawns worse
than best" — never from a strong model instructed to blunder at random. Nothing
about pursuing maximum strength conflicts with this. It is a prerequisite for
it: you cannot derive a good 1500 from a bad 2400.

`scripts/acceptance.py` (the blind human test) still exists and is still worth
running eventually. It is no longer a gate on anything.

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
   `python-chess` for the hot path, then kernels — and only as far as the
   profile justifies at each rung.
2. **Search quality per simulation.** Leaf deduplication (the batch is up to
   98.6% duplicated work at large batch), `c_puct` and FPU tuning, which have
   never been measured, policy-prior temperature, and a proper handling of
   transpositions.
3. **Evaluation capacity — and the 9M is NOT capacity-bound.** It is
   *underfitting*: held-out loss 2.1438 sits **below** train loss 2.2106, so
   there is no generalisation gap and nothing has been memorised. The evidence
   once cited for "capacity-bound" (train loss falling while puzzle accuracy
   flattened) is the signature of a data/compute-bound model; a capacity-bound
   model has its *train* loss flatten. The puzzle plateau was 0.679 -> 0.675
   against a sigma of 1.5, so the ruler ran out, not the curve. `lab.py` has
   said this in-tree all along. **Nothing in `runs/` measures d(loss)/d(params)
   at all** — no smaller model was ever trained, so the scaling curve has zero
   points, not two. A 3-point width sweep costs ~6 GPU-hours and should precede
   any further scaling.
4. **Better targets and better data — and action-value is NOT the free win it
   was recorded as.** The "65.7% BC vs 88.9% action-value" pairing is
   apples-to-oranges: 65.7% is a small ablation, 88.9% is a full-data run.
   DeepMind's own *data-matched* comparison has state-value and action-value
   statistically tied (+264+-22 vs +252+-22) with BC the only genuinely weaker
   target. Worse, the action-value bag is shuffled per (position, move), so an
   AV net needs ~35 rows per node instead of 2 — roughly 17x the GPU rows,
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
