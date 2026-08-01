# SumoFish — project context

> ## READ `PHILOSOPHY.md` FIRST. It is short and it is the objective function.
>
> **It was rewritten 2026-07-29 and the objective changed.** Ignore any summary
> of it older than that, including earlier versions of this block.
>
> 1. **The goal is the strongest engine one person can build on one GPU, and
>    understanding every mechanism that makes it strong.** One goal, not two.
>    Cost and time are not constraints; 16 GB of VRAM and attention are.
> 2. **Strength means Elo against fixed external opposition** (Stockfish at
>    pinned levels), then the lichess rating, then pair-scored matches against
>    the previous build. Puzzle accuracy, held-out loss and nps are
>    *instruments*, never the target.
> 3. **Understand every layer is a method, not a veto.** Build it by hand, then
>    measure it, then keep whichever version wins. Do not defend a hand-written
>    component that loses. Chores still get automated.
> 4. **Measurement discipline is not optional.** A number without an interval is
>    not a result. Mirror matches are blind to what both sides share. Select on
>    held-out loss, not on a noisy eval. Never move the yardstick.
> 5. **Fun to play against is deferred, not cancelled.** When difficulties are
>    built they come from genuinely weaker models, never from a strong model told
>    to blunder. That is a constraint on that feature, not on the engine.

A chess engine that evaluates positions with a transformer and searches with
MCTS over a value net, with a separate policy net supplying priors. The
evaluation lineage is Ruoss et al. 2024, *Grandmaster-Level Chess Without
Search* ([arXiv:2402.04494](https://arxiv.org/abs/2402.04494)); searchless was
the starting point and is no longer the design.

This file is the operational layer: how to run things and what not to retry.
See `PHILOSOPHY.md` for why the project is shaped the way it is.

## Where things stand (2026-08-01, session 7)

> **The 2026-07-29 overnight block that stood here is deleted, not demoted, per
> this file's own rule.** Its run finished. Every unit-state claim in it had also
> gone false, in the direction that matters: it said the bot no longer autostarts
> (it is `enabled` again) and that the train watchdog was STOPPED (it is `active`
> and `enabled`, firing every 5 minutes). A reader trusting it would have
> mis-modelled what can wake up and take the GPU.

> **CURRENT, 2026-07-31 ~21:35.**
>
> - **The 9M continue run FINISHED.** `runs/9M-sv-continue` reached step 600,000.
>   Best held-out is step 595k: **val 2.1117, puzzles 0.699**. The unit is gone
>   (it was transient; `is-enabled` reports `not-found`, which is expected and
>   not a fault).
> - **DEPLOYED 2026-07-31 23:17:28.** `runs/value.pt` is now the 600k checkpoint,
>   selected on held-out loss per PHILOSOPHY's checkpoint rule and NOT on a match:
>   the sizing pilot went 4/4 draws by threefold repetition at 406s/game, so the
>   300-game gate would have cost ~34 GPU-hours to most likely terminate on the
>   null. Both arms share a policy net and are two checkpoints of one run, which
>   is close to the mirror-match blindness PHILOSOPHY warns of. The promotion is
>   UNMEASURED in Elo terms; `scripts/promote.py --rollback` reverses it and
>   `runs/value.pt.previous` holds the outgoing net.
> - **The bot is DOWN** (stopped 21:12:20, clean exit, no game abandoned) and its
>   opponent window was changed today; see "Open, smaller". It is `enabled`, so it
>   WILL come back on next login/boot. That is a hazard during any wall-clock
>   experiment: `systemctl --user disable sumofish-bot` if one is due to span a
>   reboot.
> - **`sumofish-train-watchdog.timer` is ACTIVE**, every 5 minutes. It is
>   currently harmless -- it logs "no live training process; nothing to watch" and
>   exits, verified at 21:25 and 21:30 with a match already running, so it does
>   NOT mistake a match for training. But its remedy is restarting the LAB, so it
>   is only harmless while no training process exists.
> - **The lab is unparked.** `current` was `train-136m` for ~2.7 days; that job is
>   now DELETED from the plan (see roadmap 7/8 below) and `current` is cleared, so
>   the runner starts from the first unsatisfied job. **It is still inactive and
>   starting it now begins ~10 GPU-h of re-earning the retracted ladder** -- the
>   four `sims-*` rungs are deliberately marked FAILED, not completed.
> - **Nothing is measuring strength right now.** The pilot was stopped at 4 games
>   once its purpose (sizing, and the 4/4 draw finding) was served; its games and
>   config are kept. The bot is down for the overnight retrain, so the lichess
>   anchor is paused too.
>
> > **HOW TO READ THE RATING, and why it is confounded.** Two things changed
> within 20 minutes of each other on 2026-07-31, so any rating movement from
> here has two candidate causes and the rating alone cannot separate them:
>
> | when | change |
> |---|---|
> | ~21:05 | opponent window 1846-2846 (symmetric) -> 2200-3000 (asymmetric, harder) |
> | 23:17:28 | value net ~300k-era -> 600k checkpoint |
> | 23:17:50 | bot restarted, both changes live together |
> | ~03:15 (08-01) | bot drained for the policy retrain |
>
> They push in OPPOSITE directions, which is the awkward part: a better net
> should raise the rating, a harder pool should lower it, and a null result is
> equally consistent with "both worked" and "neither did".
>
> First data, 08-01 03:15: **8 games, rapid 2339 -> 2357 (+18), 170 -> 178
> games.** That is not a result and must not be quoted as one -- 8 games is worth
> roughly +-300 Elo, so the interval swamps the point estimate by an order of
> magnitude. What it is: not negative, which is weak evidence against the worry
> that the harder pool would sink the rating outright.
>
> **Two of the games after that are ARTIFICIAL LOSSES and must be excluded.**
> Steven conceded the two in flight (`WUtbaAsG`, `qCL90DHS`) at ~03:14 to free the
> GPU for the overnight run. They are resignations by the operator, not by the
> engine, and they are indistinguishable from real losses in `logs/rating.jsonl`
> and in the lichess history. Any rating read that spans 2026-08-01 03:14 is
> carrying two losses the engine did not earn -- worth roughly -14 rapid Elo at
> RD 45. **The clean baseline for judging the promoted net is the rating BEFORE
> 03:14, or a fresh sample after ~30 more games have washed them out.**
>
> To actually separate them, revert ONE and hold the other for ~50 games. The
> cheaper revert is the window (`config/lichess-bot.yml`, one line); the net has
> `scripts/promote.py --rollback`. Doing neither is also defensible -- both
> changes are believed good and the combined effect is what gets played -- but
> then stop treating the rating as evidence for either one individually.

> **OVERNIGHT RUN FINISHED 2026-08-01 11:31, and it is a clear win.**
> `runs/9M-bc-2026-08-01` -- the policy prior retrained, roadmap item 4, the
> "biggest neglected lever", frozen at 40.9% puzzles since session 1. Warm start
> from `runs/policy.pt` transferred **93/93 tensors**; 300,000 steps at lr 2e-4;
> exit 0.
>
> | | incumbent `runs/policy.pt` | new `best.pt` |
> |---|---|---|
> | held-out loss | 1.66670 | **1.59138** (-0.0753) |
> | puzzles | 0.409 | **0.435-0.447** |
>
> The incumbent's held-out number did not exist before today -- `runs/9M-causal`
> logged 31 puzzle evals and zero `val_loss` -- so `scripts/eval_heldout.py` was
> written to compute it on `train.py`'s own terms. It reproduces train.py's
> logged 1.59485 for step 280k EXACTLY, which is what licenses the comparison.
>
> **NOT PROMOTED.** The policy net shapes every search the engine runs, and
> `scripts/promote.py` only swaps `runs/value.pt`, so this needs a deliberate
> decision and a different mechanism. Held-out loss is also an instrument, not
> the target. `best.pt` == `final.pt`, i.e. the run was still improving at step
> 300k and never overfit -- **more training is available for free.**
>
> Note for whoever tunes next: `c_puct`/FPU tuning is gated behind this retrain
> (roadmap 4 before the tuning), so the values depend on the prior's sharpness
> and should not be settled against the OLD prior.

**Standing rule, promoted out of the deleted block because it is not status:**
> if you run two Claude sessions at once, say so up front. A second session
> starting the bot during a wall-clock match silently invalidates the match, and
> starting it during training halves the throughput of both. This has happened
> once already (an unexplained bot start at 19:42:31 on 07-29, from outside the
> repo -- the watchdog declined and nothing in-tree does it).

**The engine is Rust, provably identically, and 3.6x faster. The two extra
speed flags were measured and cost 168 Elo, so they are off.**

**1. The Rust core is integrated and live-capable.** `CHESSGPU_CORE=rust`
selects `sumofish/rust_mcts.py`; unset keeps Python, so rollback is an
environment variable. Ten oracles plus `tests/identity_engine.py` (which uses
the REAL nets, not the port's mock) show byte-identical root visit vectors.
Identical visits means identical moves, which is why this shipped without a
match: it is the one claim on the board that does not depend on the retracted
exchange rate.

Measured **3.6x** on the engine profile, which is **4.5x more search on the
same clock**. The earlier "8.9x" was extrapolated from the *Python* profile
where the network was 9% of wall clock; with the tree in Rust the network is
**89.3%** and the tree is 1.3%, so the same arithmetic gives a different answer.
Re-profile after every port.

**Re-measured 2026-07-31 on an idle box** (`runs/lab/profile-2026-07-31.json`,
batch 64). Neither of the two figures above had an artifact behind it; only one
survives:
- Network share in **Rust: ~100%** (network 102.2%, tree -2.2%). Over 100% is
  slop -- a synthetic full batch costs marginally more than the search's real
  ragged ones -- and means the tree is *unmeasurable* at this batch, not that
  the arithmetic broke. So **89.3% is essentially confirmed**; if anything it
  understates it.
- Network share in **Python: 38.2%, not 9%**. The 9% is the bad number. Every
  extrapolation that used it as the denominator was wrong by ~4x.
- **The 3.6x does not reproduce here: 2.67x** (3,041 -> 8,135 nps at batch 64).
  Not necessarily a contradiction -- different batch, different day -- but 3.6x
  has no artifact either. Re-earn it before quoting it again.

**2. `dedup` and `compile` are OFF, and that is a measured decision, not
caution.** Both are faster per call and both preserve the tree at a fixed
simulation count. At a fixed *clock* they cost **-168 Elo** (20 games, W0 D11
L9, LOS 0.0%). The diagnostic: at 0.5s the fast arm ran 7,297 nominal
simulations against plain's 4,161 and got **3,464 unique evaluations against
plain's 4,160**. It bought 75% more claimed search and 17% less knowledge,
because dedup frees network time, the search spends it on more descents, and
those descents collapse onto leaves already evaluated. Duplicates still back up
values, so visits and Q inflate on no new information.

The general rule this produced is now in PHILOSOPHY: **an identity proof at
fixed simulations says nothing about strength at a fixed clock.** They are
different experiments, and only the second one gets played.

**3. The measurement loop is honest, and the archive is not.**
`scripts/match.py` no longer caps `--time` at `--sims`, refuses to resume across
a spec change (fingerprinting code + config, exiting 2), and adjudicates only
when a fixed-node Stockfish agrees. `scripts/verify_replays.py` reports **0 of 8
matches trusted, 4 REPLAYED, 4 unprovenanced**. See
`docs/2026-07-29-ladder-retraction.md` for what that invalidated and
`docs/induced-failures.md` for the two bugs found by inducing the guard.

**Consequence, still in force: no proposal may be justified by an
Elo-per-doubling figure.** The ladder can be re-earned for ~10 GPU-hours,
visit-denominated, with a distinct seed per rung. It buys the ordinal only.

**4. Six supervisors were no-ops and are fixed.** `train_watchdog.py` watched a
unit that does not exist; `watchdog.py` could not restart a unit in `failed`
because it never called `reset-failed`; `promote.py` wrote `runs/current.pt`,
which nothing reads, gated on a metric PHILOSOPHY forbids selecting on, and
restarted the bot for a swap that needs no restart. `research/run.py` now
enforces the frozen constants and the editable region it always claimed to.

**5. The 136M run was killed.** `--init-from` transferred **0 of 93 tensors**
(donor width 256, target 1024), so it was the cold start its own job comment
said must not happen, and it was scheduled for 102.4M positions, 19.3% of an
epoch, against the 9M's 307M. A promotion match against it would have measured
the defect, not the width.

**6. CUDA streams are not the shortcut.** Two streams alone is 1.04x; 1.19x on
top of `compile`. Halving the two forward passes honestly costs the ~40
GPU-hours a shared-trunk two-head net needs to train. At 89% network share that
is now the single largest speed item, which is a reversal: it was item 4 when
the network was 9%.

## Next session, in order

**Done: the Rust core is deployed to rated play, and now the default.**
Live with `CHESSGPU_CORE=rust` since 2026-07-30 14:32, both speed flags off.
As of 2026-07-31, `select_mcts_class()`'s default flipped too: an unset
`CHESSGPU_CORE` now selects Rust, not Python -- `CHESSGPU_CORE=python` is the
rollback, same as `=rust` used to be the opt-in. `sumofish.mcts` (the
pre-port Python search) was NOT retired: it's still exactly where it was and
still the oracle `tests/identity_*.py`, `scripts/match.py` and
`scripts/acceptance.py` compare Rust against, it's just no longer what an
absent env var silently falls back to. Verified with a new
`tests/verify_core_default.py`; existing `tests/verify_rust_flag_guard.py`
is unaffected (it exercises `ignored_rust_flags` on an explicit dict, not
`select_mcts_class`'s default).

**1. Play the acceptance test. It has still never been run.**
`scripts/acceptance.py --games 20`. Twenty blind games, one rating each, arms
hidden until `reveal`. Every other number here measures strength, which
PHILOSOPHY ranks third; this is the only thing pointed at goal two, and it
needs a human by construction.

**2. Re-earn the exchange ladder** (~10 GPU-h). Visit-denominated, distinct seed
per rung, on an idle machine. Until it exists, every "worth N Elo" in any plan
is a guess, and the plan's ordering is unfalsifiable.

**3. Stockfish as an absolute anchor**, at pinned nodes. Every match here is
relative, so the whole ladder floats. The adjudicator half is done.

**4. Retrain the policy prior.** Frozen at 40.9% puzzles since session 1 and it
is the biggest neglected lever: it sets the shape of every search, and
`c_puct`/FPU/temperature tuning is not worth doing before it, since the right
values depend on the prior's sharpness.

**5. Shared-trunk two-head net** (~40 GPU-h). Worth ~1.8x now that the network
is 89% of the search. Warm-start it: 91 of 93 tensors transfer from the 9M body.

**6. Cross-game batching.** N games feeding one evaluation service. The engine
is launch-bound below ~128 rows -- one row costs the same 7.2ms as 128 -- but
that is a per-call latency observation, not an end-to-end throughput number.
**Measured 2026-07-30** with `scripts/batch_payoff.py` (self-play, sims=150,
32 games/arm, concurrency=8, GPU shared with a live training run at 100%
util the whole time): **2.45x games/hour**, batched vs. today's one-game-at-
a-time. Smaller runs landed 2.16x-3.99x, noisy at low n. So: real and worth having eventually, but "close to free" overstated it --
this is a moderate multiplier under contention, not an order of magnitude,
and it is NOT a blocking dependency for verifying the 3 MCTS defect fixes
(leaf dedup, virtual loss, mate distance -- see "Open, smaller" below): that
verification is affordable on today's unbatched harness per the audit's own
GPU-hour estimate. Production integration (adjudication, PGN/games.jsonl
logging, SPRT stopping across concurrent games) is real additional work the
sizing script does not do.

**7. The scaling curve has zero points.** A width sweep (~6 GPU-h) at matched
tokens is what makes "scale it up" an argument instead of a preference.
**Now IN THE PLAN, 2026-08-01**: `sweep-tiny` / `sweep-9m` / `sweep-136m` +
`scaling-curve` in `scripts/lab.py`, three arms at 20k steps and batch 1024
(20.5M positions each) sharing `--seed 1234`, so the arms differ in width and
nothing else. Matched TOKENS, not matched clock -- matching clock would hand the
small net 15x the data and measure the two effects summed. They replace
`train-136m`, which is deleted: see below.

**8. `d(loss)/d(params)`, then kernels.** Not before 4-7. Item 7 is now the
thing that produces it.

**`train-136m` is DELETED from the lab plan (2026-08-01), not deferred.** It was
35 GPU-hours resting on `scale_bar = D x log2(m) = 265 Elo`, and both inputs were
unsound: `m` was 2.17 measured on the *Python* tree (really 3.06, so 1.61
doublings of forgone search, not 0.83) and `D = 237.2` came from the retracted
ladder. The 9M it would have replaced is measured to be underfitting and still
improving from training alone. The sweep answers the same question for a sixth of
the compute and answers it *before* spending. `runs/lab/state.json.bak-2026-07-31`
restores the old plan if wanted. `match-136m` went with it, and `promote` now
needs only `match-9m-long`.

**The four `sims-*` ladder rungs are now marked FAILED in the lab state**, not
completed, and their Elo figures are withheld from the status board. They were
the replayed matches; leaving them rendering as `+218 / +283 / +237` on the
`sumofish-lab` board was a live-looking claim at a point of use, which is exactly
what PHILOSOPHY says to delete rather than annotate. **Consequence to be aware
of: the lab will now try to re-earn them (~10 GPU-h) the next time it runs.**
That is roadmap item 2 and is intended -- but it is a real resource commitment
that will start on its own, so decide before `systemctl --user start sumofish-lab`.

**Open, smaller:**

- ~~Remove `opponent_max_rating: 2200`~~ **Done 2026-07-31.** Rapid is settled
  (2346, RD 45, 168 rated games) so the training wheel came off. The real find
  was that it had never been ON: `opponent_rating_difference: 500` silently
  overrides both bounds whenever the rating is known (upstream
  `matchmaking.py:175-178`), so the effective window was always [1846, 2846] and
  the 2200 ceiling was dead config. Replaced with an ASYMMETRIC window,
  `opponent_min_rating: 2200` / `opponent_max_rating: 3000`, because the bot pool
  is bottom-heavy relative to us and a symmetric window cannot stop feeding us
  weaker opponents at any width. Measured against the live 333-bot online list:
  the old window gave 150 candidates, 94 of them BELOW us; the new one gives 114,
  only 31 below. These bounds are static and do not track the rating -- revisit
  if rapid moves more than ~150.
- **Fixed 2026-07-30.** This line previously said "virtual loss is applied to
  N and not Q" -- backwards. Direct code read confirmed the actual defect was
  the opposite: virtual loss was applied via a real `backup()` call, so it
  went straight into `value_sum` (Q) at every node on the path, not just a
  visit count. Now behind a `vloss_fix` flag (`rust/src/tree.rs`), default
  off, proven correct by 5 new Rust unit tests (never leaks, never touches
  `value_sum`, and is proven to actually change search rather than being a
  silently-inert no-op). It needs Elo, so it still waits for the ladder --
  the fix existing is not the same as the fix being worth shipping.
- A blocking pre-push hook running `verify_replays.py --check`,
  `tests/verify_data.py`, and `tests/run_all.sh`.
- Write down the operating point: opponent pool, time control, and how the
  budget is distributed. Several arguments have quietly assumed different ones.
- `CHESSGPU_BATCH` defaults to 64 and 256 is faster per call, but batch size
  trades against search quality (more virtual loss in flight, more collisions).
  Given item 2 above, settle it on `unique/s` and then in a game, never on nps.
- Time management: ponder, early stopping, instamove. Untouched.
- The value net enables resignation, draw offers, and calibrated difficulty.

**Numbers worth remembering:**
- Behavioural cloning, 8.5h, 307M positions -> 40.9% puzzles. Now the search's
  policy prior.
- State value warm-started from its body -> 57.4% puzzles at 12% trained.
- The 9M state-value curve: 48.6% at 10k, 64.8% at 100k, 67.0% at 150k, 68.7%
  at 280k; adjacent evals bounce 0.77 points. Flat from ~200k, while train loss
  was still falling (2.2422 at 200k, 2.2133 at 292k). That is **underfitting**,
  not capacity-bound -- 307M samples is under one epoch of a 36GB bag, so
  overfitting is not available as an explanation. An earlier version of this
  file asserted the opposite and steered a 37-hour run.
- **Extended to 600k, and it is STILL underfitting.** `runs/9M-sv-continue`:
  val 2.1522 at 305k -> **2.1120 at 600k**, puzzles 0.682 -> **0.700**. Held-out
  loss sits *below* train loss at all 63 evals and the gap never widens
  (-0.092 at 305k, -0.066 at 600k), so nothing has been memorised. Val was still
  falling at the end (-0.0049 over the last quarter). Caveat that cuts against
  the point and is recorded anyway: there is no dropout in `model.py`, so that is
  not the cause, but train loss is a running average while val is a clean pass,
  so some of the negative gap is bookkeeping. The load-bearing part is that the
  gap does not WIDEN across a 2x extension. **Doubling the training at fixed
  parameters bought real held-out gain at zero cost per move in a game.**
- **The cost of scaling, re-measured on the core that plays** (2026-07-31): a
  136M value net costs **3.06x per node**, = **1.61 doublings of search
  forgone**, not the 2.17x / 0.83 doublings in `runs/lab/state.json`. That
  stored figure came from `bench_search.py` back when it hardcoded the *Python*
  tree. `m` is not a property of the net, it is the net divided by the tree
  around it -- so **the Rust port made every future scale-up more expensive**,
  and did so invisibly, because `m` is an input to the port's justification
  rather than an output of it. Cost side measured; benefit side still zero
  points. See `runs/lab/profile-2026-07-31.json`.
- Puzzle accuracy at n=1000 has a binomial sigma of +-1.5%, larger than the
  entire 150k->300k gain anyone was reading off it.
- Post-port profile: network **89.3%**, tree 1.3%. Launch-bound below ~128 rows.
  Re-measured 07-31 at batch 64: network ~100%, tree unmeasurable. The Rust
  search is network-bound to the point where tree work does not show up.
- The prior softmax **cannot** be ported: numpy's float32 `exp` is not correctly
  rounded and differs from a correctly-rounded double `exp` on 39.7% of inputs,
  which reaches 95.4% of positions getting at least one differing prior. A
  1-ULP prior only changes a move on a PUCT tie, so it is exactly the error
  class that passes a casual test. `rust/src/softmax.rs` documents it.

**Left unanswered:** what looks wrong about the evaluation panel. The two-games-
on-one-curve bug is fixed, but an earlier complaint about that box was never
pinned down. Ask for a screenshot of just that panel rather than guessing.

## Layout

```
sumofish/            frozen infrastructure. tokenizer/bagz are VERIFIED, do not edit
  tokenizer.py       77-token FEN encoding + 1968-move action space
  bagz.py            ChessBench container + Apache Beam record decoding
  data.py            streaming loader (see Lab Notes on why it streams)
  model.py           LLaMA-shaped decoder, ports upstream exactly
  policy.py          logits -> legal move (masking is load-bearing)
  rules.py           is the game over here, and what is it worth. read it
                     before "optimising" it back to board.outcome()
  evaluate.py        puzzle protocol, ported faithfully
  engines/           random_engine.py (Phase 0) and neural_engine.py (real)
  telemetry.py       the engine's narration channel: append-only JSONL, off-thread
train.py             the production run
research/            karpathy/autoresearch port: train.py is agent-editable
tests/verify_data.py the correctness gate. run it after touching sumofish/
tests/verify_search.py the same, for rules.py, tokenize_board and tree reuse
tests/verify_layout.py the same, for the dashboard's board column: what Plan
                     reserves is what board_panel draws. No torch, no terminal
scripts/match.py     head-to-head match play. the measurement instrument
scripts/elo.py       its statistics, separate so lab.py need not import torch
scripts/lab.py       the experiment queue: the plan, the runner, the viewer
scripts/promote.py   swap the live bot to a checkpoint
dashboard/           the `sumofish` dashboard, in Rust. Read dashboard/CLAUDE.md.
                     Nine crates; sf-panels is the enforcement point (it depends on
                     nothing that can do I/O). Its own gate sweeps every terminal
                     size and asserts panel presence is monotone.
  sprites.py         GENERATED cburnett piece bitmaps; make_sprites.py rebuilds it
systemd/             four --user units, symlinked into ~/.config/systemd/user
reference/           upstream source, gitignored, for diffing. read-only.
```

## The lab, and what it is allowed to decide

`scripts/lab.py` is an ordered list of jobs and a runner that walks it, under
`sumofish-lab.service`. It exists because everything left on the roadmap needs
the GPU, needs hours, and needs to happen in an order where later steps read
earlier results -- which by hand means being present at every handoff, at 3am,
for two days.

    sumofish-lab            what it has run, what it concluded, what is next
    sumofish-lab watch      the same, refreshing
    sumofish-lab reset --job <id>    forget one job so it runs again
    journalctl --user -u sumofish-lab -f

Two kinds of job. A **command** is a subprocess that owns the GPU, gets a
wall-clock deadline, and is stopped with SIGTERM rather than SIGKILL because
`train.py` checkpoints on SIGTERM and killing it outright throws away hours. A
**decision** is a Python function that reads earlier results and returns facts
later jobs interpolate; that is what lets the plan branch with nobody awake.
State lives in `runs/lab/state.json`, so restarting the unit resumes rather
than restarting, and a reboot mid-run costs the run and not the plan.

**The boundary is the important part.** It measures freely and changes exactly
one thing: `runs/value.pt`, and only when a match of >=300 games says the
candidate is better with >=95% LOS *and* the Elo interval excludes zero. The
old checkpoint is kept at `runs/value.pt.previous` and the swap is a file copy
the bot picks up on its next game, so it is reversible with another file copy.
It does not edit code, does not touch units, and does not change engine
defaults. A match concluding "batch 256 is free" goes in `runs/lab/report.md`
for a human to act on, because applying it is a code change.

**Two failure modes are designed against explicitly.** Foreign-process
detection reads `/proc` rather than calling `pgrep -f`, because this file's own
command line contains the strings it searches for and pgrep would match itself
-- a trap in the Lab Notes below that has already been hit twice here. And the
runner waits for quiet before *decisions* as well as commands, because a
decision that reads a log still being written is a decision made on part of the
evidence; the first draft would have chosen an attention mask for a 35-hour run
off a 3,000-step sample.

## The dashboard, and why it is shaped that way

`sumofish` is the instrument for PHILOSOPHY.md's "watch real games". Four
decisions in it are load-bearing and none are obvious, so they are written down
here rather than left to be rediscovered.

**The engine narrates; the dashboard only listens.** Every number about the
search comes from `logs/engine.jsonl`, written by `search_engine.py` during the
search as well as at the end of it. The viewer never imports torch, never
evaluates a position, and therefore cannot disagree with the engine or take GPU
time from it. Measured cost to the engine: none detectable (4353/4289 nodes
with narration on, 4289/4417 with it off, at a fixed 3s budget). Setting
`CHESSGPU_TELEMETRY=` empty turns it off completely, including the work of
*producing* the records.

**The channel is an append-only file, not a fifo or a socket.** `open(fifo,"w")`
blocks until a reader attaches, which would put an unbounded stall inside
`choose()` on a running chess clock whenever the dashboard happened not to be
running. The engine must never be able to block on whether anyone is watching.

**Three layers, one direction.** `dash/sources.py` writes into `dash/state.py`,
`dash/panels.py` reads it. Sources are threads with their own cadences (a
bullet move and a fifteen-minute rating sample are not the same kind of event);
the render loop is a pure function of state at a fixed frame rate. That is why
a lichess timeout degrades one panel instead of freezing the screen, which the
old single-timer loop could not do.

**The board is a real image when the terminal can take one.** Konsole answers
`ESC[c` with a `4` in the list, which is the standard claim of sixel support,
and renders sixel with no configuration at all. So `dash/sixel.py` rasterises
`chess.svg.board()` -- the same cburnett SVGs, in lichess's own board colours --
and places it with a cursor move. That removes the resolution ceiling entirely:
the board is the one from the website, not an impression of it. It needs
`rsvg-convert` and ImageMagick at run time, so it probes and silently falls
back to the text renderer when anything is missing.

**The pieces are cburnett, not an approximation of it.** (Still true, and it is
what draws the board wherever sixel is unavailable.) That is the set
lichess draws by default, and `python-chess` already ships the SVGs, so
`dashboard/xtask/codegen_cburnett.py` lifts them out of python-chess, and the
board is rasterised at the pixel grid the
terminal can afford (8, 16 or 24 px per square) and checks the result in as
`sprites.py`. Two bytes per pixel, luminance and alpha, so the renderer can
substitute any two inks and keep the artwork. `rsvg-convert` and ImageMagick
are needed to regenerate, never to run.

**The board column is sized to the picture, and the picture is centred in it.**
`board_w` is the image plus `BOARD_GUTTER` on *both* sides, `board_h` is the
image plus `PLAYER_ROWS` above and below, and the player blocks are set to the
image's own left and right edges rather than to the column's. Two rows per
player, not one: the name, rating, clock and win estimate on one and the
captured material on the other. Those rows are not new space -- they are the
rows the old layout reserved and left blank under the board, which is what made
the picture look pinned to the top of a column it was not filling. The material
is grouped and counted (`♟6`) because six pawns drawn as six identical
eight-pixel silhouettes is a texture, not a number.

**The board gets the width the panels beside it do not need.** `RIGHT_COLS`
(60) is the working limit on how big the picture gets, not `BOARD_SHARE`: on a
16:9 window a square board is width-bound long before it is height-bound, so
the share only bites on a terminal wide enough that half of it would overflow
the column's height anyway. 60 is where the ladder's bar comes down to half of
`GAUGE_MAX`, which is the first thing over there that visibly loses by being
narrower; `MIN_WIDE_COLS`'s 52 is what those panels need to be *correct*, and
the 8 columns between the two are worth a third of the board's area. Measured
at 160x96: 598px of board before, 754px after, and the right column still 62.
`tests/verify_layout.py` is the gate on all of this.

**Staleness is in the data model, not the styling.** `Field.track` is live /
coast / lost and every panel prints it. The failure it exists to prevent is the
old loop's `profile = api(...) or profile`, where a dead connection rendered
pixel-identical to a live one.

## Commands

```sh
sumofish-lab                             # the experiment queue: state and conclusions
sumofish-lab watch                       # the same, live
sumofish                                 # the dashboard
sumofish demo                            # same layout on a fixture, no game needed
sumofish mind                            # raw engine telemetry as it is written
tests/verify_data.py                     # correctness gate, run after any sumofish/ change
tests/verify_search.py                   # the same, for the search's rules and tokenizer
tests/verify_layout.py                   # the same, for the dashboard's board column

# "did that help?" -- two checkpoints, equal thinking, a few hundred games
scripts/match.py --a-value runs/A.pt --b-value runs/B.pt --games 400 --sims 400
# what a speedup is worth, without needing the old code to compare against
scripts/match.py --a-sims 1000 --b-sims 400 --games 300
# fixed time instead of fixed sims: strength as deployed, speedups included
scripts/match.py --a-value runs/A.pt --b-value runs/B.pt --time 0.5
journalctl --user -u sumofish-train -f  # watch training
systemctl --user status sumofish-bot    # the lichess bot
scripts/promote.py runs/9M-causal/best.pt
research/run.py --note "hypothesis"      # one autoresearch experiment
research/run.py --status                 # leaderboard
```

Python is `.venv/bin/python` (3.12). Never the system python.

## Non-negotiables

- **Never edit `sumofish/tokenizer.py` or `bagz.py`** without re-running
  `tests/verify_data.py`. The tokenizer is byte-exact against DeepMind's
  published implementation over 12,000 real positions. Break that and every
  number stops being comparable and their pretrained checkpoints stop being a
  valid reference.
- **Never change the eval, the time budget, or the seed in `research/train.py`.**
  Moving the yardstick to make the metric improve is the exact failure the
  harness exists to prevent.
- The lichess token is in `~/.config/sumofish/bot.env`, chmod 600, outside the
  repo. Refer to `LICHESS_BOT_TOKEN` by name only. Never echo it.
- SumoFish plays **casual only** until the engine genuinely tries to win.
  Rated play with a weak-or-random engine is sandbagging under lichess ToS.


---

## Where the rest went

This file was 68 KB and growing, and its problem was not length: a true statement
and a false one were typographically identical, so a stale claim read exactly like
a current one. `CLAUDE.md` open item 5 said "the 9M is capacity-bound" while
`lab.py` said, in-tree, "the 9M is UNDERFITTING, measured" -- and the false one
steered a 37-hour training run.

Split 2026-07-29:

* **`STATE.md`** (this file) -- small, current, REWRITTEN each session. Allowed to
  be wrong only until corrected. If a section here is older than the session it
  describes, delete it rather than demoting it.
* **`LAB-NOTES.md`** -- append-only, dated, never edited. Scar tissue is the most
  valuable thing in this repo and it was being buried under status updates that
  expire in a day.
* **`docs/session-archive.md`** -- superseded "where things stood" sections, kept
  because they carry reasoning, moved because nobody needs four of them at the top
  of the file they read first.
