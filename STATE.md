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
> 5. **Fun to play against is CANCELLED (2026-08-06), not deferred.** There is
>    one goal and it is strength. Nothing here measures enjoyment and nothing is
>    gated on it. The only survivor is a design constraint on a feature that does
>    not exist: if difficulty levels are ever built they come from genuinely
>    weaker models, never from a strong model told to blunder. Do not re-raise
>    this, and do not propose work on those grounds.

A chess engine that evaluates positions with a transformer and searches with
MCTS over a value net, with a separate policy net supplying priors. The
evaluation lineage is Ruoss et al. 2024, *Grandmaster-Level Chess Without
Search* ([arXiv:2402.04494](https://arxiv.org/abs/2402.04494)); searchless was
the starting point and is no longer the design.

This file is the operational layer: how to run things and what not to retry.
See `PHILOSOPHY.md` for why the project is shaped the way it is.

## Where things stand (2026-08-28, session 13)

**THE 19M IS TRAINED, GATED AND DEPLOYED.**
`19M-fused` finished all 1.2M steps at 00:34 PDT, exit 0. `best.pt` is step
1,185,000, selected on held-out value loss. Against the deployed nets
(`scripts/eval_heldout.py`, full set / clean subset):

    policy head   1.45227 vs 1.59138   (-0.139, 10x the floor)   clean 1.515 vs 1.660
    value head    2.08321 vs 2.06744   (+0.016, 1.2x the floor)  clean 2.156 vs 2.139
    calibration identical (brier 0.00343 both, ece 0.0014 vs 0.0013)

**The value head missed plan 3.6's 600k gate** (needed 2.053, read 2.134) and
the run continued because the criterion was never wired into anything; see
LAB-NOTES 2026-08-28. Whether the net is stronger is therefore an open question
at the clock: a marginally worse value head, a much better prior, and one
forward per node at 12.6k nps against the two nets' 9.8k.

**The gate is decided and the 19M is DEPLOYED (13:17 PDT).**
`runs/matches/fused-gate`, 19M fused vs deployed two-net, `--time 0.5`, seed
20260828, drained box, SPRT elo0=0 elo1=20: **concluded after 48 pairs / 96
games, A better, LLR +2.96 against a +2.94 bound. W25 D62 L9, 58.3%,
+58.5 +-37.0 Elo, LOS 99.9%.** Per-game search record: the fused net got
1.105 +-0.005 of the two-net engine's evaluations in the same clock. So the
win is the prior plus one forward per node, carried in spite of a value head
that is marginally worse. 62 of 96 games were threefold repetitions, the
mirror-match signature PHILOSOPHY warns about, so the +58 is a lower-confidence
number than its interval says; the Stockfish anchors are what settle it.

`scripts/promote.py runs/19M-fused/best.pt` ran: smoke passed, `runs/value.pt`
is the fused net, `runs/value.pt.previous` is the 9M, `runs/policy.pt` is left
in place and IGNORED (the engine says so at boot: `fused step=1185000
bins=64`). `sumofish-bot.service` is up and connected on it. Rollback is
`scripts/promote.py --rollback`, one file copy, the two-net engine exactly.

**Next, in order.** Plan 4.4, the continuity anchors (700n and 1600n vs
Stockfish, 400 sims, ~1.9h each, drained box), so the external ladder crosses
the model change; run them under `gpu_lock.py --drain-bot` when the bot is not
in a game worth keeping. Then watch the rapid rating against the 2626 it was
frozen at (peak 2663). Then, before any further run at this width, the
`--policy-every` sweep on a `tiny` smoke, because the value head losing the
trunk is the mechanism the held-out numbers point at.

**What is new in the tree.** `sumofish/engines/loader.py`, the one place that
builds (value, policy) from files, fused or two-net, detected from the
checkpoint width and cross-checked against the recorded `--target`.
`search_engine.py`, `match.py`, `smoke.py`, `promote.py` and `eval_heldout.py`
all go through it; three of them would otherwise have run a fused net with
priors off by 64 columns (LAB-NOTES 2026-08-28, the three failure modes).
`NeuralPolicy(offset=)` and `ValuePolicy` slicing, both on the SAME module so
`rust_mcts` still collapses to one forward; `RustMCTS.fused` reports whether
it did. `tests/verify_fused_engine.py` proves all three against a hand-sliced
reference. `eval_heldout.py` now prints the clean subset next to every number,
which was plan item 1.6's actual deliverable. `gpu_lock.py` runs a bare `.py`
under its own interpreter. `scripts/train_done_notify.py` and its timer
(desktop alarm when a training unit stops, however it stops) are committed.

## Where things stand (2026-08-24, session 12)

**THE 19M IS TRAINING.** `sumofish-train-fused.service`, started 11:15 PDT,
`19M-fused`: d=384, 8 layers, 12 heads at head_dim 32, one trunk, value logits
in `[:, :64]` and policy logits in `[:, 64:]`. **19,681,648 parameters**, 1.14x
the deployed two nets' 17,332,720 and 1.15x their speed under `compile`.
1.2M steps at effective batch 1024 (512 x accum 2, because 1024 OOMs with the
bot resident on a 16 GB card), ~3,000-5,800 positions/s alongside the live bot,
so **roughly 60-70 GPU-hours**. Watch it with
`tail -f runs/19M-fused/log.jsonl` or `journalctl --user -u sumofish-train-fused
-f`; both heads' training loss is logged per `--log-every` and both heads'
held-out loss every 5,000 steps.

**It is a COLD start, which reverses plan 3.2, and the reason is a retraction.**
The growth numbers that made a warm start look worth building were measured on
`torch.randint` token ids, which are boards that cannot exist. On 4,096 real
held-out positions:

    donor (deployed d=256 value net)   2.0601
    grown d=384, plan 3.2's scheme    10.6799   worse than random init
    COLD START d=384                   4.6914
    grown d=512 (2x, exact)            2.0601   wp delta 0.0000

Growth at 1.5x does not merely fail to preserve the donor, it starts the run
*confidently wrong*. Three repairs were measured and all helped in the
predicted direction (halve the write side so LayerNorm's mean is preserved, a
gain multiplier for the std shift, and the best of all 70 four-head duplication
picks) but the best reachable was 4.28 against a cold start's 4.69. That is
minutes of training on a 68-hour run, so it does not buy the thing growth was
for: a deployable checkpoint at step 0 and therefore a free early probe. The
early probe still happens, it is just an ordinary probe of an ordinary run.
**d=384 as the width is unaffected**; the fusion was always the purchase.
Full numbers and mechanism in LAB-NOTES 2026-08-24.

**What is new in the tree.** `PRESETS["19M"]` (the only preset whose head count
comes from `heads_for`, because head_dim must stay 32 across a growth step);
`tests/identity_grow.py`, which carries the 2x positive control that catches
this whole class of bug and asserts the step-10k recovery gate when pointed at
a run; `train.py --pos-dim` plus a refusal when an `--init-from` donor was
built against a different positional encoding, which is invisible to the
existing shape check because `pos` is a non-persistent buffer;
`sumofish-train-fused.service`. Three bugs fixed while here: `--eval-every 0`
and `--ckpt-every 0` divided by zero where every other "0 disables" flag means
never, and `scripts/train_watchdog.py` hardcoded `sumofish-lab.service` as the
thing to restart on a stall, so a stall in THIS run would have restarted a unit
that is not loaded and logged a repair that repaired nothing. It now reads the
owning unit out of the training pid's cgroup.

**Next, in order.** Watch the first held-out points: the value head should pass
the donor's 2.0674 and the policy head the deployed policy net's 1.59138, and
if either curve flattens above its incumbent the fused trunk is being fought
over, which is what `--policy-every` exists for. Then
`scripts/build_clean_indices.py` for the state-value bag, which is a hard
prerequisite for the capacity decision and not hygiene: the policy bag has
16.19% train/test overlap and contamination flatters the LARGER model on
exactly the comparison this run exists to settle. Then the step-50k probe
against the deployed engine, `--time 0.5`, 600 games, drained box. The plan's
abort criterion stands unchanged: negative beyond its interval at the probe, or
the value head not better than the deployed net's held-out by 0.014 at 600k
steps, means stop and write it up.

## Where things stand (2026-08-16, session 11 continued)

**The width decision, re-measured under `compile`, which is what ships since 08-15.**
Control 0.9979x, floor +-1.0%. Fused against today's compiled two-net engine:
**d256 1.62x, d320 1.33x, d384 1.15x, d448 0.93x, d512 0.79x.** The knee that made width
nearly free below d=448 is GONE: marginal cost per width step went from 9/10/87/68% of
what the FLOPs imply to 65/53/90/74%. CUDA graphs removed the launch overhead that was
hiding it.

d=384 still passes the plan's GO rule (1.14x parameters AND 1.15x speed, and 15% is ~3x
the acceptance instrument's +-5.3% resolution) but by a far narrower margin than the plan
believed. d=448 and d=512 are no longer free capacity: they cost 7% and 21% of the search.

**The fused net is BUILT** (`--target both`, alternating loader, per-head held-out from
step 1, engine collapses to one forward, `scripts/grow_net.py`). What is NOT decided is
the width, and it is now a genuine trade with the axes pointing opposite ways:
d=384 is +14% capacity / +15% speed but grows only APPROXIMATELY (MSE 1.03 against a cold
start's 3.44, so not playable at step 0); d=512 is +100% capacity / -21% speed but grows
EXACTLY (0.0000, playable at step 0, so the 6-GPU-hour early probe works immediately).

## Where things stand (2026-08-14, session 11)

**Read `docs/PLAN-2026-08-14.md` first.** It is the ordered plan, it came out of a
three-agent audit plus an eleven-seat Council, and it carries a status block saying what
is done. It did not exist on disk until this session: it was written into a session that
was then closed, and was recovered from the subagent transcript. Everything below is
committed; the working tree is clean.

**RUNNING RIGHT NOW, and the bot is down for it.** `sumofish-compile-gate.service`,
started 17:01, 600 games at `--time 0.5` on a drained exclusive box, ETA ~03:10.
`journalctl --user -u sumofish-compile-gate -f` to watch. It restores the bot itself in a
finally block. If it dies and needs resuming, its `config.json` records
`code: 052a1ed+ee2e4a1ab896` and `code_fingerprint()` includes the bare git SHA, so a
resume will refuse purely because HEAD has moved: check that SHA out, resume, come back.

**`compile` is DEPLOYED as of 2026-08-15 and the -168 that kept it off is retired.**
Final: **600 games, 178W 400D 22L, +92.5 +-14.6 Elo, LOS 100%**, isolated, fresh seed,
drained exclusive box. The per-game search record settles the mechanism independently:
compile ON gets **1.6260 +-0.0039** of OFF's evaluations in the same wall clock, and
`scripts/smoke.py` reproduces it on the live config at 9,827 nps against 6,073 with it
off. Its entire evidence base for being off since 2026-07-30 was two arms of 20 games
with compile bundled with dedup, one of them named "contended". The attribution is
corrected at every point of use: `PHILOSOPHY.md`, two places in this file, and the
comment in `systemd/sumofish-bot.service`. Rollback is one commented line plus a
daemon-reload and a drain.

**SUPERSEDED 2026-08-16 by the re-measurement under compile: see the block below this
one. d=384 falls from 1.64x to 1.15x, and d=448/d=512 go from wins to losses.** The
structural findings (42.7% policy share, the self-check, the method) stand.

**The cost side of the capacity decision, as measured BEFORE compile shipped (plan 1.1).** Arm C: the policy
forward is **42.7%** of per-node cost against a 25% kill threshold, self-checked by the
two same-architecture nets agreeing to 0.9%. Fused arms against today's two d=256 passes:
**d256 1.76x, d320 1.69x, d384 1.64x, d448 1.33x, d512 1.14x** faster per node, against a
+-0.5% noise floor from a same-shape control. There is a **sharp knee between d=384 and
d=448**: below it width costs 9-10% of what its FLOPs imply, above it 68-87%. d=384 is the
last rung before the cliff, which is the width the Council chose by an unrelated argument.
Caveats that bound it: the knee moves DOWN with batch size, and `compile` cuts the launch
term that makes width free, so re-read this after the gate lands.

**CORRECTED 2026-08-15**: the "refund at identical capacity" claim was wrong. Fused d=256
is 8,927,472 params against today's two nets' 17,332,720, i.e. **0.52x**, because one trunk
now serves both heads. It is 1.76x faster at HALF the parameters and should be expected to
play worse per position. The rung that is actually free is **d=384: 1.14x the parameters
AND 1.64x faster**, and it is free because of the knee. See LAB-NOTES 2026-08-15.

**Three claims died on contact with the numbers**, all recorded in LAB-NOTES: "further
training is worthless" was the cosine anneal, the calibration finding was the match
result (R^2 = 0.90 on score alone), and the abandons are not stalled searches but engines
handed the game 10s to 33 minutes late. The last one rescopes 0.9b: the mid-batch deadline
fixes the forfeit population, not the abandons, and 186s and 1995s are still unexplained.

**Next, in order.** Deploy compile when the gate lands. Then run
`scripts/build_clean_indices.py` (written, positive control passes, needs an idle box for
the 36 GB read). Then the fingerprint fix, which is deferred only because changing it is
what breaks an in-flight resume. Phase 0 leftovers: 0.6 write-once run dirs, 0.12 heads
and positional-buffer plumbing. Untested cheap idea worth an hour: overlap the two forward
passes on separate CUDA streams, which would buy part of the fusion prize with no retrain.

## Where things stand (2026-08-11, session 10)

> **The 2026-08-01 session-7 block that stood here is deleted, not demoted, per
> this file's own rule.** Two of its claims had gone false in the direction that
> matters. It said the retrained policy net was **NOT PROMOTED**; it was promoted
> at 11:5x that same morning, about twenty minutes after the block was written,
> and has been what the bot plays with ever since. It said the four `sims-*`
> rungs were **deliberately marked FAILED**; they were re-run honestly on 08-02
> and all four completed. A reader trusting it would have believed the live
> engine was a configuration that has not existed for four days, and would have
> re-earned a ladder that already exists.

> **The 2026-08-05 session-8 CURRENT block that stood here is deleted, not
> demoted, per this file's own rule.** Two of its claims had gone false. It said
> the rating was **"converged, not still climbing"** at ~2475; it is 2538 three
> days later and was still drifting up the whole time. And it said **no version
> had been cut**; v4 was cut 08-06. Its durable content (the drain that cannot
> converge, the `systemctl --user disable` symlink trap) is preserved below and
> in the sections further down.

> **CURRENT, 2026-08-11 ~18:00, session 10.**
>
> - **THE STOCKFISH RULER DOES NOT TRANSFER TO SUMOFISH, and that withdraws
>   `scale_D`, `scale_bar` and four of the five ladder absolutes.** This is the
>   result of the session and it was found by cross-checking two numbers that
>   both landed today, not by running anything new.
>
>   `sim_ladder.py` makes a rung absolute by adding a walk along the
>   Stockfish-vs-Stockfish ruler: `absolute = rung + (ruler(N) - ruler(700))`.
>   That step assumes a node budget worth X Elo to Stockfish is worth X Elo to
>   SumoFish. Two anchors of the same v6 configuration now test it directly:
>
>   | SF@700n -> SF@1600n, 1.193 doublings | Elo |
>   |---|---|
>   | measured Stockfish vs Stockfish (the ruler) | **280.8 +-16.4** |
>   | the same span measured through SumoFish | **192.8 +-18.0** |
>   | difference | **88.0 +-24.4**, z = **7.1** |
>
>   A transfer factor of **0.69**. `scale_D` was 185.2 because the ruler walk
>   contributes +1026 of it against the rungs' -286, so a ~31% overstatement of
>   the walk is a bias several times the +-12.2 it was published with. Put
>   differently: the ladder predicts the 1600-node anchor at **-248.5** and the
>   anchor measured **-162.4 +-13.4**. An 86-Elo miss on a number the ladder
>   claims to know to +-23.
>
>   **Do not quote a corrected `scale_D`.** Applying 0.69 across the whole
>   360-10700 range gives ~105, but that factor is measured at exactly one place
>   on the scale and assuming it is constant is the same species of mistake as
>   the chain itself. What is established is the direction and that 185.2 is an
>   upper bound. Reproduce with **`scripts/ruler_transfer.py`**, which refuses
>   to answer at all when fewer than two anchors share a configuration.
>
>   **The mechanism is not settled, but the two populations are not the same
>   kind of match.** SF-vs-SF ruler rungs draw **7-13%** of games and end
>   **74-87%** by arbiter adjudication; the SumoFish anchors draw **28-38%** and
>   adjudicate **30-35%**. Elo inferred from a score is draw-rate dependent, so
>   a decisive population and a drawish one are not on the same scale even when
>   both are right about who is stronger. Adjudication is proposed by the
>   PLAYERS' own eval curve and only confirmed by the arbiter, which is why two
>   Stockfish instances trip it far more readily than a SumoFish game does.
>
>   **What survives, untouched:** the anchors themselves. **SumoFish@400 sims is
>   +30.5 +-12 Elo on Stockfish@700 nodes and -162.4 +-13.4 on Stockfish@1600
>   nodes**, 2000 games each, no chain in either. That is the external Elo this
>   project has. Also untouched: every rung's own number, which is a direct
>   measurement against a pinned opponent, and the whole width-sweep and
>   held-out-loss line of argument, which never touched the ladder.
>
> - **Adjudication is NOT the mechanism, and that closes the cheap repair.**
>   The obvious suspect was the arbiter: ruler games end 74-87% by adjudication
>   against the anchors' 30-35%, and adjudication is PROPOSED by the players'
>   own eval curve, which two Stockfish instances trip far more readily than a
>   SumoFish game does. If that were inflating the ruler, the fix would be a
>   CPU-only re-run and the ladder would survive. Re-ran the 700-vs-1400 edge
>   with `--no-adjudicate`, 2400 games, same seed 99, ~11 minutes of CPU:
>
>   | 700 -> 1400, Stockfish vs Stockfish | edge | draws |
>   |---|---|---|
>   | adjudicated (the ruler) | 229.3 +-16.1 | 8.8% |
>   | played out, `--no-adjudicate` | **245.4 +-14.8** | 18.8% |
>
>   Difference **+16.1 +-21.9**, and the wrong sign: without the arbiter the
>   ruler edge is if anything LARGER. Draws double and the edge does not move.
>   So the mismatch is intrinsic to comparing a Stockfish-vs-Stockfish
>   population with a SumoFish-vs-Stockfish one, not an artefact of how games
>   are ended, and there is no CPU-only repair. Kept as
>   `runs/matches/ruler-noadj-700-vs-1400`.
>
> - **THE v6 SEARCH CONSTANTS GET BETTER WITH A BIGGER BUDGET, not worse, and
>   this was measured on 2026-08-09 and never written down.** Head to head, v6
>   (`c_puct_init=0.875`, `fpu=-0.05`) against the pre-v6 constants (1.25,
>   -0.2), same nets, same seed, 8x the budget between the arms:
>
>   | arm | W/D/L | Elo | draws |
>   |---|---|---|---|
>   | `transfer-400`, 400 sims, 600 games | 166/356/78 | **+51.3 +-17.0** | 59% |
>   | `transfer-3200`, 3200 sims, 400 games | 123/252/25 | **+86.9 +-19.6** | 63% |
>
>   Difference **+35.6 +-26.0, z = 2.68.** The control reproduced the
>   vs-Stockfish estimate (+51.3 against +63.5 +-13, a 12.2 +-21 difference), so
>   the two designs agree, and the answer arm says the gain does not decay with
>   budget. `MCTS.c_puct_at`'s docstring warns that exploration constants tuned
>   at one N need not transfer to another; here they transfer and then some.
>   Draw rates 59% and 63% are high, but the unit's own blindness threshold was
>   85% and these are well under it.
>
>   **This is the third time in two days that this file asserted something the
>   archive contradicted**, after the FPU record and the "queued and NOT started"
>   line I repeated on 08-11 without checking. `runs/matches` is the source of
>   truth and it is 95 directories deep; STATE.md is a summary of it and drifts.
>   `scripts/run_transfer_test.sh` now checks for a complete arm and reuses it
>   rather than starting one, which is what caught this, and the same check
>   belongs in front of any unit that costs GPU hours.
>
> - **THE PARITY LADDER LANDED, 05:00 on 2026-08-12, and it replaces `scale_D`
>   with a number that has no Elo in it.** All four arms ran, 11h14m, and all
>   three re-aimed rungs landed near parity (+33.1, +37.1, -25.9), so the aim
>   points were right and nothing needs replaying.
>
>   | sims | vs | rung | PARITY budget |
>   |---|---|---|---|
>   | 200 | SF@360 | -7.4 +-19 | 351 nodes |
>   | 400 | SF@845 | -24.8 +-19 | 777 |
>   | 800 | SF@1259 | **+33.1 +-18** | 1409 |
>   | 1600 | SF@1754 | **+37.1 +-17** | 1990 |
>   | 3200 | SF@3046 | **-25.9 +-19** | 2789 |
>
>   **The exchange rate, in node-doublings bought per doubling of SIMULATIONS:**
>
>   | 200->400 | 400->800 | 800->1600 | 1600->3200 |
>   |---|---|---|---|
>   | 1.15 +-0.13 | 0.86 +-0.13 | 0.50 +-0.12 | 0.49 +-0.13 |
>
>   **THE RATE IS NOT CONSTANT: it falls 0.66 +-0.18 from bottom to top.** Do
>   not quote a mean of these; a mean would hide the only thing the ladder
>   found. At the bottom a doubling of our search outruns a doubling of
>   Stockfish's nodes; by 1600 sims it buys half of one. That is diminishing
>   returns to search, measured without a chain, and it is the first version of
>   this number that the transfer failure cannot reach: both axes are BUDGETS,
>   so no cross-population Elo conversion appears anywhere in it.
>
>   The one conversion that remains is each rung's own Elo into its budget, and
>   it is small by construction (every rung inside +-40 Elo, so under 0.25
>   doublings). Swapping the near-parity slope (203.5) for the far-from-parity
>   one (173) moves the decay from 0.66 to 0.70. The conclusion does not rest on
>   the slope.
>
>   **`scale_bar` restated in the same currency: the 136M net must be worth 0.79
>   node-doublings at equal simulations** to break even on a clock (1.61 sims
>   doublings forgone at the top rate of 0.49). The Elo-denominated 321 +-54 stays
>   withdrawn.
>
> - **The transfer factor is NOT a constant, and near parity it goes away.**
>   The probe arm (`transfer-sims800-vs-sf2518`) plus every other same-config
>   pair on the fixed harness, rebuilt with a harness filter:
>
>   | sims | span | perceived/dbl | ruler/dbl | factor | worst arm |
>   |---|---|---|---|---|---|
>   | 400 | 700->845 | 203.5 +-82 | 210.3 | **0.97** | 30 Elo from parity |
>   | 800 | 1259->1970 | 212.8 +-44 | 260.2 | 0.82 | 104 |
>   | 800 | 1259->2518 | 204.2 +-27 | 262.8 | 0.78 | 171 |
>   | 1600 | 1754->4600 | 188.4 +-24 | 244.6 | 0.77 | 225 |
>   | 400 | 700->1600 | 161.7 +-15 | 235.5 | 0.69 | 162 |
>   | 3200 | 3046->10700 | 147.4 +-21 | 193.3 | 0.76 | 293 |
>   | 400 | 845->1600 | 149.3 +-25 | 242.9 | 0.61 | 162 |
>
>   Every span that reaches far from parity sits at 0.61-0.82. The one span
>   where both arms are within 30 Elo of parity is 0.97, though at +-82 on the
>   slope that single point cannot carry much. **This is why the parity design
>   works**, and it is now evidence rather than the assumption it was when the
>   run was queued.
>
>   **First pass at this table produced a 0.09 and it was my own bug**: the scan
>   grouped runs by engine config and not by HARNESS, so it differenced a
>   `ladderWARM-*` arm against a fixed-harness one. That is the "+30.5, was
>   +44.4" mistake in a new hat, six commits after the file that names it.
>   `parity_ladder._spans()` now requires each run's `code` to descend from the
>   ucinewgame fix.
>
> - **DONE, was RUNNING as of 18:40: `sumofish-parity.service`**, the ladder re-aimed so
>   every rung is played near parity and reported as an equivalent Stockfish
>   NODE BUDGET rather than a rating. Four arms, ~13 GPU-hours:
>   `parity-sims800-vs-sf1259`, `parity-sims1600-vs-sf1754`,
>   `parity-sims3200-vs-sf3046`, then `transfer-sims800-vs-sf2518`, which is not
>   a rung but a second measurement of the transfer factor one doubling above the
>   800-sim parity point. `scripts/parity_ladder.py --report` prints the table;
>   200 and 400 sims are not re-run because -7.4 and -24.8 already are parity.
>   The exchange rate it produces is **node-doublings per sims-doubling**, which
>   has no cross-population Elo in it. A rung landing outside +-60 Elo gets
>   re-aimed and replayed rather than corrected on paper.
>
> - **The FPU line at the SHIPPED `c_puct_init` is mapped, and nothing changes.**
>   Five arms, 800 games each vs Stockfish@700n, seed 4242, at
>   `c_puct_init=0.875`:
>
>   | fpu | -0.2 | -0.125 | **-0.05** | 0.0 | 0.05 |
>   |---|---|---|---|---|---|
>   | Elo | +16.5 | +26.1 | **+44.1** | +32.2 | +18.7 |
>
>   All +-19 to +-21. **No arm separates from the shipped value** (every
>   difference is inside ~+-29), and reading only that gate would end the
>   session with nothing. The five points fit an interior maximum at
>   **fpu = -0.065, 95% [-0.117, +0.040]**, with P(the curve turns) = **0.966**.
>   The shipped -0.05 is inside that interval, so **NOTHING WAS APPLIED.**
>
>   This closes the 08-09 reading that "the FPU optimum is OUTSIDE the swept
>   range". That grid stopped AT the shipped value, and the edge was the
>   artefact: given a grid that brackets it on both sides, the optimum is
>   interior and lands on the value already deployed. `FPU_ARMS` is re-centred
>   on the bracketing grid so the default reproduces the run that settled it.
>
> - **`runs/lab/tune-search.json` was destroyed and rebuilt from the per-arm
>   directories for the SECOND time in one day, by the second write path in the
>   same script.** The morning fixed `--report`, which used to end by writing.
>   The afternoon's real `--stage2-only` run then did the same damage through
>   the normal write: it replaced the 08-09 `c_puct_init` sweep with
>   `{"skipped": ...}` and dropped the FPU line measured at `c_puct_init=1.25`
>   outright. Both times "the per-arm directories survived" was the recovery,
>   which is luck twice over. `tune_search.py` now MERGES: a run may add or
>   replace the groups it measured and nothing else, a skipped stage never
>   overwrites arms that were played, and stage 2 is keyed by the
>   `c_puct_init` it was measured at (`stage2_at_ci0.875`) so two runs cannot
>   share a slot. Guarded by `tests/verify_tune_merge.py`, which is IN
>   `tests/run_all.sh`.
>
> - **The power failed at ~17:16 and nothing was lost.** Everything queued had
>   finished: six ruler rungs by 14:11, five FPU arms by 16:29. The box was idle
>   for the 45 minutes before it went down. The only casualties are two rated
>   lichess games abandoned mid-move (`WkE21HvP`, `RS4ideHj`); the bot came back
>   with the machine and is playing. There is no resume path for a `match.py`
>   run, so a longer queue would have lost an arm.
>
> **2026-08-09 ~02:00, session 9. Still current except where the block above supersedes it.**
>
> - **The 900k net is LIVE, promoted 2026-08-09 01:49, and it was promoted on
>   held-out loss rather than on the match.** `runs/value.pt` is now
>   `dd436dd8` = `runs/9M-sv-long/best.pt`, step 900,000, held-out **2.0674**
>   and puzzles **0.752** against the outgoing 595k net's 2.112 / 0.700.
>   Rollback is `scripts/promote.py --rollback`; `runs/value.pt.previous` is the
>   old `102b2f2d`. Cut as **v5**. v4 finished 65W 38D 80L over 183 games.
>
> - **THE MATCH WAS BLIND, AND THIS IS THE IMPORTANT RESULT OF THE SESSION.**
>   `lab-9m-long-vs-current`, 100 games / 50 pairs at 3.0s/move, both arms with
>   `vloss_fix` on, took 9h38m and returned **+3.5 +-26.4 Elo, 8W 85D 7L, LOS
>   60%, LLR -1.24**. Not decisive, and `decide_promote` correctly refused it.
>   The termination breakdown is why:
>
>   | reason | n |
>   |---|---|
>   | threefold_repetition | **80** |
>   | checkmate | 8 |
>   | adjudicated-arbiter | 7 |
>   | insufficient_material | 4 |
>   | stalemate | 1 |
>
>   **Eighty of a hundred games were repetition draws.** Two checkpoints of one
>   lineage evaluate balanced positions identically and shuffle. This is
>   PHILOSOPHY point 4's mirror-match blindness in its extreme form, and it is
>   NOT a property of the bot at large: rated play is 17% draws. The 07-31
>   pilot had the same signature (4/4 repetition draws) and the 595k net was
>   promoted on held-out loss for exactly this reason, with `"elo": null` in its
>   sidecar. **So this protocol has now returned null on both net swaps it has
>   ever judged.** Treat "candidate vs previous net, fixed time" as an
>   instrument that can show a net is not WORSE and cannot show that it is
>   better. Resolving a modest gain would need many hundreds of games at ~5.8
>   min each.
>
> - **Do not read the null as "more training does not pay."** It rules out a
>   large gain and cannot see a modest one. What IS established, jointly with
>   the width sweep: neither more parameters nor more steps produced anything
>   this project can currently measure at 9M. That is a statement about the
>   measuring instrument as much as about the nets, and fixing the instrument
>   (the Stockfish anchor, item 3) now gates every future net decision.
>
>
> - **RECALIBRATED 2026-08-11 on the ucinewgame-fixed harness. The absolutes
>   moved; `scale_D` is unchanged but is far too WIDE to have detected a move.**
>   Stockfish was carrying its hash between games, so at a fixed node budget it
>   was a different opponent each game and the warm harness flattered us
>   (paired, z=-2.16, ~-39 Elo, though see the anchor decomposition below, which
>   says ~-77). Ladder re-run:
>
>   | sims | rung (vs SF), MEASURED | ABSOLUTE, **WITHDRAWN** |
>   |---|---|---|
>   | 200 | -7.4 +-19 vs SF@360 | -159.3 +-23.5 |
>   | 400 | -24.8 +-19 vs SF@845 | +32.3 +-22.8 |
>   | 800 | -104.4 +-22 vs SF@1970 | +256.8 +-28.1 |
>   | 1600 | -225.0 +-28 vs SF@4600 | +431.5 +-37.4 |
>   | 3200 | -293.0 +-32 vs SF@10700 | +581.4 +-42.9 |
>
>   **The absolute column was withdrawn on the evening of 2026-08-11, hours
>   after those intervals were earned.** It chains onto a ruler that does not
>   transfer to SumoFish; see the CURRENT block at the top of this file. The
>   rung column is a set of direct measurements against a pinned external
>   opponent and is unaffected. The absolutes shown are the 2400-game-ruler
>   versions, which is what they were when they were withdrawn; the numbers
>   this table carried before that (-152.6 +-57, +10.1 +-50, +278.6 +-77,
>   +477.4 +-108, +642.4 +-121, off a 200-game ruler) are deleted rather than
>   kept beside them, because two withdrawn columns are not more informative
>   than one.
>
>   **THOSE INTERVALS DID NOT EXIST UNTIL 2026-08-11 and they change what this
>   table can be used for.** Each absolute is a rung plus a walk along a chain of
>   six Stockfish-vs-Stockfish ruler rungs, each +-41 to +-74, and `sim_ladder.py`
>   propagated none of it: it published five numbers to 0.1 Elo with no interval
>   at all. The file's own docstring opens by indicting the withdrawn design
>   because "error accumulates down a chain". The chain was moved from the
>   SumoFish axis to the Stockfish axis, not removed. Now propagated properly,
>   in an edge basis, so shared chain segments cancel in a difference rather
>   than being double-counted.
>
>   **`scale_D` = 199 +-33 Elo per doubling of SEARCH** (was 195, itself +-33),
>   **and this is WITHDRAWN as of the same evening** -- it went to 185.2 +-12.2
>   on the 2400-game ruler and then out entirely, because the +-12.2 is an
>   interval on a number carrying a bias several times its size.
>   Of that +-33.5, the ruler contributes **+-32.1** and the two rungs only
>   +-9.4. **So "scale_D survived the harness fix" was never a testable claim:**
>   the difference of two +-33 numbers carries +-47, and the test could not have
>   seen a change smaller than about a quarter of the value. The CONCLUSION is
>   still right (quote absolutes only from the fixed harness) but that particular
>   argument for it is not evidence. Same correction one level down: the four
>   increments are +-77 / +-85 / +-61 / +-61, not the "~+-30 on each difference"
>   claimed further down this file.
>
>   **The cheapest measurement available to this project is the ruler, and it
>   needs no GPU.** It is Stockfish against itself: all six rungs total 550
>   seconds of logged game time. Taking them from 200 to ~2,400 games each is
>   ~1.8 h of CPU and drops the ruler term from +-32.1 to +-9.4, i.e. `scale_D`
>   to **+-13.6**. **DONE the same afternoon, 12:36-14:11, and it landed at
>   +-12.2.** Every one of the six edges came back inside its old interval
>   (largest move -37.1 against +-74.1), so this was a narrowing and not a
>   correction. The 200-game originals are kept as `rulerN200-*`, and
>   `ladderWARM-*` / `anchorWARM-*` are kept for audit.
>
>   **And it narrowed the wrong term.** The ruler's error was the biggest
>   number in the interval, so shrinking it was the right move on the evidence
>   available; what it could not do is test whether the ruler MEANS anything to
>   SumoFish, which is what the second anchor then showed it does not. A tighter
>   interval on a biased number is a more confident wrong answer.
>
>   **Anchor, re-run:** SF@700 rung is **+30.5 +-12**, SF@1600 rung is
>   **-162.4 +-13.4** (landed 12:22). Those two are the load-bearing
>   measurements in this project now.
>
>   **It is NOT "+30.5, was +44.4", and that framing stood here until
>   2026-08-11.** The two runs differ in two things, not one: `anchorWARM-700nodes`
>   is v5 (`c_puct_init: null`, i.e. the hardcoded 1.25, and `fpu -0.2`) on the
>   warm harness, `stockfish-anchor-700nodes` is v6 (0.875/-0.05) on the fixed
>   one. Same 2000 games, same seed 7, which is precisely why it read like a
>   controlled re-run. The -13.9 is two effects of opposite sign, and they
>   decompose exactly against the one same-config pair that exists
>   (`confirm-combined`, v6 warm, +107.9 +-12.6):
>
>   | term | value |
>   |---|---|
>   | harness fix on v6 at 700n | **-77.4 +-17.4** |
>   | v6 tuning gain | **+63.5** |
>   | net | -13.9 |
>
>   **The harness fix cost ~-77 Elo here, not -14**, and the tuning gain hid it.
>   Compare LAB-NOTES' -39 from the paired 845-node analysis: the two differ by
>   38 +-39. Not conclusive, but do not quote -39 as settled.
>
>   Two independent routes to "v6 at 400 sims vs SF@700" agree: **+10.1 +-50**
>   via the ladder rung plus the ruler edge, and **+30.5 +-12** direct from the
>   anchor. That is a **z = 0.78** difference. It was recorded here as
>   "0.40-sigma", which divided by a 95% half-width and not a sigma; see the
>   2026-08-11 LAB-NOTES entry on that convention.
>
> - **The old warm-harness ladder block, superseded:** Five rungs on the v6 engine, each an
>   independent measurement against Stockfish at a pinned budget, priced through
>   the measured 350-11,200 node ruler:
>
>   | sims | vs | W/D/L | rung | ABSOLUTE (vs SF@700) |
>   |---|---|---|---|---|
>   | 200 | SF@360 | 311/279/210 | +44.1 +-19 | -105.1 |
>   | 400 | SF@845 | 313/263/224 | +38.8 +-19 | **+84.2** |
>   | 800 | SF@1970 | 135/241/224 | -51.9 +-22 | +293.5 |
>   | 1600 | SF@4600 | 45/191/264 | -163.2 +-25 | +499.2 |
>   | 3200 | SF@10700 | 17/122/261 | -246.3 +-32 | +673.7 |
>
>   Increments 189.3 / 209.3 / 205.7 / 174.5, mean **194.7**, spread 35 against
>   ~+-30 on each difference: flat, so quote one number and not a curve. This
>   supersedes the WITHDRAWN `sims-*` chain and is better than it in kind, not
>   just in error: no rung depends on any other, and the scale has an origin, so
>   it answers "how strong IS it" and not only "how much did that buy".
>
>   **Do not re-run the old `sims-400-200` chain. It is superseded, not
>   pending.** `scale_D = 195` and the search-cost side (1.61 doublings for a
>   136M, m=3.06) can now be argued in the same currency.
>
> - **A third, independent read on the v6 tuning.** The 400-sim rung puts v6 at
>   **+84.2** vs SF@700 where the anchor put v5 at +44.4, i.e. **+39.8**. The
>   other two reads were +63.5 (vs Stockfish) and +51.3 (head to head). All
>   three positive, all overlapping once the ruler's own +-46 is propagated.
>   Three designs, one conclusion.
>
> - **THE ANCHOR LANDED. THIS IS THE FIRST EXTERNAL ELO THIS PROJECT HAS EVER
>   HAD.** v5 against Stockfish at pinned nodes, 2000 games a rung, 3h44m total:
>
>   | opponent | W/D/L | score | **Elo** | draws |
>   |---|---|---|---|---|
>   | Stockfish@700n | 770/714/516 | 56.4% | **+44.4 +-12** | 35.7% |
>   | Stockfish@1600n | 269/644/1087 | 29.5% | **-150.9 +-13** | 32.2% |
>
>   **Read the intervals: +-12 and +-13.** The mirror match spent 9h38m to earn
>   +-26.4. This spent 1h52m a rung to earn +-12. That is the same GPU buying an
>   order of magnitude more information, and the mechanism is in the last
>   column: 33-36% draws against the mirror match's 85%.
>
>   Interpolating between the two rungs: 195.3 Elo across 1.193 doublings of
>   Stockfish's budget = **164 Elo per doubling**, putting v5 at parity with
>   **Stockfish at ~845 nodes** (at our 400 sims). Treat 845 as an
>   interpolation across a wide gap, not a measurement; the two measured points
>   are the results. Note the 24-game calibration put 700n at exactly 50.0% and
>   2000 games put it at 56.4% -- both inside the n=24 interval of +-125, which
>   is a reminder of what a 24-game read is worth.
>
> - **THE TUNING SWEEP IS DONE: no value changed, and two real findings.**
>   Nine arms, 800 games each vs Stockfish@700n, same seed. Nothing separated on
>   its own pairwise test, and reading only that gate would have discarded both:
>
>   | `c_puct_init` | 0.5 | 0.875 | **1.25** | 1.75 | 2.5 |
>   |---|---|---|---|---|---|
>   | Elo | +43.2 | +58.3 | **+42.3** | -1.3 | -56.5 |
>
>   | `fpu` | -0.5 | -0.35 | **-0.2** | -0.05 |
>   |---|---|---|---|---|
>   | Elo | +13.0 | +23.5 | **+42.3** | +55.6 |
>
>   All +-19. **`c_puct_init` sits at the top edge of a cliff**: flat below,
>   then -43.6 at 1.75 and -98.8 at 2.5 against shipped, both far outside the
>   ~27 difference error. Upward drift in exploration is dangerous; downward is
>   free. **The FPU optimum is OUTSIDE the swept range**: monotone across all
>   four points, ~+14 a step, best at the LAST value tested. A monotone
>   four-point trend beats the pairwise test that rejects each step, and it says
>   the range was bounded wrong, not that the knob is inert.
>
>   `sumofish-tune-confirm.service` is running the two follow-ups at 2000 games
>   (+-12): `confirm-fpu0.1` pushes FPU past the edge at the shipped
>   c_puct_init, and `confirm-combined` measures (0.875, -0.05) AS A
>   CONFIGURATION, because stage 2 swept FPU at 1.25 and not at 0.875, so
>   "best + best" is an untested product of two marginals and both knobs move
>   exploration. Baseline is shipped at ~+43, which has two independent
>   measurements (anchor +44.4 +-12 on default seed, sweep +42.3 +-19 on seed
>   4242). Difference error ~+-17.
>
> - **Two free validations of the match harness.** `c_puct_init=1.25` scored
>   +42.3 +-19 in the sweep; the anchor put that same configuration at
>   +44.4 +-12 in a separate 2000-game run on different openings. And stage 2's
>   `fpu=-0.2` arm is stage 1's `c_puct_init=1.25` arm under another name: it
>   returned byte-identical 304/289/207. Same seed and config reproduce exactly
>   at the match level, whatever the training pipeline does.
>
> - **THE INSTRUMENT PROBLEM IS BEING FIXED, and one bug fell out of it.**
>   `sumofish-anchor.service` is running the first external anchor: v5 against
>   Stockfish at pinned nodes, 2000 games per rung. **The old 40/100-node
>   budgets were retired** -- calibrated 07-30 against a weaker net with the
>   vloss defect live, today's net went W3 D1 L0 at 40 nodes. Recalibrated:
>
>   | Stockfish | 100n | 400n | **700n** | 1600n |
>   |---|---|---|---|---|
>   | SumoFish scores | 68.8% | 68.8% | **50.0%** | 28.1% |
>
>   700n is dead even (8W 8D 8L over 24 games). Note an n=8 read of that same
>   point said 31%; do not calibrate off n=8. **Draws are 33-37% here against
>   the mirror match's 85%**, and games cost ~3.2s against the mirror match's
>   374s. That is the argument for an external opponent in two numbers.
>
> - **`--cpuct` HAS NEVER DONE ANYTHING, and nothing could set the knob that
>   does.** A five-arm c_puct sweep (1.0 to 4.5) returned byte-identical move
>   hashes. Under AlphaZero's schedule -- the default, and what ships --
>   `c_puct_at()` returns `ln((1+N+base)/base) + c_puct_init` and reads
>   `self.c_puct` only when `c_puct_base is None`, i.e. only under
>   `--fixed-cpuct`. And `match.py` had **zero references to `c_puct_init`**, so
>   every match this project has ever run used the hardcoded 1.25: the ladder
>   rungs, the promotion gates, all of it. `config.json` recorded the requested
>   c_puct throughout, so the archive shows five sweeps that did not happen.
>   **Provenance that records the request rather than the effect cannot catch
>   this.** Fixed: `--cpuct-init` plus per-side overrides threaded to both
>   engines, guarded by `tests/verify_cpuct_binding.py`, written up in
>   LAB-NOTES 2026-08-09.
>
> - **`sumofish-tune.service` is queued behind the anchor** and self-sequences
>   (it polls for the anchor to go inactive; `After=` does not wait for a unit
>   that is already running). Sweeps `c_puct_init` then FPU, 800 games/arm
>   against the 700n parity point, same seed for every arm so they see identical
>   openings. It applies NOTHING; it writes `runs/lab/tune-search.json`.
>
> - **`/home/nomad/dev/active/chess-gpu` is deleted.** It held three
>   `.snap.new` files, all byte-identical to the accepted snapshots in this
>   repo, and was the target of `run_stockfish_anchor.sh`'s stale `cd`.

> - **The rating climb is the OPPONENT POOL, not the engine.** 2538 rapid at
>   n=507, RD 45. Nothing in the engine has changed since 08-02 (both net hashes
>   unchanged), yet it went 2484 -> 2538 over 147 games **while scoring 0.432**.
>   Measured over all 636 games in `logs/games`, the mean opponent rating has
>   climbed the whole time:
>
>   | period | n | mean opp | score |
>   |---|---|---|---|
>   | to 07-30 | 119 | 2212 | 0.387 |
>   | 07-31 to 08-02 | 123 | 2385 | 0.671 |
>   | 08-03 to 08-05 | 154 | 2560 | 0.461 |
>   | 08-06 to now | 154 | 2592 | 0.438 |
>
>   A 0.438 score against a 2592 mean implies a true rating near **2549**, so
>   ~11 points of catch-up remain and it is nearly converged. **Never read a
>   rating move against this pool without the pool mean beside it**; gaining
>   points while losing more games than you win is the expected behaviour here,
>   not an anomaly, and the largest single sample move in the whole climb was +9.
>
> - **Three job units are now DISABLED** (`train-continue`, `train-9m-long`,
>   `sweep-136m`). All three pointed at finished work and would have started a
>   job on boot. Note for anyone re-reading the symlink trap below: **these three
>   are real files in `~/.config/systemd/user/`, not symlinks into `systemd/`**,
>   so `disable` removed only the `default.target.wants/` link and left the unit
>   intact. The trap is real but applies to the eight units that ARE repo
>   symlinks, not to one-off job units.
>
> - **PGN capture is COMPLETE and the per-game files are a red herring.**
>   `pgn_file_grouping: "all"` puts everything in `logs/games/SumoFish games.pgn`
>   (636 games). The 36 loose per-game files predate that setting. Six of them
>   duplicated games already inside the combined file, and `scripts/games.py`
>   globs `*.pgn` and reads every game in every file, so it **double-counted
>   those six**; they are moved to `logs/games-superseded/`. One duplicate
>   survives inside the combined file itself (`MCA4fslN`, written twice by
>   lichess-bot), left alone because the bot holds that file open for appending.
>   Counting games from that directory needs dedup by `Site`, not a file count.
>
> - **ONE thing that looks broken and is not**, and one that looked benign and
>   is not. `sumofish-train-watchdog` firing every 5 minutes with no training
>   running logs `no live training process; nothing to watch` and exits 0. That
>   one is fine.
>
>   **The `ReadTimeoutError` block is NOT fine, and the "do not re-investigate"
>   ruling that stood here until 2026-08-14 was a load-bearing lie.** The
>   observation was right: the stream does recover, every time. The inference
>   was wrong: *the games do not*. Deduped across `logs/games/` and
>   `logs/games-superseded/` on the `Site` URL, SumoFish has lost **10 rated
>   games on time at 900+10** and separately **abandoned 23 more by accepting a
>   challenge and never making its first move**. The abandons carry
>   `Result "*"`, so they cost no rating and are invisible to `rating.jsonl`,
>   which is why nobody counted them. 33 events at the deployed time control.
>
>   Mechanism, in source: `rust/src/tree.rs:658-659` checks the deadline
>   **between batches, not mid-batch**, so a stalled GPU callback means the
>   check never runs. Every 08-07 and 08-08 forfeit falls inside a concurrent
>   lab job. One game shows a 290-second move against a ~30-second budget, then
>   a terminal stall while still holding 384 seconds.
>
>   The rating cost is small and honest: **~+7.2 Elo**, because 6 of the 10
>   forfeits are against 2759-2981 opponents where the expected score is
>   0.10-0.25 and losing is nearly free. That is half the +-14.8 resolution of
>   the only instrument that can see it. **Fix it because it is cheap and
>   correct, and never claim the win afterwards.**
>
>   The general rule this earns: **a file with a demonstrated drift rate of
>   five stale claims in four days must not be allowed to CLOSE an
>   investigation.** "Do not re-investigate" rulings belong in `LAB-NOTES.md`,
>   where they are dated and read as "on 2026-08-08 this looked benign" rather
>   than as a standing verdict.

> **THE WIDTH SWEEP IS COMPLETE, all three arms.** Matched tokens (20k steps x
> 1024 effective batch = 20.5M positions), matched seed, matched data order;
> width is the only variable.
>
> | arm | params | held-out | puzzles | state |
> |---|---|---|---|---|
> | `sweep-tiny` | 0.3M | 2.9690 | 0.300 | done, 20k steps |
> | `sweep-9M` | 8.9M | 2.5516 | 0.361 | done, 20k steps |
> | `sweep-136M` | 134.3M | 2.7678 | 0.314 | done, 20k steps, 7.1h |
>
> **The verdict is FLAT: width bought nothing at matched tokens.** Held-out loss
> changes **+0.0549 per doubling** of parameters between 9M and 136M, i.e. the
> wrong direction, and the 136M arm is worse than the 9M it is supposed to beat.
> That is the cost-free half of the trade only. The search cost of the same step
> is a measured **1.61 doublings** (m=3.06, `runs/lab/profile-2026-07-31.json`),
> and the Elo value of a doubling is priced by the ladder below. This licenses an
> ordering and never an Elo claim.
>
> The earlier `2.8646 @ 5k` reading for this arm is gone and should not be
> quoted: it was a fifth of the way through a warmup-and-cosine schedule the
> other two arms completed, so it measured how far each run got. The arm was
> re-run from scratch rather than resumed, because PHILOSOPHY's first line says
> time is not a constraint and the arm deciding whether 136M is worth 35 GPU-hours
> gets the same protocol as the arms it is compared against.
>
> **That re-run is an accidental replicate, and the pipeline is not bit-
> reproducible.** Same arm, same `--seed 1234`, same data order, run twice: step
> 500 came out at loss **3.73954** on 08-02 and **3.76445** on 08-05, grad_norm
> 12.7 against 28.2. Not investigated (`--compile 1` autotunes against whatever
> else is on the card, and bf16 reductions are not associative) but worth keeping,
> because it is the only measurement this repository has of its own run-to-run
> noise. Calibration: **~0.025 in training loss at step 500**, against a
> tiny-to-9M held-out gap of 0.42. The sweep's ordering is two orders above its
> noise floor; a future experiment resting on a difference of 0.02 is not.
>
> **`runs/lab/state.json` had `current: sweep-136m` with a dead pid** for three
> days after a reboot. The lab does not notice a runner that died with the
> machine; check `ps` against that pid before believing the queue is working.
>
> **A reboot mid-arm still costs the whole arm.** `sweep_argv` sets
> `--ckpt-every 20000` and `--auto-resume` reads only `latest.pt`, so there is
> nothing on disk to resume from until step 20,000. Lowering it was considered
> and rejected: the sweep's whole claim is that the arms differ in width and
> nothing else, and the other two ran with this value.
> **The 136M question already has a bar to clear, and it is high.** The exchange
> ladder was re-earned honestly on 08-02, all four rungs, 300 games each:
>
> | rung | Elo | LOS |
> |---|---|---|
> | 200 -> 400 sims | +29.0 +-25.1 | 99% |
> | 400 -> 800 | +240.8 +-36.3 | 100% |
> | 800 -> 1600 | +308.2 +-43.8 | 100% |
> | 1600 -> 3200 | +233.7 +-35.6 | 100% |
>
> So a doubling of search is worth **234 Elo** at the top of the measured ladder,
> and `forward-bench` prices a 136M forward pass at **3.06x** the 9M's inside the
> search loop. A 136M net therefore has to be **+377 Elo at equal simulations**
> just to break even on a clock. The retraction in the README stands for the
> *old* ladder; this one is not that one, and `runs/lab/state.json` holds its
> numbers under `facts`.
>
> The unwelcome part of the first rung: 200 -> 400 is worth only +29, while every
> doubling above it is worth 8-10x that. **Investigated 2026-08-05, and it is
> still unexplained -- but two explanations are now dead rather than untried.**
>
> It is not a difference between the matches: all four rung `config.json`s are
> identical apart from the two sim counts. It is not prior-lock either, which was
> the obvious candidate (at 200 sims the search cannot outvote the policy, both
> arms play the prior's move, identical moves draw).
> `scripts/prior_dominance.py` measures how often doubling the search changes the
> move played, over 60 real middlegame positions: **18.3%, 20.0%, 13.3%, 13.3%**
> across the four rungs. The bottom rung changes its move MORE often than the top
> one. The moves change; they do not help.
>
> Whether the changed moves are BETTER is the surviving question and
> `scripts/rung_quality.py` cannot currently answer it. Scoring every rung's move
> against Stockfish at 1M fixed nodes gives a median loss of 5.5-6.5 cp for every
> rung **and for the bare prior with no search**, because positions sampled
> uniformly from real games are mostly positions where the move is obvious. The
> means separate, but positions losing >100 cp number 8/7/4/6/4 out of 60, so the
> ordering is two to four positions of noise and it duly came out contradicting
> the ladder. Both scripts and the full argument are in LAB-NOTES, dated today.
>
> **THE LADDER WAS MEASURED ON AN ENGINE THAT HAS NOT BEEN DEPLOYED SINCE
> 2026-07-30.** Every rung's `config.json` says `vloss_fix: false`, and the live
> bot has run `CHESSGPU_VLOSS_FIX=1` since 07-30, where it earned its default on
> an SPRT verdict of W25 D7 L0 and a point estimate of **+364 Elo at 400 sims**.
> The flags are `--a-vloss-fix`/`--b-vloss-fix`, `store_true`, defaulting to off,
> and `lab.py`'s `match_argv` never passes them. So does every other lab match.
>
> The rungs are still internally valid -- both arms had the defect, so each rung
> is a fair comparison *of the unfixed engine*. What does not follow is the use
> the numbers are put to: `D = 233.7` and the `scale_bar = 377 Elo` that prices
> the 136M decision are about the engine that plays, and that engine has the fix.
>
> **And the defect is now the leading explanation for the flat first rung.**
> With `vloss_fix` off, virtual loss is added straight into `value_sum` rather
> than to a separate in-flight counter, so it corrupts Q and not just the PUCT
> denominator (`rust/src/tree.rs`, defect 2 of 3). At batch 64 there are up to 64
> fake values in the tree at once: against a 200-simulation tree that is a third
> of the whole search corrupted at any instant, against 3200 it is 2%. The damage
> is therefore worst exactly where the ladder is flat, and the fix's own +364
> verdict was measured at 400 sims, in that same crippled regime. Doubling from a
> crippled base buys little; doubling from a healthy one buys the ~240 the upper
> rungs show.
>
> **CONFIRMED, and it is the whole anomaly.** The same rung re-screened with the
> fix ON for both arms: **+214.8 +-69.8, W60 D35 L5 over 100 games, LOS 100%**
> (`runs/matches/sims-400-vs-200-vlossfix`), against **+29.0 +-25.1** with it
> off. The intervals do not overlap and the rung lands in line with the other
> three. 100 games is a screen and not a rung -- the ladder's are 300 -- so treat
> the point estimate as indicative and the direction as settled.
>
> **Consequences, all acted on:**
>
> - `match_argv` now passes `--a-vloss-fix --b-vloss-fix`, so every future lab
>   match measures the engine that plays. Deliberately set there and NOT in
>   `match.py`, whose defaults must stay off: `tests/identity_*.py` need all
>   three defects off for the Rust/Python identity to hold. The identity test and
>   the strength test want opposite defaults.
> - **All four rungs and the `scale` decision are WITHDRAWN**, and `scale_D` /
>   `scale_bar` / `scale_why` are withdrawn out of `facts` so the 377 Elo bar is
>   not readable as live at the one place it gets used. `scale_m` (3.06, the
>   forward-pass cost ratio) survives: it never depended on the ladder.
> - Re-earning the ladder is ~10 GPU-h and **nothing starts it automatically**.
>   The withdrawn jobs are only eligible; `sumofish-sweep-136m.service` is
>   `--only`, so tonight's run cannot walk into them.
> - The withdrawal lives in **`runs/lab/withdrawn.json`**, not in `state.json`,
>   and `load_state()` overlays it. That is not tidiness: `run()` reads the state
>   once at startup and writes that snapshot back at every job boundary, so a
>   withdrawal edited into `state.json` during tonight's 8-hour job would have
>   been silently reverted at ~01:45 by a process holding a copy from 17:44. A
>   retraction has to survive a runner that disagrees with it.
>
> **The other candidate, no longer needed but not disproven: tree reuse.** Every
> rung ran with `reuse: true`, which subsidises a 200-sim arm proportionally more
> than a 3200-sim one. It predicts the same shape and the vloss fix has now
> explained that shape, so it is not worth chasing on its own evidence -- but the
> re-earned ladder will still carry it, on both arms, as the bot does.

> **The lichess rating is the only external measurement running.** No match is
> in flight and the exchange ladder is withdrawn, so the bot's rapid rating is
> currently the whole of this project's contact with opposition it did not
> build itself.


**Standing rule, promoted out of the deleted block because it is not status:**
> if you run two Claude sessions at once, say so up front. A second session
> starting the bot during a wall-clock match silently invalidates the match, and
> starting it during training halves the throughput of both. This has happened
> once already (an unexplained bot start at 19:42:31 on 07-29, from outside the
> repo -- the watchdog declined and nothing in-tree does it).

**The engine is Rust, provably identically, and 3.6x faster. SUPERSEDED on the
two extra speed flags: `compile` was isolated 2026-08-15 at +92.5 +-14.6 Elo
over 600 games and is now DEPLOYED; the -168 was dedup's and the two were never
measured apart.**

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

**2. SUPERSEDED 2026-08-15. `compile` is ON and deployed; `dedup` is still off
and is a live question.** What stood here was "both are OFF and that is a
measured decision, not caution", resting on **-168 Elo from 20 games (W0 D11 L9,
LOS 0.0%) with the two flags BUNDLED**. Isolated over 600 games on a drained
exclusive box, `compile` alone is **+92.5 +-14.6, LOS 100%**, and it takes
1.6260 +-0.0039 of plain's evaluations in the same wall clock. The -168
diagnosis below is sound and it belongs to **dedup**, which collapses descents
and can therefore buy claimed search that is not search; compile sends the same
rows and cannot. `dedup` measured +16.2 +-14.7 on its own at 0.5s (LOS 0.985),
on a CONTENDED box, so it is unresolved rather than refuted. The original
diagnostic, which is still worth reading: The diagnostic: at 0.5s the fast arm ran 7,297 nominal
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

**That consequence is LIFTED as of 2026-08-02, and only for the new ladder.**
"No proposal may be justified by an Elo-per-doubling figure" stood while the
only figure available came from the replayed rungs. The ladder has since been
re-earned on the fixed harness, four rungs, 300 games each, and its numbers are
in the status block above. What remains retracted is the OLD ladder and
everything derived from it, including the "+237 Elo per doubling" and the "+50
per doubling" the README withdraws; do not quote a number from before 08-02
because a similar one now exists.

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

That job (`train-136m`) is DELETED from the plan and the width question is now
asked by the three-arm sweep instead, at 20k matched steps rather than 400k
unmatched ones. The sweep is a cheaper and better-controlled instrument for the
same question: it prices d(held-out loss)/d(log2 params) instead of producing
one net whose one number has nothing to sit beside.

**6. CUDA streams are not the shortcut.** Two streams alone is 1.04x; 1.19x on
top of `compile`. Halving the two forward passes honestly costs the ~40
GPU-hours a shared-trunk two-head net needs to train. At 89% network share that
is now the single largest speed item, which is a reversal: it was item 4 when
the network was 9%.

## Next session, in order

> **SUPERSEDED 2026-08-14 by `docs/PLAN-2026-08-14.md`, which is the ordered
> plan. Read that first.** It comes out of a three-agent audit plus an
> eleven-seat Council and it prices the whole programme; this section is kept
> because the ladder reasoning below is still correct and still the right
> design, but the ORDER here is stale. Where the two disagree, the plan wins.
>
> The plan's short version: measure `compile` before buying anything (it has
> never once been isolated, its entire case against is 40 games with two flags
> bundled), buy the FUSION of the two nets rather than width, and if the
> arithmetic still holds after the bench, ship one shared-trunk two-head net at
> d=384. Phase 0's repairs are done and in the tree. Three claims died on
> contact with the numbers on the day it was written, all three the same
> species of error, and all three are recorded in LAB-NOTES: the width sweep
> measured the gradient clip, `9M-sv-long`'s terminal flatness measured the
> cosine anneal, and the calibration finding measured the match result. Before
> a number becomes a gate, regress it on the thing you already know moves it.

**NOW, and it blocks every Elo claim: rebuild the ladder without the chain.**
The absolutes are withdrawn because they walk a ruler SumoFish does not
experience (see the CURRENT block). Two designs, and the second is cheaper:

  a. **Anchor every rung against ONE Stockfish budget.** Chain-free by
     construction, and impossible: SumoFish@3200 sims against SF@700n scores
     ~99%, which measures nothing. An absolute scale over a 16x range needs a
     chain somewhere. That is the actual problem, not a bug in the ruler.

  b. **Re-aim each rung at its own parity, and quote node budgets instead of
     Elo.** A rung played near parity needs only a tiny local correction, and
     the local corrections are exactly where the two routes already AGREE
     (+30.5 vs +32.3 over 0.27 doublings). The ladder then reports "800 sims is
     worth Stockfish at N nodes", which is a measured equivalence with no
     cross-population Elo in it, and the exchange rate becomes node-doublings
     per sims-doubling: dimensionless, and the thing `scale_D` was for.
     Present budgets miss parity badly (the 1600-sim rung is -225 Elo against
     SF@4600); the implied parity points are ~1250, ~1750 and ~3050 nodes for
     800, 1600 and 3200 sims. Re-running those three rungs at ~800 games is
     roughly 12 GPU-hours and produces the first ladder this project has that
     does not assume transitivity.

Do (b). Do NOT publish `scale_D` x 0.69 in the meantime; see LAB-NOTES.

**DONE, and it was done on 2026-08-09: `sumofish-tune-transfer.service`.** The
claim that stood here on 08-11, that the unit had never been started, was FALSE
and I repeated it out loud before checking `runs/matches`. Both arms had run
three days earlier and the answer was on disk the whole time. See the CURRENT
block.


**0. DONE, and it changed what matters. `match-9m-long`** ran 100 games and
returned +3.5 +-26.4 with 80 repetition draws; the net was promoted on held-out
loss and cut as v5. **The instrument, not the net, is now the bottleneck**, so
item 3 below has been promoted to the top of this list: until there is an
external anchor, no net decision this project makes can be measured, and the
next one will fail exactly the same way.

**Done: the Rust core is deployed to rated play, and now the default.**
Live with `CHESSGPU_CORE=rust` since 2026-07-30 14:32, both speed flags off.
As of 2026-07-31, `select_mcts_class()`'s default flipped too: an unset
`CHESSGPU_CORE` now selects Rust, not Python -- `CHESSGPU_CORE=python` is the
rollback, same as `=rust` used to be the opt-in. `sumofish.mcts` (the
pre-port Python search) was NOT retired: it's still exactly where it was and
still the oracle `tests/identity_*.py` and `scripts/match.py` compare Rust
against, it's just no longer what an
absent env var silently falls back to. Verified with a new
`tests/verify_core_default.py`; existing `tests/verify_rust_flag_guard.py`
is unaffected (it exercises `ignored_rust_flags` on an explicit dict, not
`select_mcts_class`'s default).

**1. CLOSED 2026-08-11, and NOT by doing it. The relative ladder is
SUPERSEDED, not pending.** This item said to re-earn four `sims-*` rungs with
the virtual-loss fix on, and gave the command to reset them. That contradicted
the recalibration block at the top of this file, which says in as many words
"Do not re-run the old `sims-400-200` chain. It is superseded, not pending."
Both statements stood in this file at once for two days, one of them naming the
exact command to burn ~10 GPU-hours on work the other calls unnecessary.

The resolution: the **absolute** ladder (five rungs, 200-3200 sims, each an
independent measurement against Stockfish at a pinned budget) answers the same
question and does not depend on those four rungs at all. `scale_D` is now
**199 +-33** from it, and `scale_bar` is **321 +-54** for the 136M break-even,
replacing the withdrawn 377.

**`scale_D` and `scale_bar` are therefore REINSTATED at the point of use**, in
`runs/lab/state.json` `facts`, with the old values kept as
`scale_D_superseded` / `scale_bar_superseded` on the same pattern as
`scale_m_superseded`. `runs/lab/withdrawn.json` no longer withdraws them, so
the lab board and this file finally agree. The four `sims-*` JOBS stay
withdrawn, because what was wrong with them (measured on an engine that has not
been deployed since 07-30) is still true.

**Quote `scale_D` only with its interval.** +-33 is wide, the ruler chain is
+-32 of it, and the fix for that is CPU-only: see the recalibration block.

**2. DONE, and it recurs. Cut v4** was done 2026-08-06 (`VERSIONS.jsonl`, both
nets plus the harder opponent pool). The standing item this leaves behind:
**a promotion is not a diff, so nothing catches it.** The pre-push hook triggers
on changes to `sumofish/` and `config/lichess-bot.yml`, and `scripts/promote.py`
touches neither, so the next promotion silently re-creates exactly the scoping
problem v4 was cut to fix. If `match-9m-long` promotes, cut v5 in the same
sitting, before reading any record scoped to a version.

**3. Stockfish as an absolute anchor (NOW THE TOP ITEM, see 0)**, at pinned nodes. Every match here is
relative, so the whole ladder floats -- and now that the exchange ladder has
been re-earned, this is the only remaining thing that would tie the numbers to
something outside the project. The adjudicator half is done.

**4. Tune `c_puct` / FPU / temperature.** This was gated behind the policy
retrain, which is now done and deployed, so the gate is open. The right values
depend on the prior's sharpness and the prior changed on 08-01, which means any
value in the code today was settled against a net the engine no longer uses.

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
and it was never a blocking dependency for the 3 MCTS defect fixes (leaf
dedup, virtual loss, mate distance), and none of them needed it: vloss_fix
shipped 2026-07-30, dedup lost in 2026-07-29, and mate_distance is being priced
on the unbatched harness on 2026-08-13 for about five GPU-hours. See "Open, smaller" below for
where each landed. Production integration (adjudication, PGN/games.jsonl
logging, SPRT stopping across concurrent games) is real additional work the
sizing script does not do.

**7. DONE, and the answer was no.** The width sweep ran all three arms at
matched tokens and held-out loss moved **+0.0549 per doubling** between 9M and
136M, i.e. the wrong way: width bought nothing, and that is before the 1.61
doublings of search the bigger net costs inside the loop. "Scale it up" is now
an argument that has been made and lost, not a preference. See the sweep table
above; the surviving scaling question is depth-of-training, which is what
`train-9m-long` tested and `match-9m-long` will price.
**Now IN THE PLAN, 2026-08-01**: `sweep-tiny` / `sweep-9m` / `sweep-136m` +
`scaling-curve` in `scripts/lab.py`, three arms at 20k steps and batch 1024
(20.5M positions each) sharing `--seed 1234`, so the arms differ in width and
nothing else. Matched TOKENS, not matched clock -- matching clock would hand the
small net 15x the data and measure the two effects summed. They replace
`train-136m`, which is deleted: see below.

**Two of the three arms are done** (tiny 2.9690, 9M 2.5516) and the 136M arm is
re-running as of 2026-08-05; the table in the status block at the top has the
detail. `scaling-curve` fires on its own once the third lands, and produces the
first `d(held-out loss)/d(log2 params)` this repository has ever had.

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

**The four `sims-*` rungs were marked FAILED on 2026-08-01 and RE-EARNED on
2026-08-02.** They had been the replayed matches, and leaving them rendering as
`+218 / +283 / +237` on the `sumofish-lab` board was a live-looking claim at a
point of use, which is what PHILOSOPHY says to delete rather than annotate.
Marking them failed made the lab re-run them, which is exactly what the marking
was for: the board now shows four rungs that were actually played, 300 games
each, on the fixed harness. The figures are in the status block at the top.

What this bought beyond the numbers: the harness has now produced a ladder
twice, once fraudulently and once not, from the same code path. The difference
was `scripts/verify_replays.py` and an elapsed time that could not have produced
that many games. Keep running it; a replay is invisible to every other check.

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
- **The virtual-loss fix is SETTLED AND DEPLOYED, and this bullet said the
  opposite until 2026-08-13.** It read "it still waits for the ladder -- the fix
  existing is not the same as the fix being worth shipping", while the
  recalibration block ~330 lines above this one says in as many words that the
  live bot has run `CHESSGPU_VLOSS_FIX=1` since 2026-07-30, where it earned its
  default on W25 D7 L0, LOS 100%, and +364 Elo at 400 sims
  (`runs/matches/vloss-fix-verify`). Both statements stood in this file at once
  for two weeks, one of them holding open a piece of work the other had already
  closed, and gating it on a ladder that has since been withdrawn, rebuilt in a
  new currency, and withdrawn again. **This is the fourth instance of the same
  drift in three days**, after the FPU record, the "queued and NOT started"
  line, and the transfer test that had finished three days earlier.

  The check that settles it costs one call and reads the RUNNING unit rather
  than the file on disk, which is the distinction that matters after any
  `systemctl --user edit`: `systemctl --user show sumofish-bot.service -p
  Environment`. Do that before believing any flag claim in this file.

  What survives from the old bullet, all still true: the defect was virtual
  loss going through a real `backup()` into `value_sum` (Q) at every node on
  the path, not the "applied to N and not Q" this line asserted before
  2026-07-30; the fix is in `rust/src/tree.rs` behind five Rust unit tests that
  prove it never leaks, never touches `value_sum`, and is not a silently-inert
  no-op; it stays `store_true`-off in `match.py`, because `tests/identity_*.py`
  hold the oracle identity against `sumofish.mcts` only with all three defects
  off, and on in `lab.py`'s `match_argv`, because that is the engine that plays.

- **`dedup` is not unpriced. It LOST, and the number that keeps it off is
  weaker than the decision resting on it.** `sumofish-bot.service` cites -168
  Elo at a fixed clock. That is **20 games, W0 D11 L9**, measured 2026-07-29
  jointly with `CHESSGPU_COMPILE`, so neither flag has a number of its own, on
  the warm harness later proven to hand Stockfish its hash between games,
  before the Rust core became the default, before `vloss_fix` was deployed, and
  before the v6 constants. Every confound this project has found since, in one
  measurement.

  **I wrote "keep it off anyway, the mechanism is sound" here earlier on
  2026-08-13, and then measured the mechanism and it is not.** The unit's
  argument is that dedup delivered "3,464 UNIQUE evaluations against plain's
  4,160", i.e. 17% less knowledge of the position. That comparison does not
  hold: **`unique_evaluations` counts ROWS SENT** (`rust/src/tree.rs:558`), so
  with dedup OFF it equals the leaf-visit count by construction and is not a
  distinct-position count at all. Nothing ever measured plain's duplicates. It
  is a deduplicated distinct-count against a non-deduplicated row-count.

  The like-for-like version is now `tests/verify_dedup.py` section 3, Rust
  against Rust at v6 constants with `vloss_fix` on, which is the search that
  actually plays: **root visit vectors byte-identical 12/12 at batch 32, 64 and
  256, while sending 54.2% fewer network rows at batch 64.** Identical trees
  mean both arms visit the same leaves and hold the same knowledge. Dedup pays
  for about half as many of them.

  The other half of the 07-29 measurement is stale in the way everything from
  that week is: it predates `vloss_fix`, which is precisely the mechanism that
  stops parallel paths collapsing onto one leaf. Measured 2026-08-13 with the
  real nets at 800 sims and batch 64, the collapsed fraction falls from **72.9%
  with `vloss_fix` off to 43.8% with it on**, and dedup ran ~11% faster on that
  workload.

  **So the -168 is not trustworthy and dedup is a live question again.** The
  unit is dedup ALONE, fixed clock, idle box, on the deployed engine. It stays
  off until that runs, because "the old verdict is unsound" is not the same
  claim as "the flag is good", and this file has been burned by that exact
  substitution before.

  **RUNNING as of 2026-08-14 08:30, and the sign is already reversed.**
  `runs/matches/dedup-time`, dedup alone at `--time 0.5` on the deployed
  engine, 600 games: **+20.7 +-15.2 at 522 games, LOS 0.996, 71% draws.** Not
  yet decisive, and already nowhere near -168: the old figure sits about twelve
  half-widths below this arm's point estimate. Wait for the full 600 before
  quoting a number, but the direction is not in doubt.

- **`mate_distance` is the only one of the three genuinely unmeasured**, and
  `tests/verify_mate.py` is why it is only HALF unmeasured. Run 2026-08-13 at
  400 sims over 34 exhaustively-solved forced mates: **25 proofs claimed, 0
  bogus** against an independent exhaustive solver rather than against the
  engine's own word, and the tree **24% smaller** (55,331 nodes to 41,868).

  What that oracle cannot see is MOVE CHOICE, and the reason is the harness and
  not the fix: it drives the Rust core with the mock evaluator from
  `identity_search`, whose random priors mean the search rarely finds a
  mate-in-2 at all and therefore rarely has a fast-versus-slow mate to choose
  between. The tell is in the script's own output, and it is worth keeping as a
  worked example of an instrument reporting on itself: the mate-in-2 rate gets
  WORSE from 400 to 2000 simulations. A real policy prior concentrates on
  forcing moves and would not do that.

  **PRICED, and it is a real gain: +23.7 +-6.7 Elo, LOS 1.0000**, over 2000
  games at 400 sims (`runs/matches/mate-distance-400sims`, W457 D1222 L321, 61%
  draws, seed 4242, both arms on the deployed `vloss_fix` and the v6 constants,
  `--no-sprt`). This is the largest measured gain in the project since the v6
  constants themselves, and it costs no training.

  **The mechanism is NOT what the flag is named after.** Two things rule out
  the obvious readings. `tests/verify_mate.py --real-nets` showed move choice
  saturated: with a real prior the engine already plays the shortest mate in
  34/34 positions with the fix and without. And the "shuffles in a won position
  until the fifty-move rule" failure did not appear either: **3 fifty-move
  games out of 2000.** What is left is the tree: refuting proven-lost branches
  makes the search 52% smaller on tactical positions, so at a FIXED simulation
  count the same simulations are spent on a better-chosen part of the tree.
  That is a search-allocation gain, and it is what this arm measured.

  Correction to what stood here on 2026-08-13: this arm was called "a lower
  bound, because the tree reduction is a clock gain". Half right. The reduction
  is ALSO a per-simulation cost saving, which this arm is genuinely blind to,
  but the pruning helps at fixed simulations too and that is most of the +23.7.

- **THE 8x-BUDGET ARM CAME BACK INCONCLUSIVE, and that is the operating-point
  rule doing its job rather than a null to shrug at.** `matedist-3200`, 400
  games at 3200 sims: **+10.4 +-15.8, LOS 0.903, 74% draws.** Its own interval
  includes zero, so it does not establish a gain. It also does not establish a
  DECAY: the difference from the 400-sim arm is -13.2 +-17.1, z = -1.51, so the
  two are statistically consistent.

  **It is underpowered, by construction, and the fix is games.** 400 games at
  3200 sims cost 6.1 GPU-hours; resolving +23.7 there needs roughly 2000, i.e.
  ~30 hours. `match.py` excludes `games` from the resume fingerprint precisely
  so an arm can be extended later, so re-running with `--games 2000` continues
  rather than replays.

  **Do not ship `mate_distance` on the 400-sim number alone.** The rule earned
  in `docs/OPERATING-POINT.md` was that a deployment decision needs a second arm
  higher up the budget axis. It has one, and it says "not shown". Deployment is
  a further 6.2 doublings above 3200 anyway.
- A blocking pre-push hook running `verify_replays.py --check`,
  `tests/verify_data.py`, and `tests/run_all.sh`.
- **DONE 2026-08-13: `docs/OPERATING-POINT.md`**, regenerate the measured half
  with `scripts/operating_point.py`. It was worth doing and not for the reason
  it was listed: **the lab measures at 400 simulations and the deployed engine
  searches a median of 231,273, which is 578x, or 9.2 doublings.** The parity
  ladder's top rung is 3,200 sims, so the operating point sits **6.2 doublings
  above the highest budget this project has ever measured at**, and the ladder's
  own central finding is that the exchange rate is not constant along that axis
  (1.15 node-doublings per sims-doubling at 200 sims, 0.49 at 1600).

  This invalidates nothing. Cheap arms are what make 2,000-game intervals
  affordable, and the one direct test of upward transfer came out the good way
  (+35.6 +-26.0 MORE for the v6 constants at 3,200 than at 400, z = 2.68). The
  rule it earns is narrower: **anything meant to justify a deployment decision
  needs a second arm at a higher budget, and every write-up states the budget it
  was measured at.** `transfer-400`/`transfer-3200` is the pattern.

  Two things fell out that are not about budget. The engine spends **100% of its
  allowance on every move and has never taken under a second**, so there is no
  early stopping and no instamove, and it thinks for 19 seconds in positions
  with one legal reply. And two concurrent games cost nearly nothing (7,761
  nodes/s under `concurrency: 2` against 7,763 measured idle), which turns
  cross-game batching from plausible into measured.
- `CHESSGPU_BATCH` defaults to 64 and 256 is faster per call, but batch size
  trades against search quality (more virtual loss in flight, more collisions).
  Given item 2 above, settle it on `unique/s` and then in a game, never on nps.
- **Time management: instamove and early stopping BUILT 2026-08-13, both
  default off, neither measured yet.** This line said "untouched" and the
  operating-point work is what made the case: across 336 logged moves the engine
  spent **100% of its allowance on every move and never once took under a
  second**, including in positions with a single legal reply, where it thought
  for 19 seconds.

  `CHESSGPU_INSTAMOVE` gives a forced move 50 ms rather than the full budget.
  Deliberately not zero: `reroot()` in `rust/src/tree.rs` declines once more
  than 2 plies have passed since the last root, so skipping the search outright
  would discard the tree on the FOLLOWING move and pay for the saving twice.

  `CHESSGPU_EARLY_STOP` stops once the runner-up cannot be caught even if every
  remaining simulation were handed to it. It **provably cannot change which move
  is played**, only when: the move is `max(visits)` either way, and the rule
  requires the leader's margin to exceed the optimistic remaining simulations by
  a 1.25 safety factor. `tests/verify_time_management.py` checks that as an
  algebraic bound rather than on examples -- since `lead <= done` always, the
  rule cannot fire before **5/9 of the budget** has elapsed, and a regression
  that ate the safety margin would not show up in any single case. The test also
  asserts the rule is REACHABLE, so it cannot decay into a no-op wearing a
  feature's hat.

  Rust core only, and gated rather than merely listed: `sumofish.mcts.MCTS.search`
  has no `should_stop` parameter, so an ungated flag would be a TypeError on
  move one of a live game after a `CHESSGPU_CORE=python` rollback, not the
  silent no-op the other four RUST_ONLY_FLAGS are.

  **Neither has an Elo number, neither ships without one, and THE HARNESS
  CANNOT CURRENTLY PRODUCE ONE.** `scripts/match.py --time` is seconds per MOVE,
  not a game clock, and `Player.move` calls `self.mcts.search(board,
  deadline=...)` directly: it never goes through `search_engine.choose`, so
  `think_time` and everything built on it is not exercised by any match this
  project can run. Instamove would still fire but its saving would go nowhere,
  and early stopping's whole payoff is that the banked time raises later moves'
  budgets, which a fixed movetime cannot express.

  That is almost certainly why time management was never built: it could not be
  priced, so it never came up. It is now the top item under "Open, smaller"
  because it gates more than itself, see the game-clock item below.

  Ponder is still untouched and is a different kind of change: it searches on the
  opponent's clock, so it interacts with `concurrency: 2` on one GPU.

- **DONE 2026-08-13: `match.py --tc BASE+INC`**, a real per-side clock with
  increment, routed through the SAME `think_time` the bot uses (imported, not
  reimplemented, so it cannot drift). This is what makes instamove and early
  stopping measurable at all, and it is the only route this project has toward
  measuring near its own operating point: every arm in the archive is
  fixed-simulation and deployment is clock-bound, 9.2 doublings away.

  Verified on a 4-game smoke test: the arm with both flags used **96% of the
  other's clock**, consistently in every game. That is the mechanism, not an Elo
  number, and 4 games cannot be one.

  **Two bugs, both caught by running it and neither by reading it**, and both
  worth remembering because they are the same shape:
  1. `simulations=10**9 if spec.movetime else spec.sims` still bound under
     `--tc`, so both sides stopped at 400 sims while the flagged side, taking
     the sliced search path, ran on to its deadline. It spent 3x the other's
     clock and the match was measuring "400 sims vs clock-bound". **The comment
     on that exact line already documents this bug happening once for `--time`.**
  2. The kernel warmup was `p.move(chess.Board())`, which takes its budget from
     the match's mode. Under `--tc` that mode is "the clock decides", so with no
     clock it got `deadline=None` on top of `simulations=10**9` and searched
     forever. The first `--tc` run never played a move.

  Both flags now REFUSE rather than run inert: `--a-instamove`/`--a-early-stop`
  without `--tc` exits 2, and early stopping additionally requires the rust
  core. Per-side clock usage is recorded per game, so a flag that is on and does
  not move the clock is visibly not reaching the search.

  What a fast TC cannot do is make its number an absolute. It makes the
  MECHANISM measurable, which is what these two need.
- The value net enables resignation, draw offers, and calibrated difficulty.

**Numbers worth remembering:**
- Behavioural cloning, 8.5h, 307M positions -> 40.9% puzzles. Now the search's
  policy prior.
- State value warm-started from its body -> 57.4% puzzles at 12% trained.
- The 9M state-value curve: 48.6% at 10k, 64.8% at 100k, 67.0% at 150k, 68.7%
  at 280k; adjacent evals bounce 0.77 points. Flat from ~200k, while train loss
  was still falling (2.2422 at 200k, 2.2133 at 292k). **The "that is
  underfitting, not capacity-bound" reading of this is DELETED, 2026-08-14.**
  It rested on "307M samples is under one epoch", and the deployed net is now
  at 921.6M positions = **1.74 epochs**, so the premise expired. Worse, the
  diagnosis never discriminated: all three width-sweep arms show the same
  train-minus-val sign across a 1000x parameter range (tiny -0.077, 9M -0.099,
  136M -0.092), because held-out is scored on EMA weights against a running
  mean of raw ones. An earlier version of this file asserted the opposite of
  the underfitting reading and steered a 37-hour run; this file then asserted
  the underfitting reading for two weeks on evidence that could not support
  either. See LAB-NOTES 2026-08-14.
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
