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

## Where things stand (2026-07-29, end of session 6)

> **OVERNIGHT STATE, left running 2026-07-29 ~20:10.**
>
> - `chess-gpu-continue.service` is training the 9M value net from step 321,746
>   to 600,000. ~8 h. systemd-supervised, `Restart=on-failure`, `--auto-resume`,
>   so a crash resumes from `latest.pt` instead of losing the night.
>   Watch: `journalctl --user -u chess-gpu-continue -f`.
> - **The bot is DOWN and no longer autostarts.** It was stopped for the GPU,
>   then restarted at 19:42:31 by what was almost certainly a PARALLEL Claude
>   Code session doing dashboard work (commits by Jessupthefish at 19:41
>   "the results panel was empty" and 19:47 "snapshot every panel" -- both want
>   a live bot to populate panels). The watchdog explicitly declined ("not ours
>   to restart") and nothing in the repo starts it, so it came from outside.
>   **If you run two sessions at once, say so up front**: a second session
>   starting the bot during a wall-clock match silently invalidates the match,
>   and starting it during training halves the throughput of both. `systemctl --user disable` removed the unit symlink
>   as well as the autostart, so it was re-linked: the unit is `linked` and
>   `inactive`, startable with `systemctl --user start chess-gpu-bot`, and no
>   longer pulled in by `default.target`. Restore autostart with
>   `systemctl --user enable chess-gpu-bot` -- but find the restarter first,
>   because an unexplained start during a wall-clock match invalidates it.
> - `chess-gpu-train-watchdog.timer` is STOPPED. Its remedy is restarting the
>   LAB, and this run is direct, so it would have launched a conflicting job.
>   The systemd unit above supersedes it for this run. Restart the timer when
>   lab-managed training resumes.
> - **Nothing is measuring strength right now.** Training is not measurement,
>   and the bot -- the only absolute anchor -- is paused. The rungs get priced
>   with `scripts/ladder.py` once the run has produced them.

**The engine is Rust, provably identically, and 3.6x faster. The two extra
speed flags were measured and cost 168 Elo, so they are off.**

**1. The Rust core is integrated and live-capable.** `CHESSGPU_CORE=rust`
selects `chessgpu/rust_mcts.py`; unset keeps Python, so rollback is an
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

**1. Deploy the Rust core to rated play.** `CHESSGPU_CORE=rust`, both speed
flags off. This is a strength change (4.5x the search on the clock) with a
proof rather than a match behind it, and it touches rated games, so it is a
human decision. Nothing else on this list is blocked by it.

**2. Play the acceptance test. It has still never been run.**
`scripts/acceptance.py --games 20`. Twenty blind games, one rating each, arms
hidden until `reveal`. Every other number here measures strength, which
PHILOSOPHY ranks third; this is the only thing pointed at goal two, and it
needs a human by construction.

**3. Re-earn the exchange ladder** (~10 GPU-h). Visit-denominated, distinct seed
per rung, on an idle machine. Until it exists, every "worth N Elo" in any plan
is a guess, and the plan's ordering is unfalsifiable.

**4. Stockfish as an absolute anchor**, at pinned nodes. Every match here is
relative, so the whole ladder floats. The adjudicator half is done.

**5. Retrain the policy prior.** Frozen at 40.9% puzzles since session 1 and it
is the biggest neglected lever: it sets the shape of every search, and
`c_puct`/FPU/temperature tuning is not worth doing before it, since the right
values depend on the prior's sharpness.

**6. Shared-trunk two-head net** (~40 GPU-h). Worth ~1.8x now that the network
is 89% of the search. Warm-start it: 91 of 93 tensors transfer from the 9M body.

**7. Cross-game batching.** N games feeding one evaluation service. The engine
is launch-bound below ~128 rows -- one row costs the same 7.2ms as 128 -- so
this is close to free throughput and it is a measurement multiplier before it
is a strength one.

**8. The scaling curve has zero points.** A width sweep (~6 GPU-h) at matched
tokens is what makes "scale it up" an argument instead of a preference.

**9. `d(loss)/d(params)`, then kernels.** Not before 5-8.

**Open, smaller:**

- Remove `opponent_max_rating: 2200` from `config/lichess-bot.yml` once the
  rating is non-provisional (~30 rated games). A training wheel that exists
  only because lichess starts every BOT at a fake 3000.
- Virtual loss is applied to N and not Q. The last port fix with no proxy
  behind it; it needs Elo, so it waits for the ladder.
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
- Puzzle accuracy at n=1000 has a binomial sigma of +-1.5%, larger than the
  entire 150k->300k gain anyone was reading off it.
- Post-port profile: network **89.3%**, tree 1.3%. Launch-bound below ~128 rows.
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
chessgpu/            frozen infrastructure. tokenizer/bagz are VERIFIED, do not edit
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
tests/verify_data.py the correctness gate. run it after touching chessgpu/
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
`chess-gpu-lab.service`. It exists because everything left on the roadmap needs
the GPU, needs hours, and needs to happen in an order where later steps read
earlier results -- which by hand means being present at every handoff, at 3am,
for two days.

    sumofish-lab            what it has run, what it concluded, what is next
    sumofish-lab watch      the same, refreshing
    sumofish-lab reset --job <id>    forget one job so it runs again
    journalctl --user -u chess-gpu-lab -f

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
tests/verify_data.py                     # correctness gate, run after any chessgpu/ change
tests/verify_search.py                   # the same, for the search's rules and tokenizer
tests/verify_layout.py                   # the same, for the dashboard's board column

# "did that help?" -- two checkpoints, equal thinking, a few hundred games
scripts/match.py --a-value runs/A.pt --b-value runs/B.pt --games 400 --sims 400
# what a speedup is worth, without needing the old code to compare against
scripts/match.py --a-sims 1000 --b-sims 400 --games 300
# fixed time instead of fixed sims: strength as deployed, speedups included
scripts/match.py --a-value runs/A.pt --b-value runs/B.pt --time 0.5
journalctl --user -u chess-gpu-train -f  # watch training
systemctl --user status chess-gpu-bot    # the lichess bot
scripts/promote.py runs/9M-causal/best.pt
research/run.py --note "hypothesis"      # one autoresearch experiment
research/run.py --status                 # leaderboard
```

Python is `.venv/bin/python` (3.12). Never the system python.

## Non-negotiables

- **Never edit `chessgpu/tokenizer.py` or `bagz.py`** without re-running
  `tests/verify_data.py`. The tokenizer is byte-exact against DeepMind's
  published implementation over 12,000 real positions. Break that and every
  number stops being comparable and their pretrained checkpoints stop being a
  valid reference.
- **Never change the eval, the time budget, or the seed in `research/train.py`.**
  Moving the yardstick to make the metric improve is the exact failure the
  harness exists to prevent.
- The lichess token is in `~/.config/chess-gpu/bot.env`, chmod 600, outside the
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
