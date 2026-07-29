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

## Where things stand (2026-07-29, end of session 5)

**Read this section first. Two things were built today and NEITHER IS IN USE.**

**1. The measurement loop is honest now, and it is merged.** `scripts/match.py`
no longer caps `--time` at `--sims` (it did, so every "clock" match ran at 400
simulations), refuses to resume across a spec change, and adjudicates only when a
fixed-node full-strength Stockfish agrees. `scripts/verify_replays.py` audits the
archive and currently reports **0 of 8 matches trusted, 4 REPLAYED**. See
`docs/2026-07-29-ladder-retraction.md` for what that invalidated, and
`docs/induced-failures.md` for the two bugs found by deliberately inducing the
new guard's refusal -- including one where a refusal exited 0, which `lab.py`
gates on as success.

**Consequence: no proposal may be justified by an Elo-per-doubling figure.** The
ladder can be re-earned for ~10 GPU-hours, with a distinct seed per rung, and it
buys the ordinal only.

**2. The hot path is ported to Rust and verified, but NOTHING CALLS IT.** Ten
independent oracles, byte-identical to `chessgpu/` throughout. Measured 8.9x on
the engine profile; the tree alone is ~300x. `rust/README.md` has the numbers and,
more usefully, the three findings that only a benchmark could produce -- including
that the prior softmax **cannot** be ported, because numpy's float32 `exp` is not
correctly rounded and differs on 39.7% of inputs.

Two real fixes landed there behind flags: **leaf dedup**, which turned out to be
identity-PRESERVING (dedupe the evaluation, not the backup) and removes 66.4% of
network rows at the live batch of 64; and **mate distance** via MCTS-Solver, with
sound proofs and a 24-49% smaller tree.

**What is NOT done: integration.** The live engine still runs Python. The 8.9x is
potential. Integrating it changes what a rated bot plays, so it wants the
re-earned ladder first.

**The 136M is still training** (~step 66,000 of 400,000 as of this writing) and is
compromised twice over: `--init-from` transferred **0 of 93 tensors** because the
donor is width 256 and the target is 1024, so it is a cold start its own job
comment said must not happen; and it is scheduled for 102.4M positions, 19.3% of
an epoch, against the 9M's 307M. A loss from its promotion match is uninformative.
Decide whether to sever its outputs before that match runs.

## Next session, in order

**1. Read the lab first: `sumofish-lab`, then `runs/lab/report.md`.** Two
training runs and two clock matches were in flight at handoff. If a candidate
won and passed the smoke gate it was promoted automatically and a version was
cut, so check `VERSIONS.jsonl` and `sumofish-games` before assuming the bot is
still v1.

**2. Play the acceptance test. It has never been run.**
`scripts/acceptance.py --games 20`. Twenty blind games, one rating each,
arms hidden until `reveal`. Every other number in this repo measures strength,
which PHILOSOPHY ranks third. This is the only thing pointed at goal two, it
needs a human by construction, and no amount of GPU substitutes for it.

**3. Finish what Stockfish is for.** Live move grading is done and in the moves
panel. Three uses remain, and the second is the most valuable:

  - **A third-party adjudicator for `match.py`.** Today the engine under test
    judges its own adjudicated wins, so a net that is more *confident* rather
    than more correct banks more of them, and miscalibration and strength
    become the same number. That decided 65 of 300 games in one match.
  - **An absolute Elo anchor.** Every match here is relative, A against B, so
    the whole ladder floats. Stockfish at a pinned Skill Level and depth turns
    "+21 over the previous build" into a number that survives a rebuild.
  - **An outside notion of "sharp position"** for the blunder-locality proxy,
    which is contaminated when read off the engine's own eval curve.

**4. Merge `search-and-lab` into `main`** once the lab is quiet. 29 commits.
Nothing may edit `chessgpu/` while a match runs -- use a worktree.

**5. The search is still the lever, and it is CPU-bound.** After this session's
2.6x, `board.push`/`pop` is ~24% of wall clock and move generation ~20%. That
is 44% inside python-chess and the network is 9%. At roughly +50 Elo per
doubling at the deployed budget, halving that overhead is worth real rating.
Kernels remain the wrong target until it is fixed.

**Open, in rough priority order:**

0. ~~No way to measure whether a checkpoint is better.~~ **Done:
   `scripts/match.py`.** Kept here because the reasoning still governs how to
   read every other number in this file: puzzle accuracy is a progress metric
   with a +-1.5% sigma at n=1000, the bullet rating has an RD of +-72, and the
   old 7-17-0 searchless result was 24 games, good for about +-200 Elo. None of
   those can see a 20-Elo change. The harness can, by playing more games.
1. Remove `opponent_max_rating: 2200` from `config/lichess-bot.yml` once the
   rating is non-provisional (~30 rated games). It is a training wheel that
   exists only because lichess starts every BOT at a fake 3000.
2. Two harness bugs from the code audit, neither affecting the live bot:
   `research/run.py::best_so_far` returns the global minimum bpm rather than
   the score belonging to `best.py`, so NOISE runs permanently raise the bar
   and nothing can ever be promoted again. And nothing actually enforces the
   "FROZEN" constants in `research/train.py` -- an agent editing it could
   lengthen the budget or rewrite the eval undetected.
3. ~~Free search speed in `mcts.py`.~~ **Done, 2.6x.** See the top of this
   file. What is left of it: `CHESSGPU_BATCH` still defaults to 64 and 256 is
   36% faster, but batch size trades against search quality (more virtual loss
   in flight, more collisions) and that trade has not been played out yet.
   Settle it with the harness before changing the default.
3b. **The selection loop is now the bottleneck: `_puct` plus its `max()` is
   29% of the search.** Children are a `dict[Move, Node]` scored by a Python
   function per child per ply per simulation. Lay a node's children out as
   parallel arrays (prior, visits, value_sum) and score them with one numpy
   expression. This is the memory-layout work PHILOSOPHY.md step 3 names, and
   it is the last big algorithmic win before kernels are worth writing.
4. One net with two heads, not two nets. `_simulate_batch` runs two forward
   passes per batch, `_priors_batch` through the policy net and `evaluate`
   through the value net. A shared trunk with a 1968-logit policy head and a
   64-bin HL-Gauss value head halves per-node GPU cost, frees VRAM, and is the
   architecture self-play RL (roadmap step 5) needs anyway. Warm-start it: 91
   of 93 tensors transfer.
5. Scale to the 136M preset. The 9M is capacity-bound, not data-bound -- train
   loss was still falling at 300k (2.2422 at 200k, 2.2133 at 292k) while
   puzzles went flat, and 307M samples is under one epoch of a 36GB bag, so
   overfitting is not possible. Same 8 layers at width 1024, and 9M runs at
   ~11% MFU, so benchmark 200 steps rather than assuming 15x wall clock.
6. CUDA kernels, *after* 3-5, when the GPU is actually the bottleneck.
7. Tune `c_puct` (2.0) and `fpu` (-0.2). Never measured. Needs the harness.
8. The value net enables what the policy net could not: resignation, draw
   offers, and calibrated difficulty ("play the move X worse than best").

**Numbers worth remembering:**
- Behavioural cloning, 8.5h, 307M positions -> 40.9% puzzles. That model is now
  the search's policy prior, not dead weight.
- State value warm-started from its body -> 57.4% puzzles at 12% trained.
- Search beat searchless 7-17-0 (64.6%) with a value net 3% trained.
- First human game: won in 9 moves, real Sicilian theory.
- The whole 9M state-value curve: 48.6% at 10k, 64.8% at 100k, 67.0% at 150k,
  68.7% at 280k, and adjacent evals bounce by 0.77 points. Flat from ~200k.
- The engine gets ~1600 nps at 12% GPU, and it is **CPU-bound in python-chess**,
  not launch-bound. The network is 5% of the search's wall clock.
- Puzzle accuracy at n=1000 has a binomial sigma of +-1.5%, which is larger
  than the entire 150k->300k gain anyone was trying to read off it. `train.py`
  logs no held-out loss at all; add one on `data/test/state_value_data.bag`.

**Left unanswered:** what looks wrong about the evaluation panel. The chart
bug behind "one side is winning when it is not" is fixed (two games on one
curve), but there was an earlier complaint about that box that was never
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
scripts/watch.py     the `sumofish` dashboard (composition root)
scripts/dash/        its three layers: sources -> state -> panels
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
`scripts/dash/make_sprites.py` rasterises them straight to the pixel grid the
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
