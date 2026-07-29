# Session archive

Superseded "where things stood" sections, in reverse order. Moved out of
`CLAUDE.md` on 2026-07-29.

Kept rather than deleted because they carry the REASONING behind decisions, which
a summary loses. Not kept in the file everyone reads first, because four
overlapping status sections at the top of a 68 KB file is how the current one
stops being findable.

Anything here may be out of date by construction. `STATE.md` is the only current
account.

## Where things stood (2026-07-29, end of session 4)

**Read this section and "Tomorrow" below; the rest is reference.**

**The bot is v1 and rated 2379 rapid, up from 1950 bullet.** `VERSIONS.jsonl`
is the registry, `scripts/release.py` cuts one, and every record shown counts
from the current version's boundary rather than over engines that no longer
exist. `sumofish-games --all` still shows all 120 historical games.

**The single biggest win was a config change, not a model.** It plays 15+10 and
nothing else now. The engine is clock-bound, so that took it from ~0.4s and a
few hundred simulations a move to ~30s and ~36,000, and the rating moved 444
points. Search speed is worth far more here than anything in the roadmap
predicted.

**The exchange-rate ladder used to be the number that priced the rest of the
project. It is retracted.** Elo per doubling of simulations, 300 games each:

    RETRACTED 2026-07-29. All four rungs were REPLAYS: the jobs credited with
    them re-served existing match logs rather than playing. Detectable only as
    an elapsed time that could not have produced that many games -- 5 seconds
    credited for 0.7 to 2.6 HOURS of logged play on three of them, and 1,070s
    for 5.8h on the fourth. Run `scripts/verify_replays.py`, which checks
    `sum(game.seconds) <= job.seconds` and currently reports 0 of 8 matches
    trusted.

The "+50 Elo per doubling" extrapolated from that ladder is withdrawn with it,
and so is "two independent estimates agreeing is the strongest evidence in the
project". The +74 from the rating jump is confounded -- v1 bundles a 2.5x search
speedup with the 1+0 -> 15+10 switch under a single rating delta -- and the +50
was chosen so the extrapolation would land near it. One number was fitted to the
other; they were never independent.

**No proposal may currently be justified by an Elo-per-doubling figure.** The
ladder can be re-earned for ~10 GPU-hours once the fingerprinted resume in
`scripts/match.py` is merged, and it buys the ORDINAL only: at 300 games a rung
the standard error is +-35-40 Elo, so flat-versus-attenuating is not resolvable
at that budget. Say so when reporting it.

**What is running right now, unattended:** `chess-gpu-lab`, 9 of 15 jobs done,
currently training the 136M (~35h), then a 900k-step 9M (~40h), then both are
played against the live net on a CLOCK, and the winner is promoted
automatically if it also passes `scripts/smoke.py`. Promotion cuts a version by
itself. `sumofish-lab` to watch.

`scale` already did its arithmetic: a doubling is worth 237 Elo at 2400 sims
and 136M costs **2.17x per node inside the search** (not the 10x its parameter
count implies -- the network is only ~9% of wall clock), so the 136M must be
265 Elo better at equal simulations merely to break even on a clock. That is a
steep bar and it is running anyway; the fixed-time match decides, not the
estimate.

**Things that exist now and did not yesterday:** `scripts/match.py` (Elo with
intervals, pair-scored, SPRT), `scripts/elo.py`, `scripts/lab.py` (the queue),
`scripts/smoke.py` (does a checkpoint actually work), `scripts/games.py` +
`sumofish-games`, `scripts/acceptance.py` (the blind human test, **never yet
run** and still the only instrument aimed at PHILOSOPHY's second goal),
`scripts/release.py`, `chessgpu/rules.py`, Stockfish 18 in `tools/` grading
every move in the dashboard.

**Two things that look like bugs and are not.** `sumofish` must be RESTARTED to
pick up anything in this session -- the game log panel, Stockfish move grading,
the full-game eval curve, the trimmed header -- because the running process
imported the old modules at start. And v1's record reads 0W 0D 0L until the
first 15+10 game finishes, which takes ~25 minutes; that is the versioning
working, not a broken counter.

**Branch `search-and-lab`, 33 commits, NOT merged to main.** Everything is
pushed. `main` still holds only the old dashboard work.

## Where things stood (2026-07-28, session 3)

**The measurement instrument exists.** `scripts/match.py` plays two
configurations against each other from paired ECO book openings, reports Elo
with a 95% interval, LOS and a sequential test that stops the match once the
answer is decided, and resumes from its own log after a Ctrl-C. Open item 0 is
closed: "did that help?" is now a question with an answer.

**The search is ~2.6x faster and nothing about the nets changed.** Same
midgame position, same method as the profile below:

    | config                        | before | after |
    |-------------------------------|--------|-------|
    | batch 64  (what was live)     |   1475 |  2447 |
    | batch 256                     |   1853 |  3322 |
    | batch 1024                    |      - |  3813 |

Four changes, in the order the profiler ranked them, none of which touch a
weight:

1. `chessgpu/rules.py::terminal_value` replaces `outcome(claim_draw=True)` at
   every leaf. That call was **41%** of the search. The module docstring
   explains why the version it replaces was also wrong for a tree search, not
   just slow.
2. **Tree reuse** (`MCTS._reroot`). The tree is re-rooted into the node the
   opponent's reply reached instead of being rebuilt. Measured on a 200-sim
   self-game it inherits 70-460 visits per move, so a 200-sim search often runs
   on 400-660 simulations' worth of evidence.
3. `tokenizer.tokenize_board` reads the 77 tokens off the bitboards instead of
   building a FEN string and parsing it back. 37.3us -> 9.6us, byte-identical
   over 100,836 positions.
4. `tokenizer.ACTION_BY_MOVE_KEY` replaces `MOVE_TO_ACTION[move.uci()]`, which
   was building 510,000 throwaway strings per 5-second search.

`tests/verify_search.py` is the gate for all of it: 80,561 positions comparing
the new terminal test against the old, the tokenizer diff, the mate-in-one
canary, and the re-rooting accept/decline cases. `tests/verify_data.py` still
passes untouched.

**Where the time goes now** (same 5s profile, after the changes): `_puct` and
the `max()` around it **29%**, move generation ~20%, the network **9%**. The
next algorithmic win is the selection loop -- children live in a dict and are
scored by a Python function called once per child per ply per simulation. An
array-of-children layout (Lc0's) is the fix and it is a memory-layout problem,
which is the systems half PHILOSOPHY.md step 3 is about. Kernels still come
after that: the GPU is at 9%, not 90%.

**Deploy note:** the live bot picked these up with no restart, because
lichess-bot spawns a fresh engine per game and the engine imports from the
working tree. Games starting after the edit landed logged 2400-2650 nps. That
is the same property as the checkpoint swap: convenient, and it means an edit
to `chessgpu/` is a live deploy to a rated bot whether or not it was meant as
one.

## Where things stood (2026-07-28, end of session 2)

**The 9M state-value run is FINISHED.** 300,000 steps, final loss 2.2106, best
puzzle accuracy 68.7% at step 280,000. `runs/value.pt` is that checkpoint,
deployed 22:12, replacing step 150,000 (67.0%).

**The curve is flat and that is the headline.** Steps 10k->150k bought 18.4
points of puzzle accuracy; steps 150k->300k bought 1.7, against a
sample-to-sample noise of 0.77. More steps at 9M parameters are done paying.
The next real strength comes from a bigger model, better data, or more search.

**More search is the one with evidence behind it.** With training stopped the
GPU sits at 12% and 2.1/16G, and the engine's throughput did not move: 1615 nps
while training was competing for the card, 1602 with the card to itself. So
freeing the GPU is not on the critical path.

**Corrected 2026-07-28: the search is CPU-bound in `python-chess`, not
launch-bound.** cProfile over a 5s search on a real midgame position, live
nets, batch 64, total 4.79s: `can_claim_threefold_repetition` **41%** (1.95s
cumulative from only 3393 calls, 573us each), `board.push`/`pop` 24%,
`generate_legal_moves` 19%, `board_fen`+`tokenize` 13%, `_puct`+`max` 10%, and
`torch._C._nn.linear` -- the network itself -- **5%**. The card is at 12%
because it is only being asked to do 5% of the work, so CUDA kernels are capped
at ~1.05x by Amdahl until the tree is fixed. Measured on the same position:

    as-is (claim_draw=True)      1795 nps
    claim_draw=False             2724 nps   +52%
    claim_draw=False batch=1024  3138 nps   +75% over the live batch-64 config

The culprit is `_expand` calling `board.outcome(claim_draw=True)` at every
leaf, which rescans the whole move stack for repetitions. Do not simply drop
the flag -- repetition awareness is real strength -- keep an incremental
`dict[zobrist, count]` alongside the push/pop that already happens.

| unit | what |
|---|---|
| `chess-gpu-lab` | **the experiment queue.** Owns the GPU roadmap. `sumofish-lab` to look |
| `chess-gpu-train` | inactive by design now -- the lab starts training runs, not this |
| `chess-gpu-bot` | live, **searching** engine, playing rated bullet/blitz against bots |
| `chess-gpu-watchdog.timer` | bot stall detection (note: not `-bot-`) |
| `chess-gpu-train-watchdog.timer` | training stall detection |
| `chess-gpu-rating.timer` | samples the lichess rating every 15 min |

**Deploy the current best whenever it is meaningfully ahead** -- there is no
reason to wait for the run to finish, because the swap is free and takes effect
on the next game. Done once already at step 150,000, replacing step 10,000.

```sh
python -c "import shutil,os; shutil.copyfile('runs/9M-sv-warm-full/best.pt','runs/value.pt.staged'); os.replace('runs/value.pt.staged','runs/value.pt')"
sumofish rating        # the rating jump should show against the deploy marker
```

The searching engine reads `runs/policy.pt` (behavioural cloning, frozen) and
`runs/value.pt` (state value). **Swapping the value net is a file copy and
nothing else** -- no restart, no code changes. lichess-bot spawns a fresh
engine process per game, so the next game picks the new checkpoint up on its
own; the engine's `boot` records in `logs/engine.jsonl` show which one each
game loaded. An earlier version of this file said a restart was needed. It is
not, and restarting costs whatever game is in progress (#1101).

Copy atomically -- write beside it and rename -- because a game starting
mid-copy would otherwise read a torn file.

