# Lab Notes: what not to do

**Append-only. Dated. Never edited.**

This is the experimenter's notebook, split out of `CLAUDE.md` on 2026-07-29 so
that status updates which expire in a day stop burying findings that do not.
Read it before starting work in this repo: it exists to carve dead approaches out
of the search space so the next session does not rediscover them.

Format for a new entry: what was tried, why it failed, what to do instead. Add to
the top. Never delete an entry -- if it turns out to be wrong, append a correction
and say so, because a note that was believed for a month is itself evidence.

## Lab Notes: what not to do

- **Judging one rasteriser against another by counting differing pixels.** Comparing resvg
  against `rsvg-convert` on the same board SVG gave 1.96% of pixels differing by more than
  2/255, max channel delta 108, which reads like a failed port and nearly got resvg rejected.
  It was **entirely antialiasing gamma on piece outlines and coordinate glyphs.** Two measures
  separate AA from structure and both are cheap: mean-downsample both images 4x and re-compare
  (AA averages away, a missing or shifted glyph does not -- worst delta fell from 108 to 16),
  and count only differing pixels whose 3x3 neighbourhood in the other image contains no near
  match. Then **write an 8x-amplified diff PNG and look at it**, because no scalar settles
  "does it look right". Both live in `dashboard/xtask/src/probe_svg.rs`.
- **`r#"..."#` around a Python snippet that contains a `"#rrggbb"` colour literal.** The `"#`
  closes the raw string in the middle of the palette and the compiler reports fifteen errors
  about unknown prefixes and missing semicolons, none of which mention strings. Use `r##"..."##`.
  Cost one build cycle in the M0 probe, which embeds the board palette to call python-chess.
- **The sixel Lab Notes below about `rich` erasing the picture do not apply to the kitty
  protocol.** Measured 2026-07-29 (`dashboard/docs/probe-results.md`): Konsole 26.04.3 keeps
  kitty graphics in a layer separate from the text grid, so ten lines of text written *on the
  image's own rows* left it completely intact, and `a=d,d=i` deletes a placement by id so
  `ESC[2J` is no longer the only eraser. Those notes remain true for the sixel fallback. Do
  not port their workarounds -- the reserved-bare-region trick, the ban on animated cells in
  the board column -- into the kitty path, where they buy nothing.
- **Believing any match result without checking `sum(game.seconds) <= job.seconds`.**
  On 2026-07-29 all four rungs of the exchange-rate ladder were found to be
  **replays**: `match.py` resume keys on `rec["game"]` index alone, so a job with
  different code, config and budget lands on an existing `runs/matches/<name>/`
  and reports it as its own work, and `match.py:458` then rewrites `config.json`
  over it, so the directory asserts a provenance it never had. Three rungs were
  credited 5 seconds for 0.7-2.6 hours of logged play; the fourth 1,070s for
  5.8 hours. **The per-game timings inside `games.jsonl` are organic, so the
  replay is invisible in the artifact everyone reads** -- it exists only in a
  wall-clock field nothing checked. The inequality above is physically
  impossible to violate legitimately and catches all four. The structural fix is
  to content-address the run directory on (code sha, config, args) and make it
  write-once; hash file CONTENT, because `runs/value.pt` is a mutable path that
  promotion overwrites in place.
- **Editing `lab.py` or `elo.py` to change the behaviour of a running queue.**
  `lab.py run` is a long-lived process; its own module and everything it
  imported are frozen in memory, so the edit is a no-op and restarting to load
  it kills the training run in flight. Only `match.py` and `smoke.py` are
  re-read, because they are spawned as fresh subprocesses.
- **Forcing the smoke gate to fail as a kill switch.** `lab.py:358-363` writes
  the refusal into the permanent record as a *cause* (`"NOT promoted despite
  winning by +X Elo: {reason}"`), so a doctored gate manufactures false
  provenance, and `lab.py:357` overwrites the last genuine `smoke.log`. It is
  also unnecessary: promotion keeps `value.pt.previous`, is an atomic
  `os.replace`, and records its own rollback command, so an unwanted promotion
  is one `cp`. If a hold is needed, use a truthful `runs/lab/HOLD` marker that
  reports the real reason.
- **Quoting "+50 Elo/doubling and +74 from the rating jump agree".** They are
  not independent: the ladder covers four doublings and the deployment six or
  seven, so reconciling them needs a decay assumption, and +50 *is* that
  assumption, chosen to land near +74. The second number is the target the first
  was fitted to. Also `VERSIONS.jsonl` v1 bundles a 2.5x search speedup with the
  1+0 -> 15+10 switch under one rating delta, so it measures nothing separable.
- **"The 9M is capacity-bound."** It is underfitting: held-out 2.1438 is *below*
  train 2.2106. Train loss falling while puzzle accuracy flattens is the
  signature of a data/compute-bound model; a capacity-bound model has its
  *train* loss flatten. The puzzle plateau (0.679 -> 0.675) was inside a sigma
  of 1.5. `lab.py:524-529` has said so in-tree all along.
- **`--init-from` across a width change.** It copies only tensors whose name AND
  shape match, silently, and prints the count to nobody. A width-256 donor into
  a width-1024 model transfers **0 of 93 tensors**; the 136M run was a cold start
  its own job comment said must not happen. Make it refuse below ~90% transfer.
- **Porting action-value on the strength of "88.9% vs 65.7%".** That pairs a
  full-data run against a small ablation. Data-matched, state-value and
  action-value are tied. And the AV bag is shuffled per (position, move), so an
  AV net needs ~35 rows per node instead of 2 -- plausibly Elo-negative here.
- **Calling the shared-trunk two-head net a speed win.** It halves the GPU's
  share of wall clock, and the GPU is 9%, so it is ~+3 Elo. Build it for VRAM
  and for co-versioning the prior, not for throughput.
- **Treating local Syzygy or the "unused" tablebase as free Elo.** `online_egtb`
  is already enabled at `max_pieces: 7`. A corollary that contaminates analysis:
  **lichess's servers play the endgames**, so any per-phase strength breakdown
  will report an endgame competence this engine does not have.
- **Adding Geometric Attention Bias "without touching the frozen tokenizer".**
  GAB presupposes board squares as tokens; the 77-token FEN carries castling,
  en-passant and halfmove fields that have no geometry.
- **Renting CPU cores to escape the serial GPU.** Measured on this box: 70
  positions/s per core, 385/s across 8 processes. A 64-vCPU rental yields
  ~1,500-2,200 evals/s against the local GPU's ~3,200. The "91% CPU-bound"
  profile was taken *with the evaluator on the GPU*; renting cores does not rent
  an evaluator, it relocates a transformer onto the worst hardware for it.
- **Coupling `MIN_SPRT_GAMES` to `MIN_DECISIVE_PAIRS` before the instrument is
  honest.** It raises P(promote at +100 Elo) from 0.167 to 1.000, which is the
  wrong direction while the only live candidate is confounded.
- **Enabling `lichess_cloud_analysis`.** ~10 requests/game/side against
  lichess.org from the IP already running the bot, which the notes above record
  getting 429s at one request a minute. `chessdb_book` is fine: different host,
  different budget, but set `max_retries: 0`.
- **Applying a follow-up fix to the engine without re-running the boot test.**
  `search_engine.py`'s value loader was changed twice in one session. The first
  change was verified by piping `uci` into the engine and watching for `uciok`.
  The second added `dataclasses.fields(ModelConfig)` without adding
  `import dataclasses`, and was not re-tested because the first one had passed.
  `NameError` at boot, so the engine died before `uciok`, lichess-bot's
  `EngineTerminatedError` took the unit down, and `Restart=always` burned
  through the start limit into `failed`. **The live rated bot was down for
  ninety minutes and nothing said so** -- `systemctl is-active` had been checked
  earlier in the session and reported `active`, and no watchdog covers the bot's
  *unit state*, only its game stalls. Two rules: re-run the boot test after
  every edit to an engine entry point, not after the first one; and when the
  bot has gone quiet, check `systemctl is-active` before believing the last
  `is-active` you ran.
- **Reading a draw rate off a mirror match.** Two configurations of the same
  net playing each other drew 60% of games and hit threefold in 55%. That
  looked like an engine pathology and a council spent a round on it. Ground
  truth from `logs/games/`: 85 real lichess games, 3 threefold, 3 draws, **73
  checkmates**. 3.5%. A match between near-identical engines is structurally
  blind to everything they share and inflates everything they agree on. Any
  statistic about *how* SumoFish plays must come from real games; the harness
  only ever answers "did that change help".
- **Selecting `best.pt` by puzzle accuracy.** Sigma is +-1.5% at n=1000 and
  best-of-twenty on a noisy metric is biased upward by roughly two sigma.
  Measured on the finished 9M run: `final.pt` (300k) scores 2.1438 held-out
  against `best.pt`'s (280k) 2.1459, so the checkpoint that got promoted is the
  marginally worse of the two. `val_loss` is logged now; select on it.
- **A variance floor that decays with n.** `max(var, 0.25/n)` looks like the
  obvious regularisation and is wrong: the LLR numerator grows as n, so when
  the floor binds the LLR grows as n-squared and an all-draw match crosses the
  bound at n=34. Floor at a constant. And when you fix a statistical guard,
  grep for every function with the same expression -- the first version of this
  fixed `sprt_llr` and left the identical hole in `score_stats`, which is the
  one the promotion gate actually reads, where it produced a **zero-width 95%
  interval** on a 400-game match.
- **Raising `CHESSGPU_BATCH` on the strength of an nps number.** `nps` counts
  `len(boards)` sent to the GPU, and a batch of N descents into a root whose
  ~35 children are all still unexpanded can only reach ~35 distinct leaves.
  The rest are the same positions evaluated again. Measured on one midgame
  position, 6s per setting:

        batch   nps(raw)   unique/s   duplicated
           64       1760       1248        29.1%
          256       2468        507        79.5%
          512       2595        189        92.7%
         1024       2656         37        98.6%

  So "3813 nps at batch 1024" was 98.6% wasted work and a *collapse* in real
  search. Virtual loss is supposed to prevent this and cannot: it discourages a
  path, but with nothing below depth 1 to descend into, every walk in the batch
  lands on the same shallow frontier. **Batch must stay well under the
  branching factor times the depth the budget can reach.** The honest fix is to
  dedupe `pending` by leaf before the forward pass and back the shared value up
  to each path, at which point a larger batch buys throughput instead of
  repetition. Until then, report `unique/s`, never raw nps.
- Editing anything under `chessgpu/` while a match or the bot is running. Both
  import from the working tree and both start a fresh process per game, so an
  edit lands mid-match: the first half of the games played one engine and the
  second half played another, and nothing in the log says which. The result
  looks like a normal match and means nothing. Finish the match, or copy the
  tree, before touching the package.
- Calling `board.outcome(claim_draw=True)` anywhere inside a tree search. See
  `chessgpu/rules.py`, which exists entirely because of it. Short version: to
  decide whether a draw is *claimable* python-chess plays out every legal move,
  so it costs 573us and it answers a question about the children rather than
  about this position.
- "Optimising" `tokenize` itself. It is the function verified byte-exact
  against DeepMind's implementation, and every published number here depends on
  it staying that way. `tokenize_board` is the fast path and it is checked
  *against* `tokenize`, which is why both still exist.
- Expanding a reused root. `search()` used to call `_expand(root, board)`
  unconditionally, and `_expand` assigns `node.children[move] = Node(...)` for
  every legal move -- so re-rooting into a subtree and then expanding it wipes
  the subtree you just went to the trouble of keeping. The expand is
  conditional on `not root.expanded` for that reason and it is not optional.
- Concluding "launch-bound" from `nvidia-smi` showing a low GPU percentage. A
  low utilisation number cannot distinguish "the GPU is starved by kernel
  launch overhead" from "the GPU is idle because 95% of the work is happening
  on the CPU", and this project asserted the first for a whole session when the
  truth was the second. One cProfile over a 5-second search answered it: the
  network is 5% of the search, `can_claim_threefold_repetition` alone is 41%.
  Nearly a session of planning pointed at CUDA kernels that Amdahl caps at
  1.05x. Profile the whole loop, do not infer the bottleneck from one gauge.
- Reading `logs/engine.jsonl` as one stream. **Fixed at the source now**: the
  engine stamps `pid` and the lichess `game` id on every record
  (`chessgpu/telemetry.py`), the id arriving over UCI as `setoption name
  GameId` from lichess-bot's own `extra_game_handlers` hook (patches/0003).
  Deploy order matters: the engine must declare the option *before* lichess-bot
  sends it, or `engine.configure()` raises and closes the engine. The
  dashboard still carries the inference below, because it is what covers a game
  that started under an older engine process.
- Reading `logs/engine.jsonl` as one stream. `concurrency: 2` means lichess-bot
  plays two games at once, spawns an engine per game, and **both append to the
  same file**, with nothing in a record saying which game it came from.
  Measured in one log: 468 places where consecutive searches alternate between
  two positions. Read as one stream it put both games' plies on one evaluation
  curve -- a graph that says one side is winning when it is not, which is
  exactly how it was reported -- flipped the search panel between two boards,
  and let the picture jump to the other game. The stream's own position history
  is the only anchor that the other game cannot fake, plus the fact that the
  engine only searches on our turn, so `stm` alone separates them whenever the
  two games have us on opposite colours. Both live in `dash/fusion.EngineBoard`.
- A tailer that starts at the end of the file, with nothing to fill in what it
  missed. Correct for a log that is mostly other people's business, and it
  meant that attaching to a game in progress showed an empty search panel, an
  empty chart and a board that would not move until the engine's *next* move.
  Attach during a long think and it looks broken; attach onto a game whose
  opponent then flags and it never shows one thing about that game. "It opens
  and is frozen" is what that looks like from the outside, and it is
  indistinguishable from actually frozen. Read back over the log on attach.
- Rasterising the board at any size that is not a multiple of 26.
  `chess.svg.board` is a 390-unit square (8x45 plus a 15-unit margin), so a
  square edge lands on an exact pixel only at multiples of 390/15. Anywhere
  else rsvg blends the two square colours across the boundary and the board
  grows a faint grid it does not have. Counting pixels on a scanline that are
  neither square colour: eight per line at 1152, zero at 1144.
- `magick -colors N` with dithering left on. Error diffusion over a picture
  this flat buys nothing, is recomputed from scratch every render so the noise
  lands differently after every move (the board visibly shimmers), and is
  **three times slower**: 286ms against 96ms at 1144px. `-dither None -colors
  64` is 100ms and cleaner. That 400ms encode the whole off-thread renderer was
  built around was mostly dithering.
- Seeking backwards on a text-mode file handle. `seek(-n, SEEK_CUR)` raises
  `io.UnsupportedOperation: can't do nonzero cur-relative seeks`, because a
  text handle's position is an opaque cookie. The tailer did this to hand a
  half-written line back to the writer, so every torn write raised, lost the
  record, and left the handle mid-line so the next read lost another one. Open
  the file `rb` and decode after splitting.
- `/api/game/export/{id}` is a 404; the endpoint is `/game/export/{id}`. And it
  serves PGN unless you send `Accept: application/json`, which `json.load` then
  rejects as a network error. Both together meant the finished-game summary
  never appeared once. Worse, the source only remembered a game id after a
  *successful* fetch, so it re-asked every four seconds forever, from the
  address the bot plays from.
- Iterating a dict that a source thread writes to. `moves_panel` walked
  `state.curve` directly while `record_eval` inserted into it, which can raise
  `dictionary changed size during iteration` inside `draw` -- and nothing
  catches that, so the dashboard exits mid-game. Snapshot under the lock.
- Treating `nowPlaying[0]` as "the game we are watching". With two games in
  progress lichess orders that list however it likes and the order changes on
  its own, so the board could swap games mid-game and swap back. Stay on the
  current game while it is still in the list.
- Using one number both for the size the dashboard *asks* the terminal for and
  the size below which it gives up on the full layout. They are different
  questions, and at 149 columns -- one short of the 150 it asks for, on a
  window already as wide as that monitor gets -- the answer was a 46-column
  text board, a fifty-row empty search panel and no evaluation panel at all.
  `WANT_COLS` is the ask, `MIN_WIDE_COLS` is the floor, and the floor is what
  the layout actually needs (right-hand column plus a board worth drawing).
- Re-emitting the board image after a resize without checking it was drawn for
  the *current* layout. The plan rebuilds instantly and the fresh render takes
  ~400ms, so the frame in between paints the previous, larger board over a
  screen that was just wiped for the new one -- and its bottom and right edges
  land in rows the new layout never writes to, where nothing will ever repaint
  them. On screen: a strip of squares and half a pawn under the player line,
  surviving every frame afterwards. The size is already part of the image key;
  compare it before emitting, and accept the board being briefly absent rather
  than briefly wrong. `ESC[2J` on a geometry change clears whatever did get
  stranded (verified: it does erase image data in Konsole).
- `pkill -f "scripts/watch.py"` from a shell whose own command line contains
  that string kills the shell, and `-f "watch.py"` also matches the *live*
  dashboard, not just the fixture you started. Both happened in one session:
  the second one closed the window the user was watching the bot in. Match on
  something specific, **and bracket a character in it**: `pkill -f "watch.py
  --demo"` is specific enough to spare the live one and still kills its own
  shell (exit 144, nothing else happens). `pkill -f "watch[.]py --demo"` is
  the version that works.
- `nohup konsole ... &` and then killing `$!`. Konsole does not run one process
  per window here -- `ps` shows three `/usr/bin/konsole` processes for six
  windows -- so a new invocation hands its window to an existing instance and
  the pid `$!` gave you has already exited. Killing it is at best a no-op and
  at worst takes windows you did not open: one session lost the *live*
  dashboard to that kill and could not prove what it hit. Never kill a konsole
  process. Kill the program inside the window (`pkill -f "watch[.]py --demo"`)
  or close the window with `kdotool windowclose`.
- Driving the GUI to check a layout while other sessions are working in this
  repo. Windows opened for a screenshot get closed by somebody else, `kdotool
  search` answers differently on consecutive calls, and `CLAUDE.md` changes on
  disk between two edits of it. Half a session went into chasing which of those
  was a bug of its own making. `tests/verify_layout.py` exists so the geometry
  can be checked without a terminal at all; screenshot only to judge how it
  *looks*, never to check that the numbers add up.
- `CSI 8;rows;cols t` written into another process's pty from outside
  (`/dev/pts/N` or `/proc/PID/fd/1`) does not resize the window, though the
  same sequence printed by a shell running *in* that terminal does. Do not
  spend time on it: launch the program in a terminal that is already the size
  you want.
- Asking Konsole for more rows than the window can hold. `CSI 8;rows;cols t`
  with 96 rows on a 1440px screen leaves the window at the screen height and
  the app is told about rows that do not exist, so `rich` renders a screen
  taller than the terminal and the top of it scrolls away: the header panel
  simply is not there, and the whole layout sits three rows high. It looks
  exactly like a layout bug in the dashboard and is not one. 94 rows fits.
- `matchmaking.challenge_timeout: 1` is a trap. Matchmaking calls
  `/api/user/{name}` to size up each candidate opponent, that endpoint has a
  small per-IP budget, and once a minute exhausts it. **Every retry against a
  429 renews the penalty**, so the bot sits unable to create a single challenge
  indefinitely rather than recovering: 40 minutes of "No challenge will be
  created" in a row. Verified blocked for both authenticated and anonymous
  requests while `/api/account` stayed fine, so it is the endpoint that is
  limited, not the credential. 5 minutes recovers. Back off, do not retry.
- `GET /api/user/{name}` returns **429 from this machine even at one request a
  minute**, because lichess-bot is already talking to lichess from the same IP
  and the public per-IP budget is shared and small. The dashboard polled it and
  showed a permanently empty rating panel with no error. Use `/api/account`
  with the token: identical shape for our own account, separate budget. Keep
  the public endpoint only as a fallback.
- A fifo or unix socket as the engine -> dashboard telemetry channel: `open()`
  for writing **blocks until a reader attaches**, so a dashboard that is not
  running becomes an unbounded stall inside `choose()` against a running chess
  clock. Append-only JSONL has no such failure mode. The engine must never be
  able to block on whether a spectator exists.
- Tailing `journalctl --user -u chess-gpu-bot -f` as the primary telemetry
  source: it interleaves engine stderr with lichess-bot's own logging, a unit
  restart silently breaks the follow with no reconnect, and a pipe read that is
  not strictly line buffered can hand back half a record. Fine for eyeballing,
  wrong as a data path.
- Trusting lichess's **public** game stream for the live position. Measured on
  a live game over thirteen consecutive plies: it runs a median **8.7s behind**
  the engine's own view, very consistently (8.4-9.1s). That is the feed, not
  this code, and it cannot be tuned away from this side. The engine's telemetry
  carries the FEN it is searching -- the position *after* the opponent moved --
  is local, and is instant. Drive the board from that (`dash/fusion.py`) and
  keep the stream for the clocks, the move list and the record.
  Corollary for measuring: comparing your parser against a second read of the
  same feed tells you nothing about the feed's own lag. That mistake cost two
  rounds of optimising a 400ms render while an 8700ms delay sat upstream.
- Encoding the board image on the render loop. It costs ~400ms, essentially
  all of it in ImageMagick's sixel encoder (measured: svg 2ms, rsvg 54ms,
  sixel-encode 366ms at 1150px). Fine for one move, and quietly catastrophic
  for bullet: the bot moves faster than 400ms, so every move queued behind the
  last and the board fell further behind for the whole game without ever
  catching up. `sixel.Renderer` does it on its own thread and always renders
  the *newest* requested position, dropping anything that went stale while it
  worked, which bounds the lag at one render however fast the game is.
  Verified: 16 positions requested at 100ms intervals, renderer settles on the
  final one.
- `img2sixel` is 20x faster than ImageMagick here (19ms vs 358ms) and this
  build (libsixel 1.10.5) exits 0 while writing nothing at all, to a pipe or
  to `-o`. Do not spend time on it again without checking that first.
- Ignoring "Only make one request at a time", which the lichess API docs state
  outright. The docs publish no numeric limits at all -- "various strategies",
  and "some limits may require longer" than the usual one-minute wait -- so
  that sentence is the only concrete guidance there is, and separate polling
  threads each with a request in flight breaks it. `sources.GATE` serialises
  every call and paces them 250ms apart; the game stream holds it only while
  connecting, never while reading. The machine panel shows `api N/min` so the
  answer to "am I asking too often" is on screen rather than inferred.
- Publishing every message the lichess game stream sends. It replays the whole
  history on connect, so each reconnect walks the board from move one to the
  present again -- and a game ending closes the stream, so this fires exactly
  when a game is decided. On screen it is a flicker through some earlier
  position, gone before it can be read. The replay is still needed to rebuild
  the move list after a drop; it just must not be shown. `GameStream` tracks
  the furthest ply published per game id and skips anything behind it.
- Tailing a file by path alone: `logs/engine.jsonl` rotates at 16MB, and a
  tailer that does not track the **inode** goes quiet forever after the first
  rotation while continuing to render its last value, which looks exactly like
  a quiet game. `dash/sources.Tailer` reopens on inode change and on truncation.
- Converting a White-framed probability to our side's at each call site. One
  will be missed. With SumoFish playing Black and being mated, the move list
  read 0.97 while the chart read 0.03: both correct in their own frame, and the
  one that happened to be White's said we were winning. `panels.ours()` is the
  single conversion; everything that shows a probability goes through it.
- A gauge whose empty half is the same colour as the panel behind it. The
  ground was #32302f on a #282828 panel: 1.12:1, which is nothing. A bar
  reading 95% then looks like a stripe floating in space rather than a bar
  filled nearly to the top, and there is no way to tell a full gauge from an
  absent one. Measure the contrast; a gauge needs its extent visible, not just
  its level.
- White-and-black ink on a gauge that measures *our* side rather than White's.
  The colours carry chess meaning that contradicts the number: playing Black
  and losing gives a mostly-dark bar, which reads as "Black is winning". Use a
  colour that means only what you intend.
- Plotting `win_prob` or any Q straight from the telemetry: MCTS stores values
  from each node's **own side-to-move perspective**, so a series across
  alternating plies is a sawtooth, not a trend. Convert to a fixed frame first.
  The records carry both `wp` and `wp_white` so no consumer has to remember.
- A chess board as one glyph in a 3-wide cell: terminal cells are ~1:2 (8.0 x
  14.9 px here), so the square is always a squashed rectangle and no glyph
  fixes it. Two cells wide per one tall is square; `▀` half-blocks give two
  vertically stacked pixels per cell, so 8x4 cells is an 8x8 pixel sprite.
  Bonus: block elements have no `wcwidth` ambiguity, while ♞ and the Nerd Font
  codepoints do, and that ambiguity drifts table borders row by row.
- Tinting a check square's background: the king covers almost all of its own
  square, so the warning hides behind the piece it is about. Draw a ring on the
  square's outer pixels, over everything.
- Assuming a terminal cannot do something without asking it. Sixel was ruled
  out here on the belief that Konsole's support is off by default and
  unreliable. It is neither: `ESC[c` returns `[?62;1;4c` and images render with
  no setup. That wrong assumption cost the entire half-block renderer, which is
  now only the fallback path.
- `sys.stdin.read(1)` to read a terminal's reply to a query. Python's text
  layer buffers: it pulls the whole reply off the fd, returns one character and
  keeps the rest, so `select` reports nothing pending and the loop exits with a
  bare ESC. The remainder then surfaces inside the *next* query's answer, which
  reads exactly like a terminal that does not support the query. Use
  `os.read(fd, n)`.
- Putting anything that *animates* anywhere in the board column, even on rows
  the image does not occupy. Two separate corruption bugs came from trying, and
  the geometry check that says a row is clear does not make the column a good
  place for it: the image is re-emitted only when the position changes, so
  anything that redraws between moves is one rich quirk away from eating it.
  Put moving things in a text panel. The evaluation gauge lives in the search
  panel, beside the number it is a picture of.
- Putting *anything* styled on a row the sixel image occupies, even far to its
  left. `rich` diffs by line: change one cell and it rewrites the line, which
  means writing the unstyled padding over the picture. A static styled cell
  survives because it never changes; an animated one erases the board a row at
  a time and the next move snaps it back, which looks like the board shifting
  every move. The evaluation gauge is horizontal and sits on its own row
  underneath for exactly this reason.
- Drawing a sixel image under any styled cell. `rich` skips writing cells that
  carry no style and rewrites ones that do, so a Panel or a console-wide
  background over the image region erases the picture between frames -- it
  never appears at all, which looks like sixel being broken. The region must be
  bare: no panel, no background, unstyled padding only.
- Re-emitting an unchanged sixel image on a timer. The terminal clears the
  region before redrawing, so the board strobes at the frame rate. Emit only
  when the position actually changes; `rich` leaves the region alone in between.
- Writing raw bytes to `sys.stdout.buffer` without flushing `sys.stdout` first.
  Rich's pending frame is still in the text wrapper and gets flushed after the
  image, painting over it.
- Hand-drawing chess piece sprites at all: `python-chess` ships the cburnett
  SVGs, which is the exact set lichess renders, so rasterising those is both
  less work and a better result than any silhouette drawn by hand. The
  hand-drawn 8x8 set that preceded it is gone; do not reintroduce one.
- Widening cburnett's stroke so the outline survives at 16px: every path in the
  set carries the same `stroke-width`, so widening it closes the king's crown
  and swallows the fill. Measured: at stroke 2.2 and 3.1 the king loses its
  cross entirely. Recover the outline at display time with a contrast stretch
  on luminance instead, and leave alpha alone -- alpha is the silhouette, and
  hardening it makes the edges jagged.
- Random-access `Dataset.__getitem__` over the 36GB bag: **965 rec/s**, vs 1.39M
  sequential. Every lookup is a disk seek on a file too big for page cache.
  ChessBench is already shuffled on disk, so stream sequentially through a
  shuffle buffer instead. Do not reintroduce a shuffling sampler.
- Benchmarking a cold mmap gives nonsense — decode-only measured *slower* than
  decode+tokenize because the second pass hit warm pages. Warm the cache or
  measure sequential and random separately.
- `StartLimitIntervalSec` / `StartLimitBurst` in `[Service]`: silently ignored,
  logged only as "Unknown key". They are `[Unit]` keys. The rate limit you think
  you set is not in effect.
- `ProtectHome=read-only` on the bot unit crash-loops lichess-bot: it creates
  `lichess_bot_auto_logs/` in its working dir. That dir also holds per-game
  takeback state, so `--disable_auto_logging` trades a loud startup crash for a
  quiet runtime one. Add it to `ReadWritePaths` instead.
- Do not infer upstream's loss semantics from variable names. Their mask is
  `[True]*77 + [False]` and True means *excluded*. Read `training_utils.py`.
- Batch size >1024 OOMs and buys nothing: training is launch-bound, throughput
  is flat 512→1024. The failing 327MB allocation is the SwiGLU intermediate.
- `pgrep -f "train.py"` matches all ten dataloader workers. Use `| head -1`.
- Python 3.14 is too new for the ML wheels. The venv is 3.12 deliberately.
- `bc` is not installed on this box. Use `awk` for shell arithmetic.
- Puzzle accuracy is worthless as a signal on short runs — a 261k model scored
  0/300 while already predicting moves 35x better than chance. Use bits-per-move
  below ~30 min of training.

## Faster next time

- One cProfile beats an afternoon of reasoning about where the time goes. The
  whole search-optimisation plan in this file was pointed at CUDA kernels on
  the strength of a GPU utilisation percentage; a single 5-second profile
  showed the network was 5% of the search and moved the entire plan. Profile
  first, and profile the *whole* loop rather than the part you suspect.
- Build the measurement before the thing being measured. Every speedup in
  session 3 could be verified by a stopwatch, but "is it still as strong?"
  could not be asked at all until `scripts/match.py` existed, and writing it
  first meant the changes landed with an answer instead of a hope.
- Measure the I/O pattern *before* designing a data loader. One 20-line
  benchmark would have skipped an entire wrong design.
- When porting a reference implementation, exec the original and diff against it
  rather than reading it carefully and reimplementing. Catches what careful
  reading misses, and it is less work.
- Pin evaluation metrics at both ends before trusting them: the puzzle evaluator
  was only believable once an oracle scored 500/500 and random scored 0.8%.
- `pgrep -f "<pattern>"` where the pattern also appears in the shell command that
  runs it matches **its own shell**, so a `while pgrep ...; do sleep; done` wait
  loop never exits. Cost a 10-minute timeout and silently skipped the work that
  was queued after it. Match on something the caller does not contain, or check
  with `ps aux | grep -v grep`.
- `git commit -m "..."` with backticks in the message: bash runs them as command
  substitution and silently deletes that text from the commit. Write the message
  to a file and use `-F`, or use a quoted heredoc.
- MCTS sign convention: values are stored from each node's OWN side-to-move
  perspective, so a parent scoring a child must use `1 - child.q`. Getting this
  wrong made checkmate the worst-scoring move on the board (q=0.0) and the
  search played a random king move instead, 103 visits to 52. Nothing crashed,
  no test failed. The file's own docstring warned about it three paragraphs
  above the line that had the bug.
- Do not compare `active.json`'s pid against the unit's `MainPID`: ExecStart
  runs python under `systemd-inhibit`, so MainPID is the wrapper and never
  matches. That check disabled the watchdog on every poll -- a fix that
  silently removed the supervision it was added to repair.
- Changing prediction target does NOT mean starting from scratch. Only the
  output layer changes shape; 91 of 93 tensors transfer. Warm-starting was
  worth +15.3 puzzle points at matched steps (and the warm arm saw HALF the
  data per step). Always `--init-from` the best existing body.
- Optimise in order: correctness, then algorithm, then kernels. Batching the
  MCTS was 19x and needed no CUDA. Writing kernels first would have
  hand-tuned the inner loop of a design that was 19x off.
- lichess starts every BOT at a placeholder 3000 rating that CASUAL games can
  never move, so matchmaking hunts 2500-3500 engines forever. The bot went
  1-19 before rated play was enabled. Rated is the only thing that corrects it.
- lichess-bot's `challenge_mode` governs OUTGOING challenges separately from
  `challenge.modes` for incoming. Setting one and not the other means it
  accepts rated games but only ever issues casual ones.
- `git commit -m "..."` with backticks in the message: bash runs them as
  command substitution and silently deletes that text. Use `-F` with a file or
  a quoted heredoc.
