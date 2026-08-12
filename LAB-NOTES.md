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
- `pkill -f "<the dashboard>"` from a shell whose own command line contains
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

## 2026-07-29, afternoon: the speed flags, priced

- **`dedup` + `compile` at a fixed clock cost -168 Elo** (20 games, W0 D11 L9,
  LOS 0.0%). Both were byte-identity-preserving at a fixed *simulation count*
  and both were genuinely faster per call, so the prediction was "clearly
  positive". The prediction was backwards, and the diagnostic says why: at 0.5s
  the fast arm ran 7,297 nominal simulations to plain's 4,161, but got **3,464
  unique evaluations to plain's 4,160**. It bought 1.75x the claimed search and
  17% LESS knowledge of the position. Dedup frees network time, the search
  spends it on more descents, and the extra descents collapse onto leaves
  already evaluated. Every duplicate still backs up a value, so visits and Q
  inflate on no new information.
- **Therefore: an identity proof at fixed simulations says nothing about
  strength at fixed time.** The two are different experiments. Anything that
  changes the sim/second ratio has to be matched on the clock, in a game, even
  when the tree is provably unchanged. "Provably identical" licensed the plain
  port with no match; it did not license these flags, and I treated the two the
  same.
- **Never report nps, sims/s, or "simulations" as a measure of search.** The
  unit is **unique evaluations per second**. Both engines print `evaluations`
  and `unique_evaluations`; the gap between them is the part that is not search.
- Do not run a `--time` match while the bot is live. 20 games were discarded
  because arm A and arm B shared the GPU with rated games, which biases a
  wall-clock experiment asymmetrically and only that kind. Stop
  `chess-gpu-bot` and `chess-gpu-lab` first, and record in the match config
  that they were down.
- CUDA streams are **not** a substitute for a shared trunk: two streams alone
  is 1.04x (nothing), and 1.19x on top of `compile`. Halving the launches
  honestly costs the ~40 GPU-hours of training a two-head net needs.
- Do not extrapolate a speedup from a profile taken before the previous
  speedup. The "8.9x" for batching came from the Python profile where the
  network was 9% of the search; with the tree in Rust the network is 89% and
  the same change is worth a fraction of that. Re-profile after every port.

## Faster next time, 2026-07-29

- The order that worked: port for speed with an identity proof (no GPU, no
  match, unfalsifiable-by-noise), then measure anything that changes the time
  budget with a real match. The order that wasted a day: build an exchange
  rate out of replayed matches and then reason from it.
- When a flag is "obviously" a win, the cheapest disproof is usually a counter
  already in the code. `unique_evaluations` existed the whole time and would
  have killed the fast config in 30 seconds instead of 20 games and an
  afternoon.
- `mate_distance` measured with the REAL prior for the first time: 34/34
  shortest mates found with the fix OFF and 34/34 with it ON, 34 sound proofs,
  0 bogus, tree 51% smaller. Both arms are **saturated**, so the suite has no
  headroom and cannot price the fix's effect on move choice. That is a limit of
  the suite, not a verdict on the fix -- mates in 1-2 are found unaided by a
  trained policy. A discriminating suite needs mates in 3-5, at which point the
  exhaustive `python-chess` solver is too slow to be the ground truth and
  Stockfish at pinned depth has to be. Do not read "unchanged" as "useless"; the
  51% tree reduction and the soundness of the proofs are the justification.
- `chess-gpu-rust` and `chess-gpu-instrument` are **git worktrees of the same
  repo**, not copies, so each has its own `chessgpu/` at its own branch. The
  `rust-core` worktree predates `chessgpu/rust_mcts.py`, so a test there that
  imports the adapter fails with `ModuleNotFoundError` while the file plainly
  exists in `~/chess-gpu`. Worse, `sys.path` resolution silently picks whichever
  `chessgpu` comes first, so an identity oracle can compare the port against a
  *different branch's* engine and still pass. `tests/verify_mate.py::real_evaluator`
  now diffs the two copies and refuses if they have drifted. Any new
  cross-worktree import needs the same guard.

## 2026-07-29, evening: the harness produces its first true positive

- **The match harness had never been shown to detect anything.** Every result
  in the archive was a replay, a null, or the -168 Elo rejection. "The
  instrument works" was an assumption. It is now tested: `9M-sv@10k` vs
  `9M-sv@280k`, same policy both sides, 400 sims, concluded **W1 D23 L12,
  -109.7 Elo +-64.1, LOS 0.0%**, SPRT terminating correctly in 36 games. Right
  sign, large effect, efficient stop. Run a known-large positive control
  BEFORE trusting an instrument to price a small change, not after.
- That is also the project's first training-Elo datapoint: **~56x more training
  is worth roughly +110 Elo** (5.12M samples vs 287M). State the confound with
  it, always: the two checkpoints are from different runs, sharing donor body,
  LR, data and target, but differing in batch size (512 vs 1024) and warmup
  (500 vs 2000). It is a positive control with a number attached, not a clean
  scaling point.
- **64% of those games were draws** (23 of 36), which is the resolution
  limiter. Both arms share the policy prior, so they open alike and the
  positions correlate. Any future match between two nets with a common prior
  needs more games than the Elo formula suggests, or sharper book openings.
- `train.py` wrote only `latest.pt` (a moving pointer) and `best.pt` (whichever
  eval got lucky), so a finished run left two checkpoints and no ladder. That is
  why the only comparable points for the above were 10k and 280k from different
  runs. `--keep-every` now retains step-tagged copies. Retention decisions have
  to be made BEFORE the GPU-hours are spent; there is no way to recover a
  checkpoint a run declined to write.

## 2026-07-30, restarting the live bot to deploy a fix

- **`systemctl --user kill --signal=SIGINT chess-gpu-bot` sends the signal to
  every process in the unit's cgroup, not just the main process** -- it hit
  the live game's engine subprocess (a `chessgpu.engines.search_engine` UCI
  process, mid-search) as well as `lichess-bot.py` itself, and the engine
  crashed with `KeyboardInterrupt` out of its `for raw in sys.stdin:` loop.
  Intent was to trigger `lichess_bot.py`'s own graceful drain
  (`quit_after_all_games_finish: true` blocks in `pool.join()` on SIGINT
  rather than dropping the game) without also invoking `systemctl stop`'s
  `TimeoutStopSec=180`, which is too short for a 15+0 game that hasn't
  reached its increment-heavy endgame yet. No game was actually lost --
  `engine_wrapper.py`'s own `chess.engine` layer caught
  `EngineTerminatedError`, backed off 0.7s, and respawned a fresh engine
  process that picked the same game back up at the same move -- but that
  recovery was luck (an already-existing resilience path), not the intended
  behaviour. `systemctl kill --kill-who=main --signal=SIGINT <unit>` is the
  correct call: it signals only the main PID, leaving whatever
  `multiprocessing.Pool` workers (including the live engine subprocess) are
  mid-game untouched, exactly what `quit_after_all_games_finish` is designed
  to protect.

## 2026-07-31: a port has to sweep the DEFAULTS of everything downstream

- **The Rust port silently broke three instruments by leaving their defaults
  pointed at Python.** `LAB-NOTES` already said "re-profile after every port".
  That was too narrow. The port changed which core is *normal*, and every tool
  that consumes the core kept its own private idea of normal:

  | instrument | stale default | what it cost |
  |---|---|---|
  | `rust_mcts.select_mcts_class` | defaulted Python | fixed 07-31, was the known one |
  | `scripts/bench_search.py` | imported `sumofish.mcts` outright, never consulted the selector | `scale_m` measured 1.7x too CHEAP |
  | `scripts/match.py:664` | `--core` defaulted `"python"` | every fixed-TIME match since 07-30 measured a 2.7x-slower engine |

- **Why `match.py` survived nine months of use is the transferable part.** The
  failure is *invisible* in a fixed-SIMULATION match -- the identity proof means
  both cores build the same tree, so the result is genuinely unaffected -- and
  *total* in a fixed-TIME match, where one arm simply gets 2.7x less search than
  the deployed engine. So the bug hid in the experiment that is run constantly
  and only bit the one that decides promotions. **When a flag is inert in the
  common experiment and decisive in the rare one, its default gets no testing
  from use.** Audit those deliberately; nothing else will.

- **`m` is not a property of the model, it is the model divided by the tree
  around it.** This is why the stale `scale_m` mattered and why the direction is
  counterintuitive: making the TREE faster makes every future model scale-up more
  EXPENSIVE, because the dearer forward pass is no longer diluted by anything.
  Measured on an idle box at batch 64 (`runs/lab/profile-2026-07-31.json`):

        core     9M nps   136M nps   m      doublings lost
        python    3,041      1,706   1.78x  0.83
        rust      7,763      2,537   3.06x  1.61

  The Rust port did not merely fail to help the case for a bigger net. It
  roughly doubled the search a bigger net has to pay for. And `m` is an INPUT to
  the port's own justification rather than an output of it, so nothing in the
  port's verification could have caught this.

- **Network share, measured directly rather than inferred** (time the evaluator
  alone, divide by per-node search time): Rust **~100%** (102.2%, tree -2.2%),
  Python **38.2%**. The long-quoted "9% in Python" is wrong by ~4x and every
  extrapolation that used it as a denominator inherited that. The `89.3%` for
  Rust is essentially right. Over 100% is not a paradox: a synthetic full batch
  of 64 costs marginally more than the search's real ragged batches, so it means
  the tree term is below the noise, ~5% here.

- **Neither of those two numbers had an artifact behind it** -- both existed only
  as prose in `STATE.md` and this file. That is how a wrong one survives next to
  a right one. A profile figure quoted in prose with no JSON beside it should be
  read as a rumour.

- **`code_fingerprint()` hashes `sumofish/**/*.py` plus the git SHA, and NOT
  `scripts/match.py`.** So editing the harness mid-match does not disturb a
  running match's provenance, but it also means the fingerprint would not catch
  a harness edit that changed how games are played. Worth knowing in both
  directions before editing anything mid-experiment.

## 2026-08-01: draining the bot is two commands, not one

- **`systemctl kill --kill-who=main --signal=SIGINT sumofish-bot` drains the bot
  correctly and then systemd puts it straight back.** The unit is
  `Restart=always` with `RestartUSec=10s`, and `kill` signals the process without
  marking the UNIT stopped, so the clean shutdown looks exactly like a crash to
  systemd: `Scheduled restart job, restart counter is at 1` ten seconds later.
  Bit me tonight -- the bot came back at 03:14:22 and spent several minutes
  contending with a training run that had just been launched precisely because
  the bot was supposed to be down.
- The earlier note recommending `systemctl kill --kill-who=main` over
  `systemctl stop` is still right about *why* (stop signals the whole cgroup and
  takes the engine subprocess out mid-game, abandoning a live rated game). It was
  incomplete. **The correct sequence is BOTH, in order**: `kill --kill-who=main
  --signal=SIGINT` to drain, wait for the games to finish and the main pid to
  exit, then `systemctl stop` to keep it down. Checking `is-active` once right
  after the drain is not enough -- the restart lands 10 seconds later, so a check
  that runs immediately sees `inactive` and reports success.
- Generalisation: **`systemctl kill` and `systemctl stop` answer different
  questions.** `kill` is "signal this process"; `stop` is "I intend this unit to
  be down". Only the second one survives a `Restart=` policy. Any unit with
  `Restart=always` cannot be taken down by signalling alone, no matter how
  gracefully.
- Unrelated but from the same hour: conceding a live rated game to free the GPU
  puts a loss in `logs/rating.jsonl` that is **indistinguishable from a loss the
  engine earned**. Two of them cost ~14 rapid Elo at RD 45 and will silently
  contaminate any before/after comparison spanning that moment. If a game must be
  ended early for machine time, record the game ids and the timestamp so the
  rating can be read around them.

- **A warm restart gets visibly WORSE for about the first quarter of the run, and
  that is not a failure.** Restarting a converged checkpoint at a fresh cosine
  schedule knocks it off its annealed minimum: the LR warms back up to near the
  donor's peak, the solution is perturbed, and it re-anneals only as the schedule
  decays. Measured twice in this repo, at two targets, with matching shapes:

        run                  lr     baseline      worst dip        final
        9M-sv-continue (SV)  3e-4   0.687 puz     0.660 (-2.7)     0.700 (+1.3)
        9M-bc-2026-08-01     2e-4   0.409 puz     0.381 (-2.8) ... pending

  The value run stayed BELOW its donor on held-out loss from step 305k to ~380k
  -- 75k steps, 27% of the run -- peaking at +0.027 worse, and then finished
  -0.032 BETTER. That is the checkpoint now deployed. So: do not judge a warm
  restart before ~30% of its schedule has elapsed, and do not kill one on an
  early eval. The first eval of `9M-bc-2026-08-01` at step 10k was -2.8 puzzle
  points and looked exactly like a botched learning rate.
- Corollary for reading these runs: **train loss above the donor's is expected
  during the dip too** (1.84 vs 1.754 here) and is not independent evidence of
  anything. Both numbers move together because both are measuring the same
  perturbation.
- The genuine failure mode this resembles -- LR too high for a warm start, model
  never recovers -- is distinguished by WHERE the curve is at ~30-40% of the
  schedule, not by how bad the early evals look. Check there, not at step 10k.
- **I then wrote a health check for that run keyed on PUZZLE ACCURACY and it
  returned the wrong verdict.** At step 90k it printed "STILL BELOW BASELINE --
  lr likely too high" off a 0.405 -> 0.386 move, which is **1.23 sigma** on a
  metric whose sigma is 1.55 points at n=1000. Held-out loss over the same span
  went 1.6966 -> 1.6701, monotonically, never once the wrong way. PHILOSOPHY
  already says "select on held-out loss, not on a noisy eval"; the lesson is that
  this applies to AUTOMATED MONITORS too, not just to promotions. A check that
  can cry wolf at 1.2 sigma will eventually kill a good run at 3am with nobody
  awake to overrule it. Gate monitors on the same metric you would gate a
  decision on.
- **`runs/9M-causal` (the BC donor, source of the live `runs/policy.pt`) has 31
  puzzle evals and ZERO val_loss.** So the policy net has no held-out-loss
  history at all, and any "is the retrain better" question is forced onto the
  +-1.5% metric or onto a match. The value net does not have this problem. Fix by
  evaluating `policy.pt` against the same held-out set the new run uses, so the
  comparison is like-for-like; until then, do not select a policy checkpoint the
  way the value checkpoints were selected, because the evidence is not the same
  kind.

## 2026-08-05: two ways to make a thing vanish while it is still running

- **`systemctl --user disable <unit>` deletes the unit file when the unit file is
  a symlink.** Every unit here is a symlink from `~/.config/systemd/user/` into
  the repo's `systemd/`, and `disable` removes *all* symlinks pointing at the
  unit, not just the `default.target.wants/` one that makes it autostart. Ran it
  on `sumofish-bot` to stop it coming back after a reboot; `is-enabled` then
  answered **`not-found`** with the bot still playing two rated games. The unit
  had not been disabled, it had been uninstalled out from under a live process.
  Restore is `ln -sf <repo>/systemd/<unit> ~/.config/systemd/user/<unit>` plus
  `daemon-reload`, after which it reads `linked`, which is the state you actually
  wanted. To turn off autostart non-destructively, remove only the
  `default.target.wants/<unit>` symlink.
- **The near-miss that made it dangerous:** a drain supervisor was polling
  `systemctl --user show -p MainPID --value sumofish-bot` and treating an empty
  answer as "the main pid is gone, safe to `stop`". A missing unit returns an
  empty string too. For ten seconds the supervisor's test could not distinguish
  "the bot finished its games" from "the unit no longer exists", and its next
  action would have been to take down a bot mid-game. Any liveness check on a
  systemd unit must test that the unit EXISTS separately from what its MainPID
  says; `is-active` and `MainPID` answer different questions and a vanished unit
  fails both in the same direction as a clean exit.
- **The 136M sweep arm has now died to a reboot twice** (2026-08-02 23:23) and
  each death cost the whole run, because `sweep_argv` sets `--ckpt-every 20000`
  and `--auto-resume` reads only `latest.pt`: at 5,000 steps there was nothing on
  disk to resume from. Lowering `--ckpt-every` for this arm alone was rejected --
  the sweep's entire claim is that the arms differ in width and nothing else --
  so the mitigation is at the unit layer: `sumofish-sweep-136m.service`, enabled,
  so a reboot restarts the arm rather than leaving `runs/lab/state.json` parked
  on `current: sweep-136m` with a dead pid, which is what it did for three days.
  **The lab does not notice a runner that died with the machine.** Check its pid
  against `/proc` before believing the queue is alive.
- **A log line that is not the engine's can silently become one of the engine's
  moves.** The dashboard's journal parser reassembles lichess-bot's wrapped
  `Got move <uci> ... for game <id>` out of an 8-line window when it sees the
  `Source: Lichess EGTB` marker on a later line. A spectator chat greeting logged
  between the two wraps to seven lines, pushes the move record out of the window,
  and the parser then took the first token of whatever was left -- `***`, the
  redacted chat line -- and paired it with the PREVIOUS move's game id and
  wdl/dtz/dtm. A real game got a move it never played, with plausible numbers on
  it. Once in 623 tablebase moves over 36 hours. The fix is to decline rather
  than guess (`Event::TablebaseUnattributed`); widening the window converts
  fabrication into stale attribution, which is the same error with no tell.
  Found only because `the_real_bot_journal_parses` runs the parser over the
  machine's actual journal instead of over fixtures someone wrote. **Keep at
  least one test whose input is real production data**; the hand-written fixtures
  had passed this whole time.
- **A drained lichess-bot does not converge, because the matchmaker ignores the
  drain.** `systemctl kill --kill-who=main --signal=SIGINT` gets you "Waiting for
  games to finish before quitting", and then `matchmaking.py` goes right on
  issuing OUTGOING challenges at `challenge_timeout: 1`. Fifteen minutes into a
  drain the bot had gone from two live games to three, one started ten minutes
  after the SIGINT. So the earlier note's "wait for the main pid to exit, then
  `stop`" is a wait with no end while matchmaking is on, and `stop` is clean only
  at zero games. **Draining this bot for real means `allow_matchmaking: false`
  first, which needs a restart, which abandons the games the drain existed to
  protect.** Plan the restart for a moment you have already chosen instead.
- **And check whether the thing you are draining for actually cares.** The drain
  above was to give a training run an uncontended GPU. Training's output at a
  fixed step count is a function of data, order and seed; contention changes the
  wall clock and nothing else. Worse, the two arms this one is compared against
  were themselves trained while the bot played, so an idle box would have made
  the third arm the odd one out. Cost: 15 minutes and a deleted unit file, for a
  premise that inverted on ten seconds of thought about what the metric is.
- **`train.py` opens `log.jsonl` with mode `"a"`, so a re-run of a dead run
  appends to the corpse.** Restarting `sweep-136M` left the log holding the
  previous attempt's steps 500-5,000 followed by the new run's step 500, and
  `train_progress` reads `steps[-1]`, so for the first ten minutes the lab board
  would have reported "step 5,000 val 2.8646" for a run that had just started.
  `best.pt` is the same hazard with worse consequences: it survives from the dead
  run until the new run's first eval beats it, so between them the file is a
  different experiment's checkpoint under the current experiment's name. Move
  both aside before relaunching (`log.jsonl.dead-<date>`, `best.pt.dead-<date>`).

## 2026-08-05: chasing the ladder's flat first rung, and failing honestly

- **The re-earned exchange ladder is non-monotonic and it is still unexplained.**
  200 -> 400 sims is worth +29.0 +-25.1 Elo, while 400 -> 800 is +240.8,
  800 -> 1600 is +308.2 and 1600 -> 3200 is +233.7. Each rung's `config.json` is
  identical apart from the two sim counts, so it is not a difference between the
  matches. Two hypotheses were tested and neither survived contact.
- **Hypothesis 1, prior-lock: REFUTED.** The idea was that at 200-400 sims the
  search cannot outvote the policy prior, both arms play the prior's top move,
  and two engines playing the same move draw. `scripts/prior_dominance.py` runs
  all five rungs over 60 real middlegame positions and measures how often
  doubling the search changes the move: **18.3%, 20.0%, 13.3%, 13.3%**. The
  bottom rung changes its move MORE often than the top one while being worth an
  eighth as much. The moves change; they do not help.
- **A six-position smoke test said the opposite and I believed it for several
  minutes**, all six having 200/400/800 agree exactly. That is what a 60-position
  sample calls noise. A smoke test proves the code runs. It is not evidence, and
  it is most dangerous when it agrees with a hypothesis you already like.
- **Hypothesis 2, move quality: NOT TESTABLE at this sample size, and the reason
  is the position set.** `scripts/rung_quality.py` scores each rung's chosen move
  against Stockfish at 1,000,000 fixed nodes. Median centipawn loss came out at
  5.5-6.5 for every rung **and for the bare policy prior with no search at all**.
  Positions sampled uniformly out of real games are mostly positions where the
  move is obvious, so the instrument spends its whole sample on decisions that
  do not discriminate. To measure what search buys, sample where search could
  matter: high prior entropy, or positions the rungs already disagree on.
- **Mean centipawn loss over a small sample is a blunder counter wearing a
  continuous disguise.** The means (51.6 / 35.2 / 28.7 / 43.4 / 38.8) look like a
  measurement and duly contradicted the ladder, rating 1600 worse than 800 on a
  rung the ladder prices at +308. The tail says why: positions losing more than
  100 cp number **8, 7, 4, 6, 4** out of 60, so the whole ordering rests on two
  to four positions and "1600 is worse" is two blunders. Print the median and the
  size of the tail beside any such mean, or the aggregate hides what it is made
  of. `rung_quality.py` now persists per-position losses; the first version saved
  aggregates only and the diagnosis cost a second ten-minute run for nothing.
- **Still open, in the order I would try it.** (1) Blunder rate per rung on
  several hundred DISCRIMINATING positions -- a 10%-vs-7% difference needs on the
  order of a thousand to separate, which is ~2 GPU-hours of picks plus CPU for
  the reference. (2) **Tree reuse**, which every rung ran with (`reuse: true`).
  Carrying the tree between moves adds roughly a fixed number of nodes per move
  whatever the nominal sim count, so it is proportionally a far larger subsidy to
  a 200-sim arm than to a 3200-sim one and would compress the bottom of the
  ladder specifically. Needs games rather than positions, so it needs the GPU.
- **RESOLVED, same day: the flat first rung was the virtual-loss defect.** Every
  ladder rung ran with `vloss_fix: false` while the bot has run
  `CHESSGPU_VLOSS_FIX=1` since 2026-07-30. Re-screening 200 -> 400 with the fix
  ON for both arms: **+214.8 +-69.8, W60 D35 L5, LOS 100%** against +29.0 +-25.1
  with it off, non-overlapping intervals, and the rung lands in line with the
  other three. The mechanism explains the SHAPE, not just the size: with the fix
  off, virtual loss is added straight into `value_sum`, so at batch 64 there are
  up to 64 fake values in the tree at once. That is a third of a 200-simulation
  search and 2% of a 3200-simulation one, so the damage is worst exactly where
  the ladder was flat -- and the fix's own +364 verdict was measured at 400 sims,
  inside that same crippled regime.
- **The general fault: the lab's matches were configured like the code's
  defaults, not like the deployment.** `--a-vloss-fix` is `store_true`, default
  off, and `match_argv` never passed it, so EVERY lab match since the port has
  measured an engine that does not play. Nothing enforces the correspondence
  between `systemd/sumofish-bot.service`'s environment and what the harness
  builds; it is now written down in `match_argv`'s docstring, with the other
  three flags (core, dedup, mate_distance) checked and agreeing. Check that list
  against the unit whenever either side changes. Note the deliberate asymmetry:
  `match.py`'s own defaults must stay OFF, because `tests/identity_*.py` needs
  all three defects off for the Rust/Python identity to hold. The identity test
  and the strength test want opposite defaults, which is why the lab states its
  own instead of inheriting.
- **A retraction has to survive a runner that disagrees with it.** `lab.py`'s
  `run()` reads `state.json` ONCE at startup and writes that snapshot back at
  every job boundary. Marking the four rungs withdrawn in `state.json` while an
  8-hour training job was in flight would have been silently reverted at ~01:45
  by a process holding a copy of the state from 17:44, putting four retracted
  numbers back on the board with nobody touching them. Withdrawals now live in
  `runs/lab/withdrawn.json` and are overlaid by `load_state()`, so the worst a
  stale runner can do is lose them for the length of one save. The same overlay
  withdraws derived FACTS (`scale_D`, `scale_bar`), by renaming rather than
  deleting, so a later job that reads one by name fails loudly instead of
  falling back to a default.

## 2026-08-09: a parameter that was never connected, and a test that could not tell

Ran a c_puct sweep against Stockfish, five arms from 1.0 to 4.5. Every arm came
back **2W 1D 1L, +88.7 +-150, byte-identical move hashes**. The obvious reading
is "c_puct does not matter at this budget", and it is wrong.

- **`--cpuct` is a silent no-op in the configuration that ships.** AlphaZero's
  schedule is on by default (`c_puct_base = 19652.0`), and `c_puct_at()` is
  `match self.c_puct_base { None => self.c_puct, Some(base) => ln((1 + N +
  base)/base) + self.c_puct_init }`. Under the schedule `self.c_puct` is never
  read. It binds ONLY under `--fixed-cpuct`. The flag has therefore done nothing
  in every match this project has run that did not also pass `--fixed-cpuct`.
- **`config.json` recorded `c_puct: 4.5` for an arm that ran 1.25.** Provenance
  that records the REQUEST rather than the EFFECT cannot catch this class of
  bug, and makes it invisible in the archive afterwards. The archive says five
  different sweeps happened. They did not.
- **Nothing anywhere could vary `c_puct_init`.** `match.py` had zero references
  to it, so every match ever run used the hardcoded 1.25 -- including the four
  exchange-ladder rungs and every promotion gate. Added `--cpuct-init` and the
  per-side overrides, and threaded it through `Spec` to BOTH engines.
- **The failure mode is the lesson, not the flag.** "This parameter has no
  effect" and "this parameter is not connected" produce identical experimental
  output, and the second is far cheaper to check: vary the input and assert the
  OUTPUT changed, before spending GPU-hours interpreting a flat result. Five
  identical rows should have been read as a plumbing alarm, not a finding. It
  cost 20 minutes here only because the arms were 4 games; at the planned 800
  games/arm it would have been six hours to conclude something false.
- Guarded by `tests/verify_cpuct_binding.py`: no GPU, pure arithmetic over
  `c_puct_at`, asserting that `c_puct` does NOT move the schedule, that
  `c_puct_init` does and additively, that `--fixed-cpuct` inverts both, and that
  `MCTS`, `RustMCTS` and `match.Spec` all actually carry the parameter. A flag
  that parses and is dropped one layer down is the same no-op in a new hat.
- Sanity check that proves the fix rather than assuming it: `c_puct_init` of
  0.5 / 1.25 / 3.0 produce three DIFFERENT games (167, 117, 114 plies), and the
  1.25 arm reproduces the exact move hash `48cd494d410b50e8` that all five
  broken c_puct arms produced. That is the bug and the fix in one measurement.

## 2026-08-09: a tuning sweep where the shape beat every pairwise test

Nine arms, 800 games each against Stockfish@700n, same seed so every arm saw
identical openings. Not one arm separated from the shipped value on its own
pairwise comparison. The sweep is still informative, and reading only the
pairwise gate would have thrown that away.

- **`c_puct_init` sits at the top edge of a cliff, not in a plateau.** 0.5 /
  0.875 / 1.25 scored +43.2 / +58.3 / +42.3, flat within +-19. Then 1.75 is
  **-1.3** and 2.5 is **-56.5**. Against shipped, 1.75 is -43.6 and 2.5 is
  -98.8, both far outside the ~27 difference error. So the shipped 1.25 is not
  wrong, but everything above it falls off fast and nothing warns you. Treat
  upward drift in exploration as dangerous and downward drift as free.
- **The FPU optimum is OUTSIDE the swept range.** -0.5 / -0.35 / -0.2 / -0.05
  gave +13.0 / +23.5 / +42.3 / +55.6: monotone, ~+14 a step, never turning
  over, with the best value the last one tested. **A monotone trend across four
  points is much stronger evidence than the pairwise test that rejects each
  step**, and it says the sweep was bounded in the wrong place. When the winner
  is at the edge of the range, the finding is "extend the range", not "no
  effect".
- **Do not assume the two combine.** Stage 2 swept FPU at the SHIPPED
  `c_puct_init=1.25`, not at stage 1's 0.875, so "best c_puct_init plus best
  FPU" is an untested product of two marginals. Both knobs move exploration, so
  interaction is likely rather than exotic. The combination has to be measured
  as a configuration, which is what the follow-up arms do.
- **Two free validations of the harness, worth more than they cost.** The
  `c_puct_init=1.25` arm scored +42.3 +-19 here; the anchor put that identical
  configuration at +44.4 +-12 in a separate 2000-game run. Two independent
  measurements of one config agreeing inside noise. And stage 2's `fpu=-0.2`
  arm IS stage 1's `c_puct_init=1.25` arm re-run under another name: it
  returned byte-identical 304/289/207. Same seed and same config reproduce
  exactly at the match level, whatever the training pipeline does.
- Cost note: 9 arms x 800 games took 10h19m wall against a 3h44m anchor of 4000
  games, because the bot was playing throughout and each arm reloads the nets.
  Budget match-harness sweeps on ~6s/game with the bot up, not the 3.2s an
  uninterrupted anchor gets.

## 2026-08-09: +63.5 Elo from two config values, and why it was not shipped

`confirm-combined` (`c_puct_init=0.875`, `fpu=-0.05`) scored **+107.9 +-13**
against Stockfish@700n over 2000 games, W980 D642 L378, LOS 100%. The shipped
config sits at ~+43 on two independent measurements (anchor +44.4 +-12 on the
default seed, sweep +42.3 +-19 on seed 4242). That is **+63.5 Elo at ~3.6
sigma**, against a 1h53m training run that bought +3.5 +-26.4.

- **The combination is strongly SUPERADDITIVE, and the marginals would have
  talked you out of it.** Alone, `c_puct_init=0.875` was +16.0 and `fpu=-0.05`
  was +13.3, neither separated from shipped against a ~27 difference error. A
  reader who stopped at the sweep's pairwise gate would have concluded "no
  change justified" and been right about each knob and wrong about the pair:
  the sum of the marginals is +29 and the joint effect is +65.6, better than
  2x. **When two parameters govern the same mechanism, sweeping them one at a
  time and adding the winners is not a plan, it is a different experiment.**
- **Sweeping stage 2 at the SHIPPED value of stage 1 is what made this
  visible.** Had stage 2 been run at stage 1's winner, the interaction would
  have been folded silently into the FPU column and never named.
- **It was NOT deployed, because it was measured at 1/150th of the deployment
  budget.** Everything above is `--sims 400`. The bot plays 15+10 at ~60,000
  nodes a move, and `MCTS.c_puct_at`'s own docstring says exactly why that
  matters: "the balance between trying something new and pursuing what already
  looks good shifts with N, and the c that balances it shifts too." An
  exploration constant is the single parameter least entitled to be assumed
  budget-invariant, and AlphaZero's schedule exists because it is not.
  `sumofish-tune-transfer.service` re-measures combined-vs-shipped head to head
  at 400 sims (the control, which must reproduce ~+63) and at 3200 sims.
- **Head-to-head is safe HERE and the mirror-match lesson still stands.** That
  85%-repetition collapse needed two configurations that play the same moves.
  These two land 65 Elo apart and demonstrably do not. Draw rate is being
  watched anyway: near 85% means the arm is blind and the number is discarded
  rather than interpreted.

## 2026-08-09: three small-sample over-reads in one night

Same error three times, in three different disguises, all within a few hours.

- **A Stockfish doubling looked like a total wipeout at n=6** (SF@700 vs
  SF@1400, 0/6, pairing r=1.000) and I read it as contradicting the anchor's
  ruler. At n=200 it is -205.0 +-57, entirely consistent. Fixed-node Stockfish
  is deterministic, so paired openings give r=1.000 and the variance lives
  almost entirely in WHICH openings got sampled: six games is three openings.
- **The FPU sweep looked monotone across four points** (-0.5/-0.35/-0.2/-0.05
  giving +13.0/+23.5/+42.3/+55.6) and I concluded the optimum was off the edge
  of the range. Testing past the edge found +0.1 at +8.7: it turns over. The
  trend was real, the extrapolation was not.
- **The ruler slope looked like it steepened 2x** across the first three rungs
  (155.5, 205.0, 284.9) and I rewrote `sim_ladder.py` around a piecewise curve
  because of it. With all five rungs the weighted mean is 213.5 and chi2 = 2.47
  on 4 dof: consistent with a CONSTANT slope, nothing beyond 1.1 sigma.

The common shape: **a monotone-looking sequence of three or four points, each
with an interval wide enough to swallow the trend, read as structure.** Every
one of these had its error bars printed right next to it.

The cheap defence, which costs one line: before describing a sequence as a
trend, check whether a flat line fits. `chi2 = sum((x-mean)**2/err**2)` against
its dof is enough. Two of the three above would have died instantly.

What saved all three was that testing the claim was cheap (a few CPU-minutes or
one extra arm) and I tested rather than shipped. Note that the piecewise ruler
was KEPT despite its justification being wrong, for a different and better
reason: it refuses to extrapolate past the measured range, where a fitted
constant would cheerfully price a rung at 50,000 nodes off data stopping at
11,200.

## 2026-08-10: the opponent was not the same opponent from game to game

`Player.new_game()` returned early for a Stockfish side and never sent
`ucinewgame`, with a comment saying each `move()` posts the full position so it
is not needed for correctness, and that hash reuse between games is "noise next
to the game-to-game variance". First half true, second half false.

- **A fixed-NODE opponent whose hash persists is a DIFFERENT opponent each
  game.** A warm transposition table changes what it finds inside the same node
  budget, so its strength depended on how many games it had already played in
  that process. Three consequences, none of them noise: a match was not
  reproducible from its seed; a RESUMED match differed from an uninterrupted
  one; and sharding was biased rather than merely different, since a shard
  plays 1/N of the games, stays colder, and would have flattered us.
- **The tell was a coverage test that passed while the content failed.**
  Splitting a 20-game match in two shards partitioned the pairs perfectly (10 +
  10, no overlap, every index present) and still changed **18 of the 20 games**.
  A partition test alone would have shipped this. Assert on the RECORDS, not on
  the bookkeeping.
- Fix is one argument: python-chess emits `ucinewgame` when the `game` token
  passed to `play()` changes. After it, a two-shard match is byte-identical to
  the same match run by one process.
- **The ruler did not move**: six Stockfish-vs-Stockfish rungs re-measured cold,
  weighted shift **+5.1 +-30.6**, chi2 0.34 on 6 dof. Expected in hindsight,
  because both sides carry the same hash behaviour and it cancels; the
  measurement that can move is SumoFish vs Stockfish, where only one side is
  affected. Note the two runs share `--seed 99` and therefore the same
  openings, so that comparison is PAIRED and more sensitive than independent
  sampling, not less.
- **Old results kept as `rulerWARM-*` rather than overwritten.** The ladder's
  `ruler-*-vs-*` glob excludes them, so the corrected ruler is what gets used
  while the superseded numbers stay auditable.
- **RESOLVED, and the direction was the opposite of what I predicted.** Fixing
  it makes Stockfish **STRONGER**, not weaker. Paired over the 400 openings both
  runs share (same seed), our score went 0.5475 -> 0.4913, **z = -2.16**, mean
  paired shift -0.0563 +-0.0510, only 39% of results identical. About **-39
  Elo**. So the warm harness FLATTERED us and every SumoFish-vs-Stockfish number
  taken on it is biased in our favour.

  The mechanism that fits: at 845 nodes the search is tiny, and a table carrying
  entries from other games and other openings pollutes it, so stale entries at
  the wrong depths cause bad cutoffs. Clearing per game helps a low-node search
  rather than hurting it. I had assumed "warm table = more information =
  stronger", which is the intuition from long searches and is wrong here.

  **Use the PAIRED comparison when two runs share a seed.** Independently the
  two intervals were +38.8 +-19 and -6.1 +-26, a 1.40-sigma difference I was
  about to call undecided. The same data compared game-for-game on shared
  openings is 2.16 sigma and conclusive. Throwing away the pairing nearly cost a
  correct call.
- Process guilt, worth recording: my first cold-vs-warm comparison was
  confounded because I passed the PRE-v6 search constants to the cold arm while
  comparing against a v6 warm rung, and read the resulting -88 Elo as a hash
  effect. It was mostly the tuning difference. When a re-measurement disagrees
  with an old one, diff the full config before believing the delta.
- What needs redoing, and what does not. **Ladder and anchor: redone**, on the
  fixed harness. **Ruler: not**, both sides are Stockfish so it cancels
  (measured: +5.1 +-30.6). **`scale_D`: probably survives**, because it is a
  DIFFERENCE between rungs and a bias roughly constant across rungs cancels --
  which the re-run tests rather than assumes. **The v6 tuning decision: stands**,
  its head-to-head arms contained no Stockfish and its vs-Stockfish arms
  compared two SumoFish configs against the SAME opponent, so the bias is
  common-mode.

## 2026-08-11: every "N sigma" in this repository was N half-widths, and one of them reverses its own lesson

`scripts/elo.py` returns **95% half-widths**, not standard deviations:
`score_stats` and `pair_stats` both build their interval as `1.96 * sigma`
(`elo.py:81`, `:219`). Four places then divided a difference by one of those
`err` values and called the quotient a sigma. Every such figure understates
significance by exactly 1.96x.

| written | where | actual |
|---|---|---|
| "a **1.40-sigma** difference" | `LAB-NOTES.md:1093` | **z = 2.73** |
| "+63.5 Elo at **~3.6 sigma**" | `LAB-NOTES.md:986` | **z = 7.24** |
| "a **0.40-sigma** difference" | `STATE.md`, anchor routes | **z = 0.78** |
| "chi2 **0.34** on 6 dof" | `LAB-NOTES.md:1070` | **chi2 = 1.29 on 5 dof** |

- **The 1.40 one inverts the lesson it was written to support.** The 08-10 entry
  says the unpaired comparison of the warm and cold harnesses was "a 1.40-sigma
  difference I was about to call undecided", that pairing rescued it at 2.16,
  and concludes "throwing away the pairing nearly cost a correct call."
  **Backwards.** 1.40 half-widths is z = 2.73, which is MORE significant than
  the paired 2.16, and the two intervals had already separated. Recomputed at
  matched n (the warm run's first 400 games against the cold run's 400):
  unpaired **z = -2.08**, paired **z = -2.16**. Pairing bought 4%.
- **And the mechanism explains why it had to be small.** `pairing_efficiency`
  on these matches is **0.89 to 1.01**. Pairing pays when the two arms' results
  on the same opening are correlated, which is what colour-swapping against a
  MIRROR buys (r = 0.44 on the one mirror match, hence PHILOSOPHY's "~2.4x in
  games"). Against a FIXED external opponent that correlation is nearly absent,
  and every match this project now runs is that kind. **The 2.4x does not
  transfer, and PHILOSOPHY still quotes it unqualified.**
- **What survives from the 08-10 entry:** the direction and the conclusion. The
  warm harness did flatter us, the paired analysis is correctly computed
  (z = -2.16 reproduces exactly: n = 400, 0 opening mismatches, 0 colour
  mismatches, mean -0.0563, SE 0.0260), and pairing is still the right default
  because it cannot hurt. What does not survive is "pairing rescued this call".
- **The chi2 is the same error and the conclusion also survives.** Recomputed
  over the six `ruler-*` / `rulerWARM-*` pairs: weighted mean shift
  **+5.1 +-30.6**, exactly as recorded, and chi2 = **1.29 on 5 dof** (p = 0.94)
  rather than 0.34 on 6. Still entirely consistent with "the ruler did not
  move". Note the dof: six rungs compared against one weighted mean is 5, not 6.

**The rule, and it is a convention rule rather than a statistics one.** `err`
in this codebase is always a 95% half-width. So:

- To combine independent errors, add them **in quadrature as half-widths**. The
  1.96 factors out, so this is exact and needs no conversion. `sim_ladder.py`
  now does this and says so at the point it does it.
- To quote a z, **divide by `err/1.96`**, never by `err`.
- Better, quote neither: say "the intervals do not overlap" or give the
  difference with its own interval. Both are unambiguous and neither invites
  this mistake. Every conclusion in the table above is unchanged; what changed
  is that two of the four figures made a real result look marginal and one made
  a marginal one look settled, so the error is not conservative in either
  direction.

**Why it survived this long:** a wrong sigma never fails. It produces a
plausible number in the right ballpark, in a sentence that reads like careful
work, and nothing downstream consumes it. It was found by recomputation from
the raw `status.json` files, not by review.

## 2026-08-11: the arbiter had the same hash bug, and its blast radius was zero

The 08-10 fix gave `Player` a per-game token so python-chess would emit
`ucinewgame`. `Arbiter.agrees()` was the other Stockfish process in the same
file and did not get one: it called `analyse()` with `game=None` every time, and
python-chess fires `ucinewgame` only when the token CHANGES
(`first_game or self.game != game`), so `None != None` is False and it fired
once per PROCESS. The adjudicator accumulated a transposition table across every
probe of every game in a match, at 200,000 nodes a probe.

This looked worse than the player-side bug, for three reasons:

- **the arbiter decides the result.** 35.0% of the 700-node anchor and 29.8% of
  the 1600-node anchor ended `adjudicated-arbiter`;
- it breaks the same three things (a match not reproducible from its seed, a
  resumed match differing from an uninterrupted one, sharding biased rather than
  merely different);
- and it is **not common-mode across rungs**, so the argument that `scale_D`
  survives because "a bias roughly constant across rungs cancels in a
  DIFFERENCE" never covered it. Each rung is a separate process warming over a
  different game count and a different position distribution.

**Measured, and it moved nothing.** `scripts/arbiter_bias.py` recomputes every
adjudicated verdict both ways from the stored `final_fen`, cold (a token per
position) against warm (one token, original order), with `Arbiter.agrees()`'s
own threshold and WDL model:

| match | adjudicated | cold upholds | warm upholds | **disagree** |
|---|---|---|---|---|
| `stockfish-anchor-700nodes` | 700 | 700 | 700 | **0** |
| `stockfish-anchor-1600nodes` | 609 | 609 | 609 | **0** |

**1,309 positions, zero flips.** So no rung needs re-running and both anchors
stand as measured.

- **Why zero is the expected answer in hindsight.** Adjudication fires at
  `wp >= 0.97`, and a 200,000-node probe of a position already past 0.97 is
  nowhere near the margin where a warm table changes a verdict. A warm table
  changes what a search FINDS inside a budget; it does not change a position
  that is already resolved by two orders of magnitude more nodes than the
  players got. Contrast the player side, where the same defect was worth about
  -39 to -77 Elo: there Stockfish played at 700-1600 nodes, a search small
  enough that stale entries at the wrong depths genuinely change the move.
  **The same bug is large where the search is small and nil where it is large.**
- **The fix ships anyway, and the justification is reproducibility, not Elo.**
  A match has to be a function of its seed. It was not, and the cost of proving
  the consequence was nil was ~20 minutes of CPU against a stored column that
  was already on disk.
- **Worth copying: the cheap measurement existed because the harness logged
  `final_fen`.** Nothing had to be replayed and no GPU was touched. When adding
  a decision point to the harness, log the input it decided on. It converts a
  future "is this bias real" argument into a script.

## 2026-08-11, evening: I narrowed the error on a number whose bias I had never checked

The morning's job was the ruler: six Stockfish-vs-Stockfish rungs at 200 games
each, contributing +-32.1 of `scale_D`'s +-33.5 while the two SumoFish rungs
contributed +-9.4. Re-earning them at 2400 games is CPU-only, costs 1.8 hours
and no GPU, and takes `scale_D` to +-12.2. That reasoning is correct and the run
did exactly what it promised. Every one of the six edges came back inside its
old interval, largest move -37.1 against +-74.1, so the old ruler was wide and
not wrong.

Then the second anchor landed and the whole quantity went out.

`sim_ladder.py` makes a rung absolute by adding a walk along that ruler:
`absolute = rung + (ruler(N) - ruler(700))`. The ruler is Stockfish against
Stockfish. The walk is only meaningful if a node budget worth X Elo to
Stockfish is worth X Elo to SumoFish, and **nothing in this repository had ever
tested that.** Two anchors of the same configuration test it in one subtraction:

    SF@700n -> SF@1600n, Stockfish vs Stockfish   280.8 +-16.4
    the same span, measured through SumoFish      192.8 +-18.0
    difference 88.0 +-24.4, z = 7.1, factor 0.69

`scale_D` = 185.2 is +1026 of ruler walk against -286 of rungs, so a 31%
overstatement of the walk is a bias several times the interval it was published
with. The ladder predicts the 1600-node anchor at -248.5; the anchor measured
-162.4 +-13.4.

**The lesson is not "the ruler was wrong".** It is that I spent the cheapest
measurement available on the largest VARIANCE term without once asking whether
the term it multiplies is on the right scale. Error bars are a claim about
repeatability, and repeatability is silent about a systematic error in the same
direction every time. The tell was on the screen for a week and I read past it:
the ruler matches draw 7-13% of their games and adjudicate 74-87%, the anchor
matches draw 28-38% and adjudicate 30-35%. Elo inferred from a score is
draw-rate dependent. Two match populations that different are not on one scale,
and no amount of games fixes it.

Practical rules this leaves:

- **Before narrowing an interval, ask what would falsify the point estimate.**
  If the answer is "nothing on the schedule", the interval is not the bottleneck.
- **A chain through a DIFFERENT population is a modelling assumption, not
  arithmetic.** `sim_ladder.py`'s own docstring indicts the withdrawn design
  because "error accumulates down a chain", and the redesign moved the chain
  from the SumoFish axis to the Stockfish axis rather than removing it. Moving
  a chain onto an axis where you cannot see it is worse than leaving it where
  the error propagation catches it.
- **Two measurements of the same span by different routes is the cheapest
  possible audit, and it needs no new games** when both already exist. The
  700-node anchor and the ladder's 400-sim rung agreed (+30.5 vs +32.3) and I
  quoted that agreement as corroboration for two days. It only spans 0.27
  doublings. The disagreement lives at long range, so agreement at short range
  is not evidence of transitivity, and I treated it as if it were.
- `scripts/ruler_transfer.py` runs the check and refuses to answer when fewer
  than two anchors share a configuration, which is the state this project was in
  from the day the ladder was designed until this evening.

**What NOT to do next: publish `scale_D` x 0.69.** The factor is measured at one
place on the scale. Assuming it is constant across 360-11200 nodes is the same
species of assumption as the one that just failed, and it would look like a
correction while being another guess.

## 2026-08-11: the same file, destroyed twice in one day, by two write paths

`runs/lab/tune-search.json` holds every tuning arm this project has run. It was
rebuilt from the per-arm match directories twice on 2026-08-11:

1. Morning. `--report --stage2-only` ended by writing the file like a real run,
   so a read replaced five completed arms with five FAILED rows. Fixed by
   returning before the write, and the commit message says the rebuilt record
   "is committed as the new tune-search.json". **It was not.** `runs/` is in
   `.gitignore`, so nothing was committed and the sentence describes a thing
   that could not have happened.
2. Afternoon. The real `--stage2-only` run then destroyed it again through the
   ordinary write path, which had been left alone: it wrote a freshly-built
   dict, so stage 1 became `{"skipped": ...}` and the FPU line measured at
   `c_puct_init=1.25` vanished entirely.

Both times the recovery was "the per-arm directories survived". That is luck
presented as a design, twice, and the second time by someone who had just
written the sentence calling it luck.

- **A partial run must MERGE.** `_merge()` now folds a run's groups into what is
  on disk: it may add or replace what it measured, and a stage that was SKIPPED
  never overwrites arms that were really played.
- **Key a result by the configuration it was measured at.** Stage 2 now writes
  `stage2_at_ci0.875`, not `stage2`, so a sweep at one `c_puct_init` cannot
  occupy the slot of a sweep at another. The bare key is what made the 08-09
  and 08-11 FPU lines collide in the first place.
- **`git status` clean does not mean a result is safe.** Every number this
  project earns lands under `runs/`, which is ignored. If it matters, it has to
  be reconstructible from the per-arm directories BY A SCRIPT, or committed
  somewhere that is not ignored.
- Guard: `tests/verify_tune_merge.py`, registered in `tests/run_all.sh`.
