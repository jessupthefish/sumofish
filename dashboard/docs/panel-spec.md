I have read every file. Here is the specification.

---

# SumoFish Terminal Dashboard — Feature Specification for Rewrite

Source of truth read in full: `/home/nomad/chess-gpu/scripts/watch.py`, `/home/nomad/chess-gpu/scripts/dash/{panels,widgets,theme,ink,board,sixel,state,sources,fusion}.py`, `/home/nomad/chess-gpu/scripts/dash/make_sprites.py`, `/home/nomad/chess-gpu/tests/verify_layout.py`, `/home/nomad/chess-gpu/CLAUDE.md`, and the telemetry producer `/home/nomad/chess-gpu/chessgpu/engines/search_engine.py`.

---

## 0. ARCHITECTURE AND DATA MODEL (prerequisite for the panel specs)

### 0.1 Three layers, one direction

`sources` (threads, own cadences) → `State` (values + provenance) → `panels` (pure functions). The render loop is a pure function of state at a fixed frame rate (`FPS = 12`, `watch.py:60`). No panel does I/O, sleeps, or blocks. No layer imports torch.

### 0.2 `Field` — the staleness model (`state.py:74-110`)

Every state key is a `Field(name, interval, value, updated, error, fills)`.

| property | rule |
|---|---|
| `age` | `inf` if never updated, else `now - updated` |
| `track` | `LOST` if `fills == 0`; `LOST` if `error` and `age > interval*2`; `LOST` if `age > interval*6`; `COAST` if `age > interval*2`; else `LIVE` |
| `set(v)` | sets value, stamps `updated`, clears `error`, `fills += 1` |
| `fail(why)` | records error only; value and `updated` untouched (so it coasts) |

`ago(seconds)`: `inf` → `"never"`; `< 90` → `"{int}s"`; `< 5400` → `"{int/60}m"`; else `"{int/3600}h"`.

### 0.3 Registered fields and their healthy intervals (`state.py:167-199`)

| field | interval (s) | contents |
|---|---|---|
| `profile` | 45 | lichess `/api/account` JSON (`perfs`, `count`) |
| `playing` | 6 (polled at 2) | `nowPlaying` list |
| `game` | 90 | `{id, meta, board, last, moves, history, wc, bc, clock_at}` |
| `engine` | 30 | newest telemetry record for our game |
| `engine_board` | 30 | `{board, last, ply}` from `fusion.EngineBoard` |
| `train` | 20 | `{run, loss[-400:], evals}` |
| `rating_log` | 120 | last 500 records of `logs/rating.jsonl` |
| `gpu` | 3 | `{util, used, total, temp, power}` |
| `units` | 8 | `{unit: {active, sub, restarts}}` |
| `finished` | 3600 | last completed game summary |
| `results` | 1800 | up to 40 parsed PGN rows |
| `grades` | 60 | `{ply: {loss, grade, cp}}` (Stockfish) |
| `eval_curve` | 60 | `{ply: wp_white}` (Stockfish, whole game) |
| `record` | 1800 | `{w, d, l, version}` for the current release |

Note the deliberate mismatch: `playing` is *polled* every 2 s but graded against a 6 s interval; `game` is pushed per move so its 90 s interval means "a long think is not a fault".

### 0.4 Non-field state

- `state.curve: dict[ply -> P(White wins)]` — the engine's own curve. `record_eval(ply, wp_white)` **resets the whole dict if `ply < max(curve)`** (a new game). `curve_series()` and `curve_items()` snapshot under the lock.
- `state.eval_smooth: Smooth(tau=0.35)` — exponential easing, wall-clock based: `value += (target-value) * (1 - exp(-dt/tau))`, snapping when `|target-value| < 5e-4`. Keeps `.target` separately: **the bar eases, the printed number never does.**
- `state.tape: Tape` — `deque(maxlen=400)` of `(timestamp, kind, text)`.
- `state.health` — worst track over `profile`, `playing`, `engine` (computed, currently unused by panels).

### 0.5 Engine telemetry record shape (`search_engine.py:176-207`)

One JSON object per line in `logs/engine.jsonl`, ~6 Hz while searching, nothing between moves.

```
ev      "think" | "move" | "boot"
ply     int (board.ply() at search start)
fen     full FEN being searched
stm     "w" | "b"
wp      P(win) for the SIDE TO MOVE, 4 dp
wp_white P(White wins), 4 dp   <- the frame every consumer must use
nodes   evaluations
nps     int(nodes/elapsed)
sims    simulations done
elapsed seconds, 3 dp
budget  seconds, 3 dp
best    SAN of top move or null
pv      list of SAN (<=8 mid-search, <=14 final)
top     list of [san, visits, q, prior]  (5 mid-search, 6 final)
mate    bool
done    bool
uci     final records only
t, pid, game  stamped by telemetry.py
```
`boot` records carry `policy_step, value_step, bins, sims, batch, params`.

---

## 1. PANEL SPECIFICATIONS

All panels are wrapped by `_panel(body, title, border=FAINT, subtitle=None)` (`panels.py:47`): a bordered box, title left-aligned in `FG`, subtitle right-aligned, `padding=(0,1)` (one column each side, no vertical padding), body background `BG`. **Exception: `board_panel` in image mode emits no panel at all** (see 1.2).

---

### 1.1 `header` — `panels.py:113`, height 3 rows, full terminal width

A two-column expanding grid, 2 columns of padding between, left column left-justified, right column right-justified. Panel title is empty string; border `FAINT`.

**Left run:**

1. `" SUMOFISH "` — bold, `BG` text on `ACCENT` background (a badge).
2. For each of `bullet, blitz, rapid, classical` **in that fixed order**:
   - Skipped entirely if `profile.perfs[tc]` absent or `games` is 0/absent.
   - Three spaces are appended as a separator **before** the liveness test, so a control that is skipped for being inactive still leaves 3 spaces of gap (existing quirk; harmless).
   - If `tc` not in `active_controls()`: nothing else is drawn for it (see below).
   - Otherwise: `"{tc} "` in `DIM`; `"{rating}"` bold `ACCENT`; `"?"` in `WARM` if `perfs[tc].prov`; `" ±{rd}"` (`"±?"` if absent) in `FAINT`; `"  {games}g"` in `DIM` (abbreviated form deliberately — the long form clipped the row at 100 cols).
   - If the rating history for that control has ≥2 samples: two spaces, then a 14-column `sparkline` in `INFO`, then `" {delta:+.0f}"` where `delta = series[-1] - series[0]`, coloured `GOOD` if `delta >= 0` else `BAD`.
3. If no control was drawn: `"   no rated games at the control we play"` in `DIM`.

`active_controls()` (`panels.py:87`): reads `config/lichess-bot.yml` → `challenge.time_controls`; falls back to all four on `OSError, ImportError, ValueError, AttributeError` **only** — deliberately narrow, because a bare `except Exception` once swallowed a `NameError` and silently returned "everything".

Rating history (`_rating_history`, `panels.py:208`): walks `state.rating_log` records, appending `rec.ratings[tc].rating` in file order per control.

**Right run:**

1. If a game is live: `"lichess.org/{game.id}   "` in `FAINT`.
2. Record: if `state.record.version` is set, use `record.w/d/l`, `total = w+d+l`, `scope = "   in {version}   "`. Otherwise fall back to `profile.count.win/draw/loss` and `scope = "   of {all}   "`.
3. `"{won} won"` `GOOD`; `" · "` `FAINT`; `"{drew} drawn"` `DIM`; `" · "` `FAINT`; `"{lost} lost"` `BAD`.
4. If versioned: `"   {total} games{scope}"` in `DIM`; else just `scope` in `DIM`.
5. `track_tag(profile_field, "lichess")`.
6. Heartbeat: `"  ◆"` when `int(time.time()*2) % 2` else `"  ◇"`, in `FAINT`. **Toggles at 2 Hz independently of any data** — it proves the render loop is alive when nothing has changed.

Degraded: no profile → all zeros and the "no rated games" text; the `track_tag` carries the truth.

---

### 1.2 `board_panel` — `panels.py:219`

Signature: `(state, user, width, height, scale="pixel2", image_rows=0, image_cols=0, indent=0)`.

**Idle (no `game`)**: a `Panel` titled `"board"`, border `FAINT`, containing `_idle_text` centred horizontally and vertically:
- `"no game in progress\n"` in `DIM`.
- If `state.finished`: `"\nlast: {RESULT}"` bold in `{win: GOOD, loss: BAD, draw: DIM}`; `" vs {opponent}"` in `FG`; `" ({opp_rating})"` in `DIM` if present; `"\n{speed} {rated|casual} · {status}"` in `DIM`; `"  {delta:+d}"` in `GOOD`/`BAD` if a rating delta exists.

**Live.** Position selection (must be identical to `watch._live_position`): if `engine_board` exists and `engine_board.ply >= game.board.ply()`, use the engine's board and last move; else the stream's. The engine view wins because the public stream measures a median 8.7 s behind.

Orientation: `we = _our_colour(players, user, playing)`; `flip = (we == "black")`. `_our_colour`: if `players.white.user.name` case-insensitively equals `user` → `"white"`; else if `playing` non-empty → `playing[0].color`; else `"black"`.

Player identity rows (`_names`): if `flip`, top = White, bottom = Black; else top = Black, bottom = White. **Our side is always at the bottom.** Each side yields `(title, name, rating)` as three separate fields — deliberately not one string, because a rating buried after a variable-length name lands in a different column on every line.

Clocks: `top_clock = bc if not flip else wc`, bottom the other. Values from `live_clocks`.

Evaluation for the player lines: `wp_white = engine.wp_white` **only if `_engine_matches(engine, board)`**, else `None`. `wp_bottom = 1 - wp_white if flip else wp_white`; `wp_top = 1 - wp_bottom`.

`span = image_cols or width` — the player blocks are set to the *picture's* width, not the column's.

**Image mode (`image_rows > 0`)**: returns a bare `Group` of
`top_block (2 rows)` + `image_rows × Text("")` (completely unstyled) + `bottom_block (2 rows)`.
No panel, no border, no background anywhere on those rows. This is load-bearing (see §E).

**Text mode**: `Group(top_block + board.render(...).split("\n") + bottom_block)`.

#### 1.2.1 Player block — 2 rows per player (`_player_block`, `panels.py:386`)

Row order: above the board → `[name, material]`; below the board → `[material, name]`. So the **material row is always the one nearest the board** on both sides; the rows read outward from the position.

The block height is fixed at 2 regardless of content — a block that changed height would move the sixel image out from under itself (`verify_layout.py` check 3).

**Name row** (`_name_line`, fixed columns `NAME_W=20, RATING_W=5, CLOCK_W=8`):

| segment | format | style |
|---|---|---|
| indent | `indent` spaces | `on BG` |
| to-move marker | `"▸ "` if to move else `"  "` | `ACCENT on BG` |
| name | `f"{name[:20]:<20}"` | `bold FG on BG` |
| title | `f"{title:>3} "` | `PLUM on BG` if titled, else `DIM on BG` |
| rating | `f"{rating or '--':>5}"` | `bold ACCENT on BG` |
| *(pad)* | spaces to right-align the tail | `on BG` |
| clock | `f"{clockstr(clock):>8}"` | `bold`, `BAD` if `clock < 10`, else `ACCENT` if to move, else `DIM` |
| win estimate | `f"{prob*100:>5.0f}%"`, or 6 blank spaces if `None` | `FG on BG` |

Both names are bold, including the waiting player — dimming the waiter costs readability to say what the marker and the ticking clock already say twice.

**Material row** (`_material_line`): `indent + MARK_W(2)` spaces, then the captured text, padded to `indent + width`. The 2-column offset puts material under the name, not under the marker.

`_captured(board, colour)` (`panels.py:454`) — what this side has taken, **derived from what is missing off the starting set** (correct on a mid-game attach; never from a move history):

- Iterate in this order: Queen (1), Rook (2), Bishop (2), Knight (2), Pawn (8).
- `taken = start_count - len(board.pieces(type, other_colour))`; skip if ≤ 0.
- Emit the figurine `{Q:♛, R:♜, B:♝, N:♞, P:♟}` in `DIM`, then the count `"{taken}"` if `taken > 1` **or a single space if exactly 1** (`♛1` reads as a worse `♛`), in `bold FG`, then a separator space.
- Material balance: `mine = Σ VALUES[p] * count(p, colour)`, same for `theirs`, with `VALUES = {P:1, N:3, B:3, R:5, Q:9, K:0}`. If `mine > theirs`, append `" +{diff}"` in `bold GOOD`. Never shows a negative — the other player's row shows their plus.

#### 1.2.2 `clockstr(seconds)` (`panels.py:53`)

- `None` → `"--:--"`; negative clamped to 0.
- `m, s = divmod(int(seconds), 60)`.
- `m >= 60` → `"{h}:{mm:02d}:{ss:02d}"`.
- `seconds < 10` → `"{m}:{ss:02d}.{tenths}"` (tenths = `int((seconds % 1) * 10)`).
- otherwise → `"{m}:{ss:02d}"`.

#### 1.2.3 `live_clocks(game)` (`panels.py:64`)

lichess sends clocks only on a move, so both are counted down locally: `elapsed = now - game.clock_at`; if the board has any moves played, subtract `elapsed` from whichever side is to move. Resync happens on every stream message (which rewrites `clock_at`). Returns `(wc, bc)` unmodified if either is `None`. **No local countdown before the first move.**

#### 1.2.4 `_engine_matches(eng, board)` (`panels.py:317`)

True iff `eng.fen`'s placement field equals `board.fen()`'s placement field, **or** `eng.ply ∈ {board.ply(), board.ply()-1}`. Otherwise the evaluation is treated as absent everywhere (gauge blank, win estimates blank). This prevents a two-ply-old eval rendering as if it described the board on screen.

---

### 1.3 `mind_panel` — `panels.py:504`, "engine search"

`inner = width - 4`.

**No engine record**: panel border `FAINT`, body `"waiting for the engine to move\n"` in `DIM` + `"logs/engine.jsonl"` in `FAINT`.

**Live.** `thinking = (eng.ev == "think")`. Panel border is `ACCENT` while thinking, `FAINT` when idle.

*Row 1 — annunciator:*
- `" SEARCHING "` or `"   IDLE    "` (both exactly 11 cells), `bold BG` on `ACCENT` (searching) / `FAINT` (idle). State is carried by position and colour, not by text you must read.
- two spaces, then `f"{eng.wp*100:5.1f}%"` in `bold FG`, then `" to move"` in `DIM`. **Note: this one number is deliberately in the side-to-move frame** (`wp`, not `wp_white`) and is labelled as such; it is the only place `wp` is shown raw.
- If `eng.mate`: `"   MATE IN LINE "` in `bold BG on BAD`.
- three spaces, then `track_tag(engine_field)` with no label.

*Row 2 — time budget, a draining gauge:*
`"time  "` `DIM` + `bar(1.0 - frac, w, colour)` + `f" {used:5.2f}s of {budget:.2f}s"` `DIM`,
where `used = eng.elapsed`, `budget = max(eng.budget, 1e-9)`, `frac = min(1, used/budget)`, `w = min(GAUGE_MAX=44, max(8, inner-34))`, colour = `BAD` if `frac > 0.9` else `COOL`. The bar empties as time runs out.

*Row 3 — counters,* all thousands-separated, labels `DIM`, values `FG`:
`"nodes {nodes:,}"`, `"   nps {nps:,}"`, `"   sims {sims:,}"`, `"   ply {ply}"` (ply not separated).

*Row 4:* blank (styled `on BG`).

*Candidate ladder:* `room = height - 8`; `ladder(eng.top[:max(1, room)], width=min(44, max(6, inner-34)), fg=ACCENT)`. See §B.

*Principal variation:* if `eng.pv` non-empty: a blank row, then a row with `overflow="ellipsis"`: `"pv  "` in `DIM`, then each SAN followed by a space, the **first** SAN in `PLUM` and the rest in `FG`.

Missing sub-fields default: `nodes/nps/sims/ply` → 0, `wp` → 0.5, `elapsed` → 0.0, `top`/`pv` → empty.

---

### 1.4 `moves_panel` — `panels.py:584`, "moves"

**Curve source, in priority order:**
1. `state.eval_curve` (Stockfish, `{ply: wp_white}`, covers the *whole* game from move 1) — `curve = [ours(state, sf[ply]) for ply in sorted(sf)]`.
2. Else `state.curve_series()` (the engine's own, only from when the dashboard attached).

Both are `wp_white` and both go through `ours()`, so the axis does not change when the fallback fires.

`rows = height - 4 - (1 if curve else 0)`.

**Empty**: if no game or no moves, the panel body is `"no moves yet"` in `DIM`.

**Sparkline row** (only if `len(curve) >= 2`):
- Label `"eval "` if the Stockfish curve is in use, `"eval·"` (interpunct) if it is the engine-only fallback — the mark is how you tell which source you are looking at. Style `DIM`.
- `sparkline(curve, width=min(44, max(8, width-20)), style=PLUM)`.
- `f" {lo:.2f}-{hi:.2f}"` in `FAINT` (the sparkline's own min/max — every trend carries its endpoints as text).

**Move list**: moves are paired into `(move_number, white_move, black_move|None)`; the **last** `max(1, rows)` pairs are shown (tail, so the current move is always visible).

Each line:

| segment | format | style |
|---|---|---|
| number | `f"{n:>3}. "` | `FAINT` |
| White SAN | `f"{san:<8}"` | `FG` |
| grade mark | `f"{mark:<2}"` | `bold`, see below |
| Black SAN | `f"  {san:<8}"`, or 10 spaces if absent | `FG` |
| grade mark | as above (only if the move exists) | |

Grade marks come from `state.grades[move.ply].grade`:
`inaccuracy` → `"?!"` `WARM`; `mistake` → `"?"` `BAD`; `blunder` → `"??"` `BAD`; anything else (`best`, `good`, missing) → `"  "` (two spaces) in `FAINT`.
**Only bad moves are marked** — annotating accurate moves would be a column of noise, and the eye is hunting for where the game went wrong.

Grade thresholds (`sources.GRADES`, lichess's own bands, centipawns lost): `<10 best`, `<50 good`, `<100 inaccuracy`, `<300 mistake`, else `blunder`.

Explicitly removed and must not come back: a per-row win-probability column. The sparkline carries the same series better, and a column of decimals next to the marks made the marks hard to find.

---

### 1.5 `curve_panel` — `panels.py:684`, "evaluation"

`inner_h = max(3, height - 5)`. The panel exists only if `Plan.curve_h >= 7`.

Series: `[ours(state, v) for v in state.curve_series()]` — **the engine's own curve only**, not the Stockfish one (unlike `moves_panel`).

**Head row (1 line):** reads `state.eval_smooth.target` (the un-eased value; a *number* must never show a value the engine did not produce).
- If `None`: `"     --"` (7 cells) in `DIM`. The gauge is blank in the same frame, so the panel says "no evaluation for this position" in both places at once.
- Else: `f"{advantage(now):>7}"` in `EVAL_WHITE`, then `f"   wp {now:.2f}"` in `FAINT`.

`advantage(wp)` (`panels.py:656`): clamp `p` to `[1e-4, 1-1e-4]`, return `f"{400*log10(p/(1-p))/100:+.2f}"` — the inverted Elo logistic, i.e. the conventional pawn scale, **signed from SumoFish's point of view** (`+` is us better whichever colour we are, deliberately *not* the White-relative sign a chess GUI uses). It is the same mapping `search_engine.centipawns` uses for UCI `score cp`, copied rather than imported because importing it would pull in torch. Monotone in `wp`, which is what makes it safe to show beside the gauge: they cannot tell different stories.

**Body (`inner_h` rows):** each row is
`[axis label 4 cells] + [gauge, 3 cells] + [2 spaces] + [chart row, if any]`
- Axis label: `"1.0 "` on row 0, `"0.5 "` on row `inner_h // 2`, `"0.0 "` on row `inner_h - 1`, otherwise 4 spaces. Style `FAINT`.
- Gauge: `evalbar(state.eval_smooth.value, inner_h, width=GAUGE_W=3)` — the **eased** value, so the boundary slides. Three columns wide, not two: at two it is twelve times taller than wide and reads as a rule rather than a bar.
- Chart: `curve_chart(series, max(8, width - 3 - 10), inner_h)` if `len(series) >= 2`, else nothing.

The chart's scale is **fixed 0..1, never fitted to the game's range**, and the drawn midline is always 0.50.

---

### 1.6 `train_panel` — `panels.py:739`, "training", 6 rows

`inner = width - 4`. **Missing / no `loss` list** → body `"no training run active"` in `DIM`.

`last = train.loss[-1]`; `total = 300_000` (hardcoded step target); `frac = min(1, last.step/total)`.

*Line 1:* `f"{run or '?'}  "` `COOL`; `f"{step:,}"` `bold FG`; `f"/{total:,}  "` `DIM`; `f"eta {eta:.1f}h  "` `DIM` where `eta = (total - step) / max(1, samples_per_s/1024) / 3600` (i.e. steps/second derived by dividing samples/s by a fixed batch of 1024); then `track_tag(train_field)`.

*Line 2:* `bar(frac, min(44, max(8, inner-8)), COOL)` + `f" {frac*100:4.1f}%"` `DIM`.

*Line 3:* `"loss "` `DIM` + `f"{last.loss:.4f} "` `FG` + `sparkline(all losses, min(44, max(8, inner-30)), INFO)` + `f" {lo:.3f}-{hi:.3f}"` `FAINT`.

*Line 4:* if `train.evals` has `puzzle_acc` values: `"puzz "` `DIM` + `f"{accs[-1]:.3f} "` `ACCENT` + `sparkline(accs, same width, WARM)` + `f" {lo:.3f}-{hi:.3f}"` `FAINT`. Else `"no puzzle eval yet"` in `FAINT`.

Note `sparkline` only ever draws the last `width` values, so both traces are a moving window.

---

### 1.7 `machine_panel` — `panels.py:785`, "machine", 5 rows

`inner = width - 4`. Exists so "is training starving the bot" is answerable by looking.

*If `gpu` present:*
- Row 1: `"gpu  "` `DIM` + `bar(util/100, min(44, max(6, inner-40)), COOL)` + `f" {util:3.0f}%"` `FG` + `f"  {temp:.0f}°C"` (`BAD` if `temp > 80`, else `DIM`) + `f"  {power:.0f}W"` `DIM`.
- Row 2: `"vram "` `DIM` + `bar(used/max(total,1), same width, PLUM)` + `f" {used/1024:.1f}/{total/1024:.0f}G"` `DIM` + two spaces + `track_tag(gpu_field)`. Units in: MiB in, GiB out; used gets 1 dp, total gets 0.

*Else:* one row, `"nvidia-smi unavailable"` in `DIM`.

*Last row — units and API budget:* for each unit in `state.units` (insertion order; sources tracks `chess-gpu-bot`, `chess-gpu-train`):
- `"●"` in `GOOD` iff `active == "active"` **and** `sub == "running"`, else `BAD`.
- `" {unit with 'chess-gpu-' stripped} "` in `DIM`.
- If `restarts` not in `("0", "?")`: `"({n} restarts) "` in `WARM`.

Then `f"  api {GATE.per_minute()}/min"` in `DIM` (requests in the last 60 s, from a 600-entry deque), and if `GATE.throttled` is non-zero, `f"  {n} throttled"` in `BAD`. This is on screen because the dashboard shares an IP with the bot and a throttled bot stops playing.

---

### 1.8 `tape_panel` — `panels.py:832`, "event log"

`state.tape.tail(max(1, height - 4))`, oldest first. Per line (`no_wrap`, `overflow="ellipsis"`):
- `strftime("%H:%M:%S ", localtime(ts))` in `FAINT`.
- `f"{kind:<7}"` coloured by kind: `move`→`ACCENT`, `game`→`INFO`, `result`→`GOOD`, `warn`→`BAD`, `engine`→`PLUM`, unknown→`DIM`.
- the text in `FG`.

Empty → `"nothing yet"` in `FAINT`.

Producers of tape lines (for parity):
- `engine`: `"watch started"`, `"engine restarted"` (on a `boot` record).
- `game`: `"game {id} vs {name} ({rating}) {speed}"` on a new game; `"game {id} over"` when it ends.
- `move`: `"{best}  wp {wp:.3f}  {nodes}n in {elapsed:.2f}s"` per completed search. **Backfilled records are deliberately not taped** — the tape is a log of things as they happened.
- `result`: `"{RESULT} vs {opponent} by {status}  ({delta:+d})"`.
- `warn`: `"game stream: lichess {code}"`, `"game stream dropped: {ExcName}"`.

---

### 1.9 `results_panel` — `panels.py:848`, "recent games", 8 rows when present

Source: `logs/games/*.pgn` written by lichess-bot itself, parsed by the source thread, filtered to games at or after the current release timestamp from `VERSIONS.jsonl`, newest first, capped at 40.

**Empty** → body `"no finished games yet"` in `DIM`, subtitle `track_tag(field, "")` (dot only).

**Width allocation is computed, never a fixed slice:**
```
HOW    = 12                    # "time forfeit", the longest lichess sends
fixed  = 7 + 2 + 1 + 8 + HOW   # " HH:MM " + "D " + gap + "  180+0 " + how
who_w  = max(8, width - 4 - fixed)
```
Every column including separators is counted. `verify_layout.py` asserts widths 52..99 render `"time forfeit"` in full — this has been broken twice, once by an 11-char slice and once by an off-by-one.

Rows: `rows[:max(0, height-2)]` (two rows of border chrome). Per row, all in one `Text` with `\n`:
- `f" {when} "` `FAINT` — local `HH:MM` converted from `UTCDate`+`UTCTime`, `"?"` if unparseable.
- `f"{mark} "` `bold` — `win`→`W`/`GOOD`, `loss`→`L`/`BAD`, `draw`→`D`/`DIM`, anything else (`none`, i.e. PGN result `*` — aborted/abandoned) → `"·"`/`FAINT`.
- `f"{opponent[:who_w]:<{who_w}} "` `DIM`.
- `f"{tc:>7} "` `FAINT` — the raw PGN `TimeControl` string, e.g. `900+10`.
- `f"{how[:12]}"` `FAINT` — lowercased PGN `Termination`.

**Subtitle** (right-aligned in the border), counted over **all** rows held, not the displayed slice:
`"last {w+d+l}: "` `FAINT` + `"{w}W"` `GOOD` + `" "` + `"{d}D"` `DIM` + `" "` + `"{l}L"` `BAD`.
Games with verdict `none` are excluded from all three counts.

---

## A. THEME

`/home/nomad/chess-gpu/scripts/dash/theme.py`. **It is gruvbox dark for all UI chrome**, with three custom families layered on: a board palette, a piece-ink palette, and an evaluation-gauge palette that is explicitly *not* gruvbox because gruvbox's own values failed a measured contrast requirement.

### A.1 Named colours

| name | hex | semantic | used by |
|---|---|---|---|
| `BG` | `#282828` | gruvbox dark0; every panel's background | `_panel(style="on BG")`, and named explicitly on **every** styled run in the whole dashboard |
| `BG_SOFT` | `#32302f` | soft ground | declared, no live use (it is the colour the eval gauge was rejected *from*) |
| `BG_HARD` | `#1d2021` | hard ground | declared, unused |
| `FG` | `#ebdbb2` | primary text | panel titles, values, names, SANs, tape text |
| `DIM` | `#928374` | labels, secondary text | field labels, opponent names, unmarked ladder rows, board coordinate labels |
| `FAINT` | `#665c54` | chrome, borders, ranges | panel borders (default), axis labels, timestamps, ranges, no-op grade marks |
| `ACCENT` | `#fabd2f` | gruvbox yellow — "ours, the thing to look at" | SUMOFISH badge bg, ratings, to-move marker, SEARCHING badge bg, chosen ladder row, active panel border, `COAST` track, puzzle-accuracy value, `move` tape kind |
| `GOOD` | `#b8bb26` | green — good/won/healthy | wins, material plus, unit dot ok, positive rating delta, `LIVE` track, chart above 0.5, `result` tape kind |
| `BAD` | `#fb4934` | red — bad/lost/alarm | losses, clock < 10 s, mistakes and blunders, temp > 80 °C, time gauge > 90 %, MATE badge bg, `LOST` track, chart below 0.5, throttling, dead unit, `warn` tape kind |
| `INFO` | `#83a598` | blue — informational trends | header rating sparklines, training loss sparkline, `game` tape kind |
| `COOL` | `#8ec07c` | aqua — resources with headroom | time-budget gauge, training progress bar, GPU utilisation bar, run name |
| `WARM` | `#fe8019` | orange — caution, not failure | provisional-rating `?`, restart counts, inaccuracy mark, puzzle-accuracy sparkline |
| `PLUM` | `#d3869b` | purple — secondary emphasis | player titles (BOT/GM), first PV move, VRAM bar, `engine` tape kind, moves-panel eval sparkline |
| `BOARD_LIGHT` | `#bdae93` | light square (text renderer) | `board.square_colours` |
| `BOARD_DARK` | `#7c6f64` | dark square | " |
| `LAST_LIGHT` | `#a9a03f` | last move, light square — the same pair shifted green | " |
| `LAST_DARK` | `#79762f` | last move, dark square | " |
| `CHECK_LIGHT` | `#cc5b47` | king in check, light square — shifted red | " |
| `CHECK_DARK` | `#9d4436` | king in check, dark square | " |
| `CHECK_RING` | `#fb4934` | hard ring on the checked square's outer pixels | `board._render_pixel` |
| `PIECE_W` | `#f9f5d7` | white piece light ink (fill) | `board`, via `ink.blend` |
| `PIECE_W_EDGE` | `#20211f` | white piece dark ink (outline) | " |
| `PIECE_B` | `#16181a` | black piece dark ink (fill) | " |
| `PIECE_B_EDGE` | `#ebdbb2` | black piece light ink (detail lines) | " |
| `LIVE` | = `GOOD` | field is being fed | `widgets.TRACK_COLOUR` |
| `COAST` | = `ACCENT` | field is coasting on its last value | " |
| `LOST` | = `BAD` | field is erroring or long silent | " |
| `EVAL_WHITE` | `#ffffff` | our share of the evaluation gauge, and the advantage figure | `evalbar`, `evalbar_h`, `curve_panel` head |
| `EVAL_BLACK` | `#504945` | the gauge's empty half — **a ground, not an opposing colour** | `evalbar` |
| `EVAL_MID` | `#7c6f64` | the chart's 0.50 line | `curve_chart` |
| `GAUGE_MAX` | `44` (not a colour) | cap on any horizontal gauge's width | every `bar()` call site |
| `BLOCKS` | `" ▏▎▍▌▋▊▉█"` | eighth-**width** cells, left-filling | `bar()` |
| `SPARK` | `"▁▂▃▄▅▆▇█"` | eighth-**height** cells | `sparkline()` |

Sixel board colours are separate and deliberately **not** themed (`sixel.COLOURS`): `square light #f0d9b5`, `square dark #b58863`, `square light lastmove #cdd26a`, `square dark lastmove #aaa23a`, `margin #2b2724`, `coord #e5e0d5`. These are lichess's own values — the whole point of the sixel path is that the board is the one from the website. The panels around it stay gruvbox.

### A.2 Documented load-bearing contrast and colour rules

1. **The 1.12:1 gauge (the Lab Note).** `EVAL_BLACK` was `#32302f`, which measures **1.12:1** against the `#282828` panel — indistinguishable. A gauge reading 95 % then looked like a stripe floating in space rather than a bar filled nearly to the top, and a full gauge was indistinguishable from an absent one. `#504945` is **1.67:1** against the panel and **8.9:1** against a white fill, so both the gauge's *extent* and its *level* are legible. **Rule: a gauge's empty half must be measurably distinct from the panel behind it. Measure it.**
2. **The gauge measures our side, not White's, so it must not use white-and-black ink.** Playing Black and losing would give a mostly-dark bar, which reads as "Black is winning" — chess meaning contradicting the number. The filled half is white (the universal ink for "the side that is winning") whichever colour SumoFish is; the empty half is a neutral ground that means only "not filled".
3. **Every styled run names its background explicitly** (`... on {BG}`), everywhere. `rich`'s `Live` diffs frames, so a block character painted with an implicit background leaves the previous frame's colour behind wherever the new frame is narrower. This is what made the old sparklines read as static noise rather than as a trend.
4. **Board squares and highlights come in light/dark pairs.** Tinting a highlighted square with one flat colour destroys the checker pattern exactly where the eye is being directed. Highlighting keeps the pattern and shifts the hue: `LAST_*` is the neutral pair shifted green, `CHECK_*` the same pair shifted red.
5. **Board neutrals are warm and far apart in luminance, and neither is close to a piece fill** — so a piece never disappears into the square it stands on.
6. **Piece inks are pushed further apart than gruvbox's own fg/bg pair** because the outline is sub-pixel at 16 px and needs all the contrast it can get.
7. **`GAUGE_MAX = 44`.** On a 2560 px-wide terminal an "as wide as the panel" bar is 140 characters, which puts its label so far from its fill that the eye cannot associate the two. A bar is a comparison, not a decoration.

### A.3 `ink.py` — per-pixel compositing (text renderer only)

The board draws each cell as `▀`: foreground = upper pixel, background = lower pixel. Two pixels per cell, each with its own colour, so there is no palette limit and cburnett's antialiasing survives.

Two blends per pixel:
```
ink    = lerp(dark_ink, light_ink, stretch(luminance))
result = lerp(square_colour, ink, alpha/255)
```
`stretch(lum)` is a contrast stretch with `KNEE_LO, KNEE_HI = 90, 190`: `≤90 → 0.0`, `≥190 → 1.0`, linear ramp between. Why: cburnett draws a 1.5-unit outline in a 45-unit box, which at a 16 px sprite is half a pixel and rasterises to a muddy mid-grey very close to the light square — the piece goes soft exactly at its edge. Widening the SVG stroke is the wrong fix (measured: at stroke 2.2 and 3.1 the king loses its cross entirely, because every path in the set shares the stroke width). **Alpha is never hardened** — alpha is the silhouette, and hardening it gives jagged edges.

`alpha == 0` short-circuits to the bare square colour. All three functions (`stretch`, `rgb`, `blend`) are memoised with unbounded LRU caches: 16 k blends per frame for a 128×64 board, but the key space is ~100 distinct (luminance, alpha) pairs × a handful of square colours, so the cache saturates in the first frame and every frame after is dictionary lookups. Output is a `#rrggbb` string, rounded per channel.

---

## B. WIDGETS — `/home/nomad/chess-gpu/scripts/dash/widgets.py`

Two rules run through all of them: **every run names its background**, and **every trend carries its endpoints as text** (a shaded trace without a scale is decoration, so `sparkline` returns its range and every caller prints it).

### `track_tag(field, label=None) -> Text`
Provenance of one field. Marks: `LIVE → "●"`, `COAST → "◐"`, `LOST → "○"`, coloured `GOOD`/`ACCENT`/`BAD` respectively. If `label` given, appends `" {label}"` in `DIM`. **If the track is not `LIVE`, appends `" {ago(field.age)}"`** in the track colour. A number with no tag beside it is a number you cannot act on.

### `sparkline(values, width=24, style=DIM) -> (Text, lo, hi)`
Takes the last `width` values, drops `None`s. Fewer than 2 values → empty `Text` and `(0.0, 0.0)`. `lo, hi = min, max`; each value maps to `SPARK[int((v-lo)/span * 7)]`, or index 0 if `span < 1e-12`. **Auto-scaled to its own window** (unlike the curve chart), which is why the range is returned and always printed.

### `bar(fraction, width, fg, bg=BG) -> Text`
Horizontal gauge at eighth-of-a-cell resolution. Clamps fraction to `[0,1]`; `full = int(fraction*width)` full blocks in `fg`, then one partial `BLOCKS[int(frac_part*8)]`, then spaces to `width`. Sub-cell resolution matters: at whole-cell steps a 20-column bar has 20 states, so a slow-moving quantity appears frozen and then jumps; eighths give 160.

### `evalbar(value, height, width=2) -> list[Text]`
Vertical gauge for the side at the bottom of the board. **`value is None` returns `height` rows of completely unstyled spaces** — nothing at all, rather than a dark column beside the board. Otherwise `filled = round(clamp(value)*height*8)`; rows are emitted top-down but fill bottom-up (`base = (height-1-r)*8`, `k = clamp(filled-base, 0, 8)`): `k==0` → space on `EVAL_BLACK`, `k==8` → space on `EVAL_WHITE`, else `EIGHTHS[k]` in `EVAL_WHITE on EVAL_BLACK`. Eighths, not halves: a 70-row half-block bar has 140 positions, and most of a game is spent between 0.45 and 0.60 — 20 of them — so small changes land on the same cell. Eighths give 560. The caller passes the **eased** value.

### `evalbar_h(value, width) -> Text`
Horizontal variant, filling left, same eighths logic with `EIGHTHS_H`. **Currently not imported by `panels.py`** — it is the legacy gauge from when the bar lived on a board row. Keep only if the rewrite reintroduces a horizontal eval bar; the reason it exists (rich diffs by line, so a gauge sharing a row with the sixel image erases it) is in §E.

### `_catmull_rom(points, samples) -> list[float]`
Resamples a series through a Catmull-Rom spline. With 20 moves stretched across 200 columns, linear interpolation is visibly straight segments meeting at corners, and the corners read as evaluation features that are not there. Catmull-Rom **passes through every real point** — it invents shape between them, never at them — which is the honest kind of smoothing here. Returns `points` unchanged if `n < 2` or `samples < 2`.

### `curve_chart(values, width, height, mid=0.5) -> list[Text]`
Braille line chart. Braille gives a 2×4 dot grid per cell: 4× the vertical and 2× the horizontal resolution of half-blocks. The cost is colour — one foreground per cell.
- Drops `None`s; fewer than 2 points returns `height` empty `Text`s.
- Pixel grid `px_w = width*2`, `px_h = height*4`. Resamples to `px_w` samples.
- `y = round((1 - clamp(v)) * (px_h-1))`; colour `GOOD` if `v >= mid` else `BAD`.
- **Consecutive samples are joined** by plotting every `y` between the previous and current, so the trace is continuous.
- Cell colour is last-writer-wins; the trace crosses level rarely, so only crossing cells can misreport.
- The midline is drawn in `EVAL_MID` **only in cells the trace does not occupy** (a braille cell has one colour so the two cannot share), as both dots of the appropriate sub-row.
- Empty cells are spaces styled `on BG`.
- **Fixed 0..1 scale, never auto-scaled.** Renormalising makes a dead-level draw look like a collapse, which is the standard way an evaluation chart lies. Vertical distance from the midline means the same thing in every game.

### `ladder(rows, width, fg, dim=DIM) -> list[Text]`
Depth ladder over 4-tuples `(san, visits, q, prior)`. Empty input → empty list. `top = max(visits) or 1`.
Per row: `f"{san:<7}"` (`bold fg` for row 0, else `dim`) + `bar(visits/top, width, fg if row 0 else dim)` + `f" {visits:>6}"` `dim` + `f"  q {q:.3f}"` `dim` + `f"  p {prior:.3f}"` `dim`.
**Sorted by the resource actually committed (visits), not by score**, because the search picks its move by visit count. One long bar means a forced move; five stubs of equal length mean the engine has no idea.

---

## C. BOARD RENDERING

### C.1 Text / half-block renderer — `/home/nomad/chess-gpu/scripts/dash/board.py`

**Cell geometry.** A terminal cell here is ~8.0 × 14.9 px, i.e. about 1:2, so *two cells wide by one tall is square*. `▀` stacks two pixels in one cell, so a square of `w × h` cells is a `w × (h*2)` pixel bitmap. Block elements also have no `wcwidth` ambiguity, whereas `♞` and Nerd Font codepoints do — and that ambiguity drifts table borders row by row.

**Scales** (`SCALES`: cells wide, cells tall, label gutter, has bitmap), tried largest-first by `ORDER`:

| scale | cells/square | sprite px | gutter | bitmap |
|---|---|---|---|---|
| `pixel5` | 40 × 20 | 40 | 5 | yes |
| `pixel4` | 32 × 16 | 32 | 5 | yes |
| `pixel3` | 24 × 12 | 24 | 4 | yes |
| `pixel2` | 16 × 8 | 16 | 4 | yes |
| `pixel1` | 12 × 6 | 12 | 4 | yes |
| `glyph3` | 8 × 4 | — | 3 | no |
| `glyph2` | 4 × 2 | — | 3 | no |
| `glyph1` | 2 × 1 | — | 2 | no |

`board_size(scale) = (cells_w*8 + gutter, cells_h*8 + 1)` — the `+1` row is the file labels. `pick_scale(width, height)` returns the first scale in `ORDER` that fits, defaulting to `glyph1`.

Below 12 px cburnett stops being legible (king and queen collapse into two blobs), which is why the small scales draw a Unicode figure instead of a worse bitmap. Available sprite sizes are exactly `(12, 16, 24, 32, 40)`.

**`_squares(flip)` is the single coordinate authority.** It yields `(rank, [(square, file), ...])` in draw order — ranks `0..7` and files `7..0` when flipped, ranks `7..0` and files `0..7` otherwise. Highlights, the check marker and the coordinate labels all read from the same iterator, so flipping cannot leave an overlay on the wrong file. This must stay structural: it is the classic way a flipped board goes subtly wrong.

**Square colours** (`square_colours(sq, marked, in_check)`): a square is *light* iff `(file + rank) % 2 == 1`. Priority: check → `CHECK_LIGHT/DARK`; else marked → `LAST_LIGHT/DARK`; else `BOARD_LIGHT/DARK`.

**Last-move highlight** (`highlight_squares`): `{from_square, to_square}`, **plus, for a castling king move (file delta ±2), the rook's origin and destination** (`h`→`f` for short, `a`→`d` for long). Without this the rook appears to have teleported.

**Check indicator.** `checked = board.king(board.turn) if board.is_check() else None`. In the pixel renderer the checked square is drawn normally (with `CHECK_*` background) and then a **hard `CHECK_RING` ring is painted on the square's outer pixel ring, over everything** — `_ring(px, w, h)` is true for `x == 0 or x == w-1 or y == 0 or y == h-1`. Reason (Lab Note): the king covers almost all of its own square, so tinting the background hides the warning behind the piece the warning is about. A square that is empty but checked is still rendered (the loop does not skip it) so the ring appears. The glyph renderer has no ring — it uses the `CHECK_*` background only.

**Pixel renderer** (`_render_pixel`). Per rank, `ch` cell-rows; per cell-row two pixel rows `ty = cell_row*2`, `by = ty+1`. Rank label is `f"{rank+1}".center(gutter)` on cell-row `ch//2 - 1`, else `gutter` spaces, in `DIM`. Per square: empty and not in check → `cw` spaces on the square colour. Otherwise per column `x`, look up the sprite `SPRITES[(sprite_px, piece.symbol())]` (2 bytes per pixel, row-major from top-left: luminance then alpha; index `(y*cw + x)*2`), pick inks — white piece → `(dark, light) = (PIECE_W_EDGE, PIECE_W)`, black piece → `(PIECE_B, PIECE_B_EDGE)` — and `blend` each of the two pixels against the square colour. **If both halves come out the same colour, emit a plain space with that background** rather than a redundant `▀` glyph. A missing sprite degrades to a flat square (`fg_c = bg_c = bg`).

**Glyph renderer** (`_render_glyph`). Label on cell-row `(ch-1)//2`; the piece is drawn only on that same row (other rows are blank squares). Glyph from `{K:♚ Q:♛ R:♜ B:♝ N:♞ P:♟}` on `piece.symbol().upper()` — **the same outline glyph for both colours**, distinguished by ink: `PIECE_W` or `PIECE_B`, bold. Centred with `left = (cw-1)//2` spaces before and the remainder after.

**File labels** (`_files_row`): `gutter` spaces, then `FILES[f].center(cw)` for `f` in `0..7` (or reversed when flipped), in `DIM`.

**Cache**: `render()` memoises on `(fen, flip, last_uci, scale)` and **clears the whole cache when it exceeds 8 entries**. A 128×64 board costs ~5 ms and the position changes at most once a second.

### C.2 Sixel renderer — `/home/nomad/chess-gpu/scripts/dash/sixel.py`

**Why it exists**: half-blocks cap resolution at one sprite pixel per cell (~8 screen px per pixel), which looks like Atari art however carefully it is drawn. Sixel removes the ceiling entirely — the board becomes the actual cburnett artwork, rendered from the same SVGs as lichess. It was ruled out early on the untested belief that Konsole's sixel support is off by default; that was wrong.

**Capability probe** (`probe(cols, rows) -> Caps | None`). Returns `None` for *any* reason, because every reason has the same consequence (draw text instead):
1. `sys.stdin` and `sys.stdout` must both be ttys.
2. `rsvg-convert` **and** `magick` must both be on `PATH`.
3. Primary device attributes: send `ESC[c`, wait for `c`; require `";4"` among the parameters (Konsole answers `[?62;1;4c`).
4. Text area in pixels: send `ESC[14t`, parse `\[4;(\d+);(\d+)t` → `(height_px, width_px)`.
5. `Caps(cell_w = w_px/cols, cell_h = h_px/rows)` — fractional on purpose.
6. `_drain()` afterwards: swallow any trailing byte, or it is echoed once the tty leaves raw mode (it turned up as a stray `t` in the bottom-left corner).

`_ask` must use **raw mode** (otherwise the reply is line-buffered, never arrives, and is echoed into the display) and **`os.read(fd, 64)`, never `sys.stdin.read`** — Python's text layer pulls the whole reply off the fd, returns one character and keeps the rest, so `select` reports nothing pending, the loop exits with a bare ESC, and the remainder surfaces inside the *next* query's answer, which reads exactly like a terminal that does not support the query.

**The 390-unit geometry and the multiple-of-26 rule.** `chess.svg.board` draws into a 390-unit square: eight 45-unit squares (360) plus a 15-unit margin on each side. A square edge therefore lands on an exact pixel only when the raster size is a multiple of `390/15 = 26`. At any other size rsvg blends the two square colours across the boundary and the board grows a hairline grid it does not have — which then *moves*, because which edges get a blended pixel depends on rounding, and the quantiser picks a different colour for it each frame. Measured on an empty board, counting scanline pixels that are neither square colour: **eight per line at 1152 px, zero at 1144 px**. `snap(size_px) = max(26, size_px - size_px % 26)` — always snap **down**; the most it costs is 25 px of board. Snapping happens in `Plan`, not in the renderer, so the reserved rows and columns are derived from the size the picture will actually be.

**Render pipeline** (`render(board, flip, last, size_px)`):
1. `chess.svg.board(board, orientation=BLACK if flip else WHITE, lastmove=last, check=board.king(board.turn) if board.is_check() else None, size=size_px, coordinates=True, colors=COLOURS)`. So: last-move highlight and check marker are python-chess's own, coordinates are on, and the colours are lichess's.
2. `rsvg-convert -w N -h N -f png` (measured ~54 ms).
3. `magick png:- -dither None -colors 64 sixel:-`. **Dithering must be off**: error diffusion over a picture this flat buys nothing, is recomputed from the whole image every render so the noise lands differently after every move (the board visibly shimmers), and is **three times slower** — 286 ms with dithering vs 96 ms without at 1144 px, 100 ms without at 64 colours. 64 colours costs 4 ms over 2 and buys smoother piece edges.
4. Any exception → `None` and a debug log line; the caller keeps the previous image.
5. Cache keyed `(fen, flip, last_uci, size_px)`, cleared wholesale above 4 entries.

Total ~400 ms, essentially all in the sixel encoder.

**`Renderer` thread.** `want(key, board, flip, last, size_px)` is cheap and safe to call every frame; it replaces any pending request with the same key ignored. `run()` waits on an event, renders, and **publishes only if nothing newer was asked for meanwhile** (`ready_key`, `ready_data`). Newest position always wins; intermediate positions are dropped, never queued. Verified: 16 positions requested at 100 ms intervals, the renderer settles on the final one. This bounds lag at one render however fast the game is — inline encoding was quietly catastrophic for bullet, where the bot moves faster than 400 ms and every move queued behind the last.

**`emit(row, col, data)`** — 1-based cell position:
```
sys.stdout.flush()          # MUST flush rich's text layer first
buffer.write(f"\x1b[{row};{col}H")
buffer.write(data)
buffer.write(b"\x1b[H")     # park the cursor; sixel leaves it wherever the image ended
buffer.flush()
```
Both orderings are load-bearing: without the first flush, rich's pending frame is still in the `TextIOWrapper` and gets flushed *after* the image, painting blank cells straight over the picture — which looks exactly like sixel not working at all. And `rich` assumes it owns cursor state, hence the park.

**`clear()`** — `ESC[2J ESC[H` (after the same flush). Verified to erase image data in Konsole. Called only when the whole layout is being rebuilt, because nothing here knows how to erase just the old image's region.

**`MARGIN_CELLS = 1`** — one cell of slack so the image never lands on a panel border, which sixel would happily paint over.

### C.3 Sprite generation — `make_sprites.py` → `sprites.py`

`sprites.py` is **generated and checked in** so that drawing a board never requires `rsvg-convert` or ImageMagick; they are needed once, by whoever changes the artwork. Format: base64 of two bytes per pixel, row-major from top-left, **luminance then alpha**. Luminance 255 is the piece's *light* ink (a white piece's fill, a black piece's detail lines), 0 is its dark ink; the renderer chooses what those inks are. Alpha is coverage against the square, kept at full resolution because it is the antialiasing. Sizes `(12, 16, 24, 32, 40)`, symbols `KQRBNPkqrbnp`, from `chess.svg.PIECES` (cburnett, CC BY-SA, already a dependency) wrapped in a `viewBox="0 0 45 45"` SVG.

Do not hand-draw pieces. The hand-drawn 8×8 set that preceded this is gone and must not be reintroduced.

---

## D. INTERACTION, CLI, AND THE DEMO FIXTURE

### D.1 `Keys` — `watch.py:435`

Reads single keypresses without blocking the render loop.

- `fd = sys.stdin.fileno()` if stdin is a tty, else disabled entirely (`get()` returns `""`).
- **`tty.setcbreak`, not `setraw`** — raw would swallow Ctrl-C.
- Saves and restores termios attributes with `TCSADRAIN`; every termios call is individually guarded so a non-tty or a hostile terminal disables key handling rather than crashing.
- `get()`: `select` with a 0 timeout, then `os.read(fd, 8).decode("ascii", "replace")`. Non-blocking, one call per frame.

**Bindings (the complete set, `watch.py:665-676`):**

| key | effect |
|---|---|
| `q` | quit the loop |
| `\x03` (Ctrl-C) | quit the loop (same branch; `KeyboardInterrupt` is also caught) |
| `r` | full repaint: set `painted = None`, `sixel.clear()` if sixel is active, `live.refresh()` |

Nothing else is bound. `r` exists because there is exactly one thing the loop cannot detect for itself: whether something else has painted over the board image. `rich` leaves the region alone as long as it renders identically, and a resize rebuilds and re-emits, but any other forced repaint destroys the picture and nothing changes the position, so nothing redraws it. The erase comes *first*, because the thing `r` is for is junk on screen and a redraw that writes nothing over blank rows cannot remove any of it. It costs a keypress rather than a periodic flicker.

**Exit path** (the `finally` block, belt and braces because `Live`'s context manager only restores the screen if it gets to run): write `ESC[?25h` (show cursor) and `ESC[?1049l` (leave the alt screen), flush, then `restore_resize(original)`. A hard failure must never leave a terminal that needs `reset` typed into it blind.

### D.2 CLI flags — `watch.py:570`

| flag | type / default | effect |
|---|---|---|
| `--no-resize` | store_true | do not ask the terminal to resize, and do not restore a size on exit |
| `--small` | store_true | force the compact (narrow) layout regardless of terminal size |
| `--demo` | store_true | seed the fixture and run the demo loop; **no source threads start** |
| `--board FRACTION` | float, default `BOARD_SHARE = 0.65` | share of terminal width the board may take |
| `--no-image` | store_true | never probe for or draw a sixel image; always the text renderer |
| `--user NAME` | str, default `"SumoFish"` | the account whose profile, games and colour are read |

Startup requires a token: `LICHESS_BOT_TOKEN` from the environment, else the first `LICHESS_BOT_TOKEN=` line of `~/.config/chess-gpu/bot.env`; if neither, exit with `"LICHESS_BOT_TOKEN not set (see ~/.config/chess-gpu/bot.env)"`. **This applies to `--demo` too** — the fixture path still requires a token today. The token is used for exactly two endpoints (`/api/account`, `/api/account/playing`) and is never printed.

`Console()` is created with **no console-wide background** — it would style every cell including those under the sixel image, and rich rewrites styled cells every frame. Each panel sets its own background; the terminal's own scheme shows through the gaps.

**Resize negotiation.** `request_resize`: if the terminal is already at least `WANT_COLS × WANT_ROWS` (150 × 44), do nothing and return `None`. Otherwise write `ESC[8;{max(rows,44)};{max(cols,150)}t` (the xterm resize sequence, honoured by Konsole), sleep 0.35 s because the resize is asynchronous, and return the original size so `restore_resize` can put it back on the way out — leaving someone's terminal a different shape than you found it is rude.

**Source threads started (non-demo only), all daemons:** `Profile(user, token)`, `Playing(token)`, `GameStream`, `EngineTail(logs/engine.jsonl, user)`, `TrainTail(ROOT)`, `RatingLog(logs/rating.jsonl)`, `Results(logs/games, user)`, `Grader(ROOT)`, `Gpu`, `Units`, `Finished(user)`.

**Main loop order per frame** (`watch.py:664`): read a key and act; compare `console.size` to the last one and rebuild the `Plan` + `sixel.clear()` + `painted = None` on any change; `draw(plan, state)`; `live.update(plan.layout)`; if sixel is active, `board_image_tick`, and only when the painted key actually changed, `live.refresh()` followed by `sixel.emit(...)`; `sleep(1/12)`.

`board_image_tick` never blocks: it calls `renderer.want(...)` and emits `renderer.ready_data` **only if `ready_key != painted` and `_fits(ready_key, plan)`** — i.e. only if the finished image was rendered for the layout currently on screen (`key[-1] == plan.image_px`). Otherwise the board is briefly absent rather than briefly wrong.

The demo loop differs: `refresh_per_second=2`, `sleep(0.2)`, no `Keys`, no key handling — exit is Ctrl-C only. It **does** rebuild the `Plan` on every size change, deliberately: the fixture exists to check the layout at whatever size the window happens to be, and an earlier version built the plan once, so dragging the window showed a board sized for the window before last.

### D.3 `seed_demo(state)` — `watch.py:371`

Purpose: a dashboard that can only be inspected while a game happens to be running is a dashboard whose layout gets checked once and never again. This is the fixture that makes rendering testable on demand. **It writes only into the state object; no source thread runs, so nothing in it can be mistaken for live data.** It is also the fixture `tests/verify_layout.py` renders against.

What it fakes:

| key | contents |
|---|---|
| `playing` | one entry: `gameId "demo0000"`, `color "white"`, `speed "blitz"` |
| `game` | id `demo0000`; players White = `SumoFish`/`BOT`/2513, Black = `TopasBot`/`BOT`/1912; a 38-ply Sicilian played out from the SAN list so `board`, `last` and `moves` (with `san`, `uci`, `ply`) are all real and mutually consistent; `wc = 94.2`, `bc = 41.8`, `clock_at = now` (so the White clock visibly ticks) |
| `engine` | `ev "think"` (so the panel shows the SEARCHING state and an `ACCENT` border), matching `ply`/`fen`/`stm` so `_engine_matches` passes, `wp = wp_white = 0.6183`, `nodes 5219`, `nps 4103`, `sims 5248`, `elapsed 1.27`, `budget 2.10` (a 60 %-drained time gauge), `best "axb3"`, an 8-move PV, a **6-row `top` ladder** with a dominant leader (3211 vs 902 …) so the bar shape is exercised, `mate False`, `done False` |
| curve | 11 points via `record_eval(i*4, wp)` for wp `.5 .51 .49 .52 .55 .53 .58 .56 .61 .6 .62` — enough for the braille chart and the sparkline |
| `profile` | bullet 2513/4g/rd199/provisional, blitz 1794/2g/rd290/provisional — exercises the `?` and `±` paths; `count` 1W/0D/25L of 26 |
| `rating_log` | 8 bullet samples `3000 → 2544` — exercises the header sparkline and a negative delta |
| `gpu` | util 97 %, 10040/16376 MiB, 72 °C, 248 W — a busy card below the 80 °C alarm |
| `units` | both `chess-gpu-bot` and `chess-gpu-train` active/running/0 restarts |
| `train` | run `9M-sv-warm-full`, 65 synthetic loss points on a decaying curve at 9150 samples/s, 7 puzzle-accuracy evals `0.486 → 0.614` |
| tape | three notes: `game "demo fixture, not a live game"`, and two `move` lines |

Deliberately **not** faked: `eval_curve`, `grades`, `results`, `record`, `finished`. So the demo shows the engine-only curve (`eval·`), no grade marks, and the "no finished games yet" results panel — the degraded paths are what you see. `Field.fills` is 1 for everything it sets and 0 for everything it does not, so track tags read `LIVE` initially and decay to `COAST`/`LOST` as the fixture ages, which is itself worth preserving as a way to eyeball the staleness rendering.

### D.4 Layout (`Plan`) — the numbers, for completeness

Constants: `WANT_COLS/ROWS = 150/44` (the *ask*), `MIN_WIDE_COLS = 100` (the *floor* — a different question that must not be answered with the same number), `BOARD_COLS = 74`, `BOARD_SHARE = 0.65`, `RIGHT_COLS = 60`, `TAPE_ROWS = 7`, `BOARD_GUTTER = 2`, `PLAYER_ROWS = 2`.

Top level: `head` (size 3) over `main` (`main_h = rows - 3`). `wide = not small and cols >= 100 and rows >= 38`.

**Wide layout:**
- `budget_w = min(cols - RIGHT_COLS, int(cols * share))`; `player_h = 4`; `budget_h = main_h - 4`.
- With sixel caps: `cell_cols = budget_w - 2*BOARD_GUTTER`; `cell_rows = budget_h - 1 - MARGIN_CELLS - TAPE_ROWS`; `image_px = snap(min(cell_cols*cell_w, cell_rows*cell_h))`; `image_rows = max(1, int(image_px/cell_h) + 1)`; `image_cols = int(image_px/cell_w)`; `board_w = image_cols + 2*BOARD_GUTTER`; `board_h = image_rows + 4`; `image_at = (4 + PLAYER_ROWS, BOARD_GUTTER + 1)` (1-based).
- Without: `scale = pick_scale(budget_w - 3, budget_h)`; `board_w = min(budget_w + 7, bw + 7)`; `board_h = bh + 4`.
- `right_w = cols - board_w`. Left column splits into `board` (size `board_h`) and `tape` (size `main_h - board_h`), unless that is `< 4`, in which case the tape is dropped and the board takes the whole column.
- Right column: `train_h = 6`, `machine_h = 5`, `h = main_h - 11`; `mind_h = max(12, min(16, h//2))`; `results_h = 8 if h - mind_h >= 20 else 0`; `moves_h = max(8, min(34, h - mind_h - results_h))`; `curve_h = h - mind_h - moves_h - results_h`, and if that is `< 7` the curve panel is dropped and its rows go to `moves`. Order top to bottom: mind, moves, curve, results, train, machine.

**Narrow layout:** `board_w = min(46, max(24, cols - 34))`; body splits row-wise into board and right; right splits into `mind` (`body_h - moves_h`) over `moves` (`max(6, body_h//3)`); `tape_h = 0 if rows < 30 else 5`; `train_h = 0 if rows < 26 else 6` and when present splits row-wise with `machine` (`min(48, max(24, cols//3))`); `curve_h = 0` (no evaluation panel); no results panel. Below the floor it **drops panels rather than clipping them, because a clipped panel is a lie and a missing one is not.**

Geometry invariants gated by `/home/nomad/chess-gpu/tests/verify_layout.py` (no torch, no lichess, no terminal — `Caps` is constructed directly, at cell 8.0 × 14.9, across sizes 160×96, 150×44, 160×48, 200×60, 320×90, 120×40, 90×30, 80×24):
1. The rows the panel renders equal the rows the plan reserved (`len(lines) == board_h` in the wide layout, `<=` in narrow).
2. `board_h + tape_h == main_h` — the column fills the screen.
3. `right_w >= RIGHT_COLS` at every size.
4. Every rendered line is **exactly** `board_w` cells, so no line bleeds into the panel beside it.
5. The picture is centred: left gutter == right gutter; `image_at[0] == 4 + PLAYER_ROWS`; the image's last row is at or above `3 + board_h`.
6. A player block is the same height for the starting position, one capture, queens-and-a-rook down, and bare kings.
7. A wider terminal gives a bigger board; a shorter one is height-bound; both keep the right column.
8. `results_panel` renders `"time forfeit"` in full at every width 52..99.
9. The header fits every width the wide layout allows.

---

## E. LOAD-BEARING CONVENTIONS — RULES THE REWRITE MUST HONOUR

Each rule is stated as an obligation, with the reason and the observed failure.

### E.1 Frames, signs and probabilities

**R1. `panels.ours()` is the single White→our-side conversion. Every displayed probability goes through it.**
Reason: converting at each call site means one will be missed. Observed: with SumoFish playing Black and being mated, the move list read 0.97 while the chart read 0.03 — both correct in their own frame, and the one that happened to be White's said we were winning. `ours(state, wp_white) = 1 - wp_white if we are black else wp_white`, and `_we_are_black` derives our colour from the game's own player list.

**R2. Never plot `win_prob` or any Q straight from telemetry.**
Reason: MCTS stores values from each node's **own side-to-move perspective**, so a series across alternating plies is a sawtooth, not a trend. The records carry both `wp` and `wp_white` precisely so no consumer has to remember. `state.curve` is documented as White-framed and `record_eval` takes `wp_white`. The one exception is `mind_panel`'s `"{wp}% to move"`, which is labelled with its frame.

**R3. The evaluation gauge and the evaluation chart must share one frame and one scale.**
They used to disagree — the gauge measured the side at the bottom of the board while the chart plotted White — which is invisible until we play Black and then they point opposite ways. The whole `curve_panel` is now "chance SumoFish wins", and the gauge is the right-hand end of the curve drawn tall enough to read.

**R4. The advantage figure must be a monotone function of the probability, not a second opinion.**
`400*log10(p/(1-p))/100`, the same mapping `search_engine.centipawns` uses over UCI, so the number matches what a lichess analysis board shows for our moves. Copied, not imported, because that module pulls in torch. Signed from our point of view, which is why it is drawn in the gauge's ink and not in black and white.

**R5. Fixed 0..1 scale on the evaluation chart, always; the midline is 0.50.**
Renormalising to the game's own range makes a dead-level draw look like a collapse. That is the standard way a chart of this kind lies.

**R6. A bar may ease; a number may not.**
`Smooth` keeps `.value` (eased, for the gauge) and `.target` (raw, for the figure). A figure must never show a value the engine did not produce. Easing is against the wall clock, not the frame count, so it looks the same whether the loop is keeping up.

### E.2 The two-games-one-log split

**R7. `logs/engine.jsonl` is not one stream. It is N streams interleaved and must be demultiplexed before anything is rendered.**
`concurrency: 2` means lichess-bot plays two games at once, spawns an engine per game, and both append to the same file. Measured in one log: **468 places where consecutive searches alternate between two positions**. Read as one stream it put both games' plies on one evaluation curve — *the graph that says one side is winning when it is not* — flipped the search panel between two boards, and let the board picture jump to the other game.

The demultiplexer, in priority order (`sources.EngineTail.tick` + `fusion.EngineBoard`):
1. **Explicit claim.** The engine stamps `pid` and the lichess `game` id on every record (`chessgpu/telemetry.py`), the id arriving over UCI as `setoption name GameId` from lichess-bot's `extra_game_handlers` hook. `_claims(rec, gid)` returns `True`/`False`/`None`. `True` → accept unconditionally, no history to satisfy. `False` → drop.
2. **Side to move.** The engine only searches on its own turn, so every record from our game has us to move. When the other game has us on the other colour — about half the time — `stm` alone separates them, *including in the opening where both games can be in the same position and the history cannot tell them apart*. `_our_side` must come from the **stream's** description of the game on screen, never from `nowPlaying[0]`.
3. **Position history.** `EngineBoard.update(rec, history)` accepts a record if it bridges from the last accepted position within `MAX_BRIDGE_PLIES = 2` (the opponent's reply plus ours), or if the streamed game has ever been in that placement. Anything else goes into a 48-deep **orphan deque** rather than being adopted — the engine is ~9 s ahead of the stream, so confirmation for a genuine record simply has not arrived yet. `reanchor(history)` adopts held records once the stream catches up and replays the rest in order, so a wrong guess self-corrects within about one stream lag instead of sticking.
4. Curve de-duplication is keyed on `(ply, placement)`, **not on ply alone**: two games are at ply 12 as often as not, and keying on the number meant that once a record from the other board slipped through, our own record for that ply was discarded as a repeat and the other game's evaluation stayed on the curve.
5. `record_eval` resets the curve when a ply arrives below the current maximum, and `EngineTail` calls `state.clear_curve()` and `eboard.reset()` on a game id change.

**R8. Never treat `nowPlaying[0]` as "the game we are watching".** With two games in progress lichess orders that list however it likes and the order changes on its own. `GameStream` stays on the current game **for as long as it is still anywhere in the list**, and re-checks membership *inside* the read loop (the outer loop is parked in `urlopen`).

### E.3 Staleness and honesty

**R9. Staleness lives in the data model, not the styling.**
`Field.track` is live/coast/lost and every panel prints it. The failure it exists to prevent is the old loop's `profile = api(...) or profile`, where a dead connection rendered **pixel-identical** to a live one; you could stare at a frozen board for twenty minutes with no way to tell. You cannot read a value out of `State` without also being handed how old it is.

**R10. The healthy-interval constants must track the real source cadences**, or the live/coast/lost tag lies. A pushed source (`game`, `engine`) gets a generous interval because silence between moves is a player thinking, not a fault.

**R11. Never show an evaluation for a position that is not on the board.** `_engine_matches` gates the player-line win estimates, and `watch._eval_target` returns `None` — blanking the gauge and printing `--` — whenever the engine's record does not describe the live position. A stale eval rendered as current is the confidently-wrong-number failure.

**R12. A source failing must degrade one panel, not the screen.** Every `Source.tick` is wrapped; an exception marks the field failed and the thread continues. Panels do no I/O and never block. The render loop is a pure function of state.

**R13. Never iterate a dict a source thread writes to.** `moves_panel` walked `state.curve` directly while `record_eval` inserted into it, which raises `dictionary changed size during iteration` inside `draw` — nothing catches that, so the dashboard exited mid-game. Snapshot under the lock (`curve_series`, `curve_items`).

**R14. The engine narrates; the dashboard only listens.** The viewer never imports torch, never evaluates a position, and therefore cannot disagree with the engine or take GPU time from it. Measured cost to the engine: none detectable. Corollary: the Stockfish grader is CPU-only, depth-capped at 12, on its own thread, never in a panel.

**R15. The channel is an append-only file, never a fifo or socket.** `open(fifo, "w")` blocks until a reader attaches, which would put an unbounded stall inside `choose()` on a running chess clock whenever the dashboard was not running. The engine must never be able to block on whether anyone is watching.

**R16. Tail by inode, not by path**, and reopen on inode change or truncation. `logs/engine.jsonl` rotates at 16 MB and a path-only tailer goes quiet forever after the first rotation while continuing to render its last value — indistinguishable from a quiet game.

**R17. Open the log `rb` and decode after splitting.** `seek(-n, SEEK_CUR)` on a text handle raises `io.UnsupportedOperation: can't do nonzero cur-relative seeks` because a text handle's position is an opaque cookie. The tailer does exactly that to hand a half-written line back to the writer, so in text mode every torn write raised, lost the record, and left the handle mid-line so the next read lost another.

**R18. Read back over the log on attach.** A tailer that starts at the end with nothing to fill in what it missed showed an empty search panel, an empty chart, and a board that would not move until the engine's *next* move — and on a game whose opponent then flags, it never showed one thing about that game. "It opens and is frozen" is what that looks like, and it is indistinguishable from actually frozen. `_backfill` reads the last 256 KB, keeps only `move` records, filters them through the same claim/side/history rules, and absorbs them `quiet=True` so they do not enter the tape.

**R19. Never publish a position earlier than one already shown for this game.** The lichess stream replays the whole history on connect, and a game ending closes the stream — so this fires exactly when a game is decided. On screen it is a flicker through some earlier position, gone before it can be read. The replay is still needed to rebuild the move list after a drop; it just must not be shown. `GameStream` tracks the furthest ply published per game id.

**R20. Obey "only make one request at a time."** The lichess docs publish no numeric limits, so that sentence is the only concrete guidance. `sources.GATE` serialises every call and paces them 250 ms apart; the game stream holds the gate only while *connecting*, never while reading (a quiet board would otherwise stall every source for minutes). The machine panel shows `api N/min` so "am I asking too often" is on screen rather than inferred. A 429 backs the source off for a full minute or more.

**R21. Use `/api/account`, not `/api/user/{name}`.** The public endpoint answers 429 from this machine even at one request a minute, because lichess-bot is already talking to lichess from the same IP and the public per-IP budget is shared and small. The authenticated endpoint returns the identical shape for our own account and is budgeted separately. Keep the public one only as a fallback.

**R22. `/game/export/{id}`, not `/api/game/export/{id}`** (the latter is a 404), and send `Accept: application/json` or it serves PGN which the parser rejects as a network error. And remember a game id even after a *failed* fetch, or you re-ask every four seconds forever from the address the bot plays from.

**R23. The header record is this version's, not lifetime.** A lifetime tally averages engines that no longer exist — 24 % across bullet and blitz against 52 % since the current build — and the average describes nothing that ever played. `Results` computes the tally over **every** game in the version and only then caps the displayed list at 40; counting the capped list would under-report the moment a version passed 40 games.

**R24. `Result "*"` is `none`, not `draw`.** Folding aborted games into draws shows a D for a game nobody drew and dilutes the score line with games that were never played.

**R25. Width is allocated, never sliced at a guess.** The results panel measures every fixed column including separators and gives the termination its full 12 characters. "time forfeit" became "time forfei" twice, from two different bugs.

### E.4 Sixel, rich diffing, and the board column

**R26. The sixel image region must be completely bare: no panel, no border, no background, unstyled padding only.**
`rich` skips writing cells that carry no style and rewrites ones that do, so a Panel or a console-wide background over the image region erases the picture between frames — it never appears at all, which looks like sixel being broken. Measured both ways: with a Panel the board never appears, bare it survives indefinitely.

**R27. Nothing animated may live anywhere in the board column**, even on rows the image does not occupy. Two separate corruption bugs came from trying, and a geometry check that says a row is clear does not make the column a good place for it: the image is re-emitted only when the position changes, so anything that redraws between moves is one rich quirk away from eating it. Put moving things in a text panel.

**R28. Nothing styled may share a row with the image, even far to its left.** `rich` diffs by line: change one cell and it rewrites the whole line, writing unstyled padding over the picture. A static styled cell survives because it never changes; an animated one erases the board a row at a time and the next move snaps it back, which looks like the board shifting every move. This is why the evaluation gauge was moved out of the board column entirely and into the search/evaluation panel, beside the number it is a picture of.

**R29. Emit the image only when the position actually changes. Never on a timer.** The terminal clears the region before redrawing, so a re-emit makes the board strobe at the frame rate. `rich` leaves the region alone in between because it renders identically, so once is enough.

**R30. Never emit an image drawn for a different layout.** The plan rebuilds instantly and a fresh render takes ~400 ms, so the frame after a resize would paint the previous, larger board over a screen just wiped for the new one — and its bottom and right edges land in rows the new layout never writes to, where nothing will ever repaint them. On screen: a strip of squares and half a pawn under the player line, surviving every frame afterwards. The size is part of the image key; compare it before emitting (`_fits`), and accept the board being briefly **absent** rather than briefly **wrong**.

**R31. `ESC[2J` on every geometry change**, before the first draw of the new layout (and on the first pass, which cleans up after a previous run).

**R32. Flush the text layer before writing raw bytes.** Rich's pending frame is still in the `TextIOWrapper` and would be flushed *after* the image, painting over it.

**R33. Rasterise only at multiples of 26.** See §C.2. Eight stray pixels per scanline at 1152, zero at 1144.

**R34. `-dither None -colors 64`.** See §C.2. Three times faster and it stops the board shimmering after every move.

**R35. Encode off the render loop, newest-wins, drop the stale.** See §C.2.

**R36. Probe the terminal; do not assume.** Sixel was ruled out here on the belief that Konsole's support is off by default and unreliable. It is neither: `ESC[c` returns `[?62;1;4c` and images render with no setup. That wrong assumption cost the entire half-block renderer, which is now only the fallback path. Conversely: every failure to probe must fall back silently to text, never to an error on screen.

**R37. Use `os.read(fd, n)` for terminal query replies, not `sys.stdin.read(1)`.** See §C.2.

**R38. Draw the check indicator as a ring on the square's outer pixels, over everything. Do not tint the background.** The king covers almost all of its own square, so the warning hides behind the piece it is about.

**R39. Highlight the last move as a light/dark *pair*, and include the castling rook's two squares.** See §C.1.

**R40. Board geometry: two cells wide per one tall is square.** A chess board as one glyph in a 3-wide cell is always a squashed rectangle and no glyph fixes it. `▀` half-blocks give two vertically stacked pixels per cell. Block elements also have no `wcwidth` ambiguity, while `♞` and Nerd Font codepoints do, and that ambiguity drifts table borders row by row.

**R41. The board column is sized to the picture and the picture is centred in it.** `board_w = image + gutter on both sides`; `board_h = image + PLAYER_ROWS above and below`; the player blocks are set to the **image's** own left and right edges, not the column's. Asymmetric gutters (they were 1 and 3) put the board permanently off-centre with the player lines hanging past one edge and short of the other. Two gutter columns, not one, because at one the picture ends flush against the panel beside it.

**R42. Two rows per player, not one**, and the material row nearest the board on both sides. These are not new rows — they are exactly the rows the old layout reserved and left blank under the board, which is what made the picture look pinned to the top of a column it was not filling. Twelve captured pieces wedged between a rating and a clock was an unreadable smear, and a block whose height depended on the game would move the image out from under itself.

**R43. Material is grouped and counted (`♟6`).** Six pawns drawn as six identical eight-pixel silhouettes is a texture, not a number. Drop the count at one.

**R44. The board gets the width the panels beside it do not need — `RIGHT_COLS` is the working limit, not `BOARD_SHARE`.** On a 16:9 window a square board is width-bound long before it is height-bound, so the share only bites on a terminal wide enough that half of it would overflow the column's height anyway. 60 is where the ladder's bar comes down to half of `GAUGE_MAX`, the first thing over there that visibly loses by being narrower; `MIN_WIDE_COLS`'s 52 is what those panels need to be *correct*, and the 8 columns between the two are worth a third of the board's area. Measured at 160×96: 598 px of board before, 754 px after, right column still 62.

**R45. `WANT_COLS` (the ask) and `MIN_WIDE_COLS` (the floor) must be two different numbers.** Using one for both meant that at 149 columns — one short of the ask, on a window already as wide as the monitor gets — the result was a 46-column text board, a fifty-row empty search panel, and no evaluation panel at all.

**R46. Below the floor, drop panels rather than clip them.** A clipped panel is a lie; a missing one is not.

**R47. Keep the geometry gate.** `tests/verify_layout.py` checks the arithmetic with no terminal, no torch and no network, because the board is drawn by two programs (`Plan`/`rich`, and sixel bytes written from outside `rich` entirely) that have to agree without ever seeing each other's work, and neither can notice a disagreement. Screenshot only to judge how it *looks*, never to check that the numbers add up.

### E.5 Presentation conventions

**R48. Layout order is a claim about what matters.** Board and what the engine thinks about it at the top, because that is the thing being watched. Ratings, training and machine health below, as context. The event tape last, because it answers a question asked after the fact.

**R49. Every trend prints its endpoints.** A shaded trace tells you the shape and nothing else; a trader's chart always prints the high and the low beside it.

**R50. Sub-cell resolution everywhere a quantity moves slowly.** Eighths, not whole cells and not halves — otherwise a slow-moving quantity appears frozen and then jumps. See §B for the measured position counts.

**R51. Rank candidates by the resource committed (visits), not by score.** The search picks its move by visit count, so visit share is what the decision was made on.

**R52. Show state as position and colour, not as text to be parsed.** The SEARCHING/IDLE annunciator is a fixed-width badge; the unit dots are dots; the heartbeat is independent of the data beside it so it distinguishes "quiet" from "dead".

**R53. Explain notation once, in a subtitle or a label — do not leave abbreviations to be guessed.** The header used to read `bul 2513? rd199 4g -456`.

**R54. Only mark the bad moves.** Annotating every accurate move is a column of noise; the eye is looking for where the game went wrong.

**R55. Spend width only on numbers that can still move.** Ratings for time controls the bot no longer plays are dropped from the header entirely — carrying them made the row read as four numbers of equal weight when only one was live. `sumofish-games` is the right place for a number you look up rather than watch.

**R56. Narrow the exception handler around a config read.** `active_controls` used `except Exception` and silently swallowed a `NameError` (`Path` was not imported), so it returned "everything" and looked like it was working. A broad except around a config read turns a bug into wrong behaviour with no signal.

### E.6 Known-open item to carry into the rewrite

CLAUDE.md, "Left unanswered": *what looks wrong about the evaluation panel*. The chart bug behind "one side is winning when it is not" is fixed (it was two games on one curve, R7), but an earlier complaint about that box was never pinned down. The instruction is to ask for a screenshot of just that panel rather than guess. The rewrite should treat `curve_panel` as the one panel with an unresolved design question — note in particular that it plots the **engine-only** curve while `moves_panel` prefers the **Stockfish whole-game** curve, so the two panels on screen at the same time can be drawn from different series over different ply ranges. That asymmetry is undocumented and is the most likely candidate.