# SumoFish dashboard: ground-up Rust rewrite

## Context

`sumofish` today is 5,758 lines of Python (`scripts/watch.py` + `scripts/dash/`) on `rich` plus 333
lines of hand-rolled sixel. Three complaints prompted this:

1. **Startup delay.** ~1.5–3s to first board. Python import is 95ms of it; the rest is a hardcoded
   `time.sleep(0.35)` after a resize request, up to 0.85s of terminal probing, then waiting on the
   lichess network, then a 154ms `rsvg-convert` + ImageMagick round trip per board.
2. **Move delay.** Located: `scripts/dash/fusion.py:180` returns `True` before an 11-line block that
   would push the engine's own chosen move from the `done`+`uci` record. The comment for that block
   sits *below* the return, which is why it reads as live code. Consequence: after SumoFish moves,
   the board shows the pre-move position until the lichess stream catches up (~8.7s), and if the
   opponent replies first the board jumps two plies and our move never gets a last-move highlight.
   Separately: that 8.7s is **documented by lichess as deliberate** — `/api/stream/game/{id}` says
   "Ongoing games are delayed by 3 moves, as to prevent cheat bots", and being the game's own player
   does not waive it. The project's Lab Note says it "cannot be tuned away from this side"; that is
   half right (see §6).
3. **Adding features is hard.** Measured from commit `772fc12`: one new panel is 8 edit sites across
   5 files. Nothing enumerates panels, so nothing can be added by declaration. The vertical budget is
   a hand-solved subtraction chain and **it is already wrong**: `watch.py:289` omits `- results_h` in
   the `curve_h < 7` fallback, so every terminal 50–78 rows tall over-allocates the right column by 8
   rows and silently drops the `machine` panel. `tests/verify_layout.py` is green throughout because
   it gates only the board column.

Decision (Steven, this session): complete redesign in Rust, same features "for the most part" with
liberties taken, modular, gruvbox theme kept, must look good full-screen and small (need not show the
same information at each size), extensible to more than one bot. Time and cost are not constraints.

**A line-by-line port would carry complaint 3 over verbatim.** The design below is built so that the
Python failure modes are unrepresentable rather than discouraged.

### Two research documents this plan compresses

Copy both into `dashboard/docs/` as step 0 — the scratchpad is session-scoped.

- `/tmp/claude-1000/-home-nomad/a9fb2bf8-3826-4fd1-9cc5-d8c40e4ec094/scratchpad/panel-spec.md`
  (78KB) — field-by-field feature spec of all 9 existing panels, every widget, both board renderers,
  every key binding, the theme, and section E: "load-bearing conventions the rewrite must honour".
  This is the requirements document; do not lose a feature without reading it.
- `/tmp/claude-1000/-home-nomad/a9fb2bf8-3826-4fd1-9cc5-d8c40e4ec094/scratchpad/plan-agent.md`
  (52KB) — the long-form design this plan summarises, including the fit-loop pseudocode, the
  falsification methods, and the full risk table.

---

## Stack (verified against docs.rs/crates.io, July 2026, not memory)

```toml
ratatui = "0.30.2"                     # kasuari solver; CellDiffOption::Skip is the key feature
crossterm = { version = "0.29.0", features = ["event-stream"] }
tokio = "1.53.1"                       # rt-multi-thread, macros, time, sync, process, fs, io-util, signal
shakmaty = "0.30.1"                    # FEN/UCI/movegen. GPL-3.0+, same as python-chess today
resvg = { version = "0.47.0", default-features = false }   # brings usvg 0.47 + tiny-skia 0.12
icy_sixel = "0.5.0"                    # sixel fallback: max_colors 64, diffusion 0.0, Wu
reqwest = { version = "0.13.4", features = ["rustls","http2","json","stream","gzip"] }
tokio-util = { version = "0.7.19", features = ["io","codec"] }   # StreamReader + LinesCodec
nvml-wrapper = "0.12.1"                # replaces nvidia-smi. <0.12.1 has wrong field IDs
zbus = { version = "5.18.0", default-features = false, features = ["tokio"] }
zbus_systemd = { version = "0.26100.0", features = ["systemd1","zbus-async-tokio"] }
pgn-reader = "0.29.0"                  # logs/games/*.pgn at zero API cost
serde = "1.0.229"; serde_json = "1.0.151"; toml = "1.1.4"; etcetera = "0.11.0"
indexmap = "2"; jiff = "0.2"           # config-order maps; PGN UTCDate -> local
anyhow = "1.0.104"; thiserror = "2.0.19"
tracing = "0.1.44"; tracing-appender = "0.2.5"   # to a FILE; the TUI owns stdout
[dev-dependencies] insta = "1.48.0"
```

**Hand-rolled, deliberately:** the lichess client (`licheszter` configures no timeouts and drops the
keepalive lines that are our only liveness signal), the file tailer (`linemux` is path-keyed not
inode-keyed and its release is from 2022), the Stockfish UCI driver (`vampirc-uci` untouched since
2022). **Kept as a subprocess:** `journalctl --user -u X -f -o json` — no trustworthy Rust journal
reader exists; add `--since` on attach and capture `__CURSOR` so a restart resumes exactly.

**Not used, against my first instinct:** `ratatui-image`. It marks Konsole+sixel broken for graphics
*clearing* — precisely the failure class behind eight of the project's Lab Notes — its default
features pull libchafa via pkg-config for a half-block renderer we get free from the Pixmap, and its
capability model will not match Konsole's experimental kitty support. Take `CellDiffOption::Skip`
from ratatui itself (that is the actual fix) and read ratatui-image's source for the escape sequences.

### The board picture: kitty graphics, not sixel

**Konsole has supported the kitty graphics protocol since KDE Gear 22.04** (MR !594, merged
2022-02-07, adding sixel + iTerm2 + kitty; code still in current master). Steven's Konsole is
26.04.3. That deletes the entire palette/dither/shimmer problem class: no quantisation, placement by
id so repositioning does not flicker, explicit deletion. Caveats: direct transfer only (`t=d`,
base64 in the escape stream, ~27 chunks of 4096 for a 1144px board), PNG via `QImage` sniffing, and
the author calls it experimental. Sixel stays compiled in as a runtime fallback, not a compile-time
choice.

Pipeline, pure Rust, zero subprocesses:

```
FEN/UCI -> shakmaty -> ported chess.svg.board() -> usvg::Tree::from_str
        -> Pixmap::new(N); resvg::render(Transform::from_scale(N/390.0, ..))
        -> kitty:  pixmap.encode_png() -> base64 chunks
        -> sixel:  pixmap.take_demultiplied() -> icy_sixel::sixel_encode
        -> text:   the same Pixmap at 8/16/24px, half-blocked
```

This deletes `rsvg-convert`, ImageMagick, libsixel (whose local 1.10.5 exits 0 writing nothing),
fonts and fontconfig — python-chess renders coordinate labels as `<path>` outlines, so
`default-features = false` gives a fully static binary. One rasteriser feeds all three outputs, which
also deletes `sprites.py` (1,481 generated lines), `make_sprites.py`, `ink.py` and the two-byte
luminance/alpha two-ink scheme with its "widening cburnett's stroke swallows the king's cross"
workaround. ~1,700 lines gone for a better result.

---

## Workspace layout

`dashboard/` inside `/home/nomad/chess-gpu`, one Cargo workspace, same git history.

```
crates/
  sf-model/    domain types, AppState, Update, apply(), the position Resolver.
               deps: shakmaty, serde, indexmap, jiff.  NO tokio, NO ratatui, NO reqwest.
  sf-theme/    the 24 hexes named + the 6 PICTURE_* colours; Style helpers; contrast fn.
  sf-layout/   Panel trait, PanelSpec, tiers, Compositions, the fit solver, Solved.
  sf-panels/   one file per panel + the registry.  deps: sf-layout, sf-model, sf-theme,
               ratatui.  NOTHING ELSE.  <- this is the enforcement point
  sf-board/    SVG gen -> resvg -> Pixmap -> {PNG, sixel, half-blocks}.  NO tokio, NO ratatui.
  sf-term/     capability probe, kitty/sixel emitters, byte-recording test backend.
  sf-sources/  ALL I/O: tokio tasks, the API governor, tailers, nvml, zbus, journal.
               MUST NOT depend on sf-panels or sf-layout.
  sf-app/      the binary: composition root, config, select! loop, encoder task.
  xtask/       cburnett path extraction, the probe harness, the size-grid gate.
```

**A workspace rather than one crate is the whole answer to "unrepresentable, not discouraged".** In
one crate `mod panels` can always `use crate::sources`, which is exactly what `panels.py:28` does
(`from .sources import GATE`, a sources global read inside a render) and what `panels.py:103` does (a
YAML file read and parse, 12 times a second, inside `header()`). `sf-panels/Cargo.toml` simply has no
tokio, no reqwest, no nvml, no zbus and no `sf-sources`. The build fails, for everyone, every time.
`Update` lives in `sf-model` so panels and sources can both name it without seeing each other.

Two residual holes, both plugged with `clippy.toml` `disallowed-methods` + `#![deny(...)]` in
`sf-panels`: `std::fs`/`std::process`, and **the clock**. `Instant::now()` in a panel is the sneakiest
defect because it makes a render unsnapshottable, and the current code does it (`live_clocks` calls
`time.time()`, `tape_panel` calls `time.strftime`). `Cx` carries `now: Instant` and
`wall: jiff::Timestamp`; that is what makes ticking clocks and staleness ages byte-deterministic in a
snapshot test.

---

## The Panel abstraction — one new file plus one line

```rust
pub trait Panel: Send + Sync + 'static {
    fn spec(&self) -> &'static PanelSpec;
    /// Does this panel have anything to say right now? Pure. New, and load-bearing:
    /// today the dashboard shows "no training run active" in a 6-row box forever.
    fn relevance(&self, st: &AppState, scope: ScopeArg) -> Relevance;
    /// Pure. No I/O, no clock, no mutation. At most once per panel per frame.
    fn render(&self, v: VariantId, area: Rect, buf: &mut Buffer, cx: &Cx<'_>);
    /// DECLARE (not perform) a demand for a raster image. Only the board implements it.
    fn picture(&self, _v: VariantId, _area: Rect, _cx: &Cx<'_>) -> Option<PictureRequest> { None }
    fn on_key(&self, _k: KeyEvent, _cx: &Cx<'_>) -> Option<Action> { None }
}

pub struct PanelSpec {
    pub id: PanelId, pub title: &'static str,
    pub region: RegionId,   // Board | Rail | RailB | Band | Full
    pub scope: Scope,       // Global | PerBot(focused) | PerBot(all)
    pub weight: Weight,     // drop order within a region. Weight::Pinned = never dropped
    pub variants: &'static [Variant],   // most detailed FIRST, mins strictly decreasing
}
pub struct Variant { pub id: VariantId, pub min: Size, pub pref: Size, pub grow: u16 }
```

Panels are zero-sized types. **All mutable view state** (moves-list scroll, which bot has focus,
whether the tape is expanded) lives in `AppState.ui: BTreeMap<PanelId, PanelView>`, which is what lets
`render` take `&self` and stay pure, and lets any panel be snapshot-tested at any scroll offset.

The registry is the only file that changes when a panel is added:

```rust
// sf-panels/src/lib.rs
sf_layout::panels! { board, mind, moves, curve, results, train, machine, lab, api, versions, tape }
```

The macro expands each identifier to `mod <ident>;` plus `&<ident>::PANEL` in
`pub const ALL: &[&dyn Panel]`. Adding a panel = create `sf-panels/src/lab.rs`, add `lab` to that
list. **Reject `inventory`/`linkme` distributed slices** despite getting to zero registration lines:
the order becomes link-order dependent (this design's safety rests on determinism), a distributed
slice destroys the enumeration you can *read*, and the list order is itself the drop-order tie-break —
so that one line is a decision, not ceremony.

One test in `sf-panels` asserts: `PanelId`s unique; every `variants` slice non-empty and strictly
decreasing in `min` on both axes; every `region` appears in ≥1 `Composition`; and
**`Σ min of pinned panels ≤ composition.min` for every Composition** — which is what makes the
solver's over-constrained branch statically unreachable. That matters because ratatui's book
explicitly documents kasuari as non-deterministic when constraints cannot all be satisfied.

---

## Responsive tiers, and how the 8-row bug class dies

Four tiers. **They only choose a Composition; they never compute a size.** The `_wide_layout` /
`_narrow_layout` mistake was not two branches, it was that each did its own arithmetic with no
contract between them, so `draw()` needs `getattr(plan, "results_h", 0)` to paper over it.

| Tier | Trigger | Composition | Board |
|---|---|---|---|
| `Micro` | < 60 cols or < 20 rows | one region, one panel, tab-cycled | text or nothing |
| `Compact` | ≥ 60 × 20 | `Column[Board, Rail]` | text or a small picture |
| `Standard` | ≥ 100 × 34 | `Row[Board, Rail]` — today's layout | picture sized to the region |
| `Wide` | ≥ 176 × 44 | `Row[Board, Rail, RailB]` | picture; second rail takes width the board over-eats |

`Wide` is the answer to "looks good full-screen": past ~1150px a square board is height-bound on a
44-row window, so the current single 60-column rail leaves the board eating width it cannot use. The
second rail is where `lab`, `matches`, `versions` and `api` go. `--tier` forces one, for screenshots.

Compositions are hand-written trees, max depth 3, all four in one file, each snapshot-tested.
**There is no general 2-D packer and there will not be one** — a packer's output is not predictable
enough to memorise, and a dashboard you look at daily should put things in the same place every time.

**The fit loop** (`sf-layout`, the only code permitted to compute a size):

```
1. Drop panels whose relevance == Hidden.
2. Pick the first (most detailed) variant whose min fits the CROSS axis; else drop the panel.
   Cross-axis first: a 52-column rail cannot host a 60-column variant, and finding that out
   after solving the main axis is how you get a clipped panel.
3. loop { need = candidates.iter().map(|c| c.min.main()).sum();     // the SAME Vec
          if need <= extent { break }
          demote-or-drop the lowest (weight, relevance, registry_order) non-Pinned panel }
4. The set provably fits. ONLY NOW does kasuari see it: Layout::vertical(Min(min)...).
   Second pass distributes slack by `grow` among panels below `pref`.
5. debug_assert: rects.len() == candidates.len(); Σ heights == area.height; every rect >= its min.
```

The `watch.py:289` defect becomes unrepresentable for a nameable reason: **step 3 sums over the same
collection it is about to lay out.** You cannot forget a term in a sum over a `Vec` the way you can
forget `- self.results_h` in a hand-written expression.

`Solved` also carries `dropped: Vec<(PanelId, DropReason)>` with
`Hidden | CrossAxisTooSmall{need,have} | Demoted | NoRoom{need,have}`, bound to a `?` debug overlay.
"Where did my machine panel go" was unanswerable for however long that bug has been live; it should
cost one keypress.

**Picture geometry comes out of the same solve, once:**
`PictureGeom { cells: Rect, px: u32, cell: CellSize, protocol }` with
`px = snap26(min(cells.w * cell.w, cells.h * cell.h))`. One source of truth replacing today's three
(`Plan.image_*`, `board_panel`'s eight parameters, and `verify_layout.py` recomputing it). The
multiple-of-26 rule is load-bearing: `chess.svg.board` is a 390-unit square, so square edges land on
exact pixels only at multiples of 26; anywhere else rsvg blends the two square colours and the board
grows a faint grid it does not have (8 stray pixels per scanline at 1152, zero at 1144).

Encoded as `Composition` validation, not a comment (from the Lab Notes): **the Board region never
merges borders with a neighbour and never hosts an animated cell.**

---

## Multi-bot data model

```rust
pub struct AppState {
    pub bots: IndexMap<BotId, BotState>,   // config order, stable on screen
    pub focus: BotId,
    pub machine: MachineState,   // GPUs (nvml), units (zbus), journal counters
    pub train: TrainState,       // one GPU, so global
    pub lab: LabState,           // runs/lab/{state.json,log.jsonl,report.md}   (new)
    pub matches: MatchState,     // runs/matches/*: elo, interval, LOS, LLR      (new)
    pub api: ApiState,           // per-endpoint buckets, 429s, effective intervals (new)
    pub ui: UiState, pub tape: Tape,
    pub engines: HashMap<Pid, EngineProc>,   // ALL engine processes, attributed or not
}
```
`BotState` holds `cfg`, `account`, `playing`, `games: IndexMap<GameId, GameState>`, `record`,
`results`, `version`. `GameState` holds `position: Resolved` (§6, the only position anyone reads),
`clocks`, `moves` (authority tagged per ply), `curve` and `sf_curve` **always in White's frame**,
`grades`, `search`, `pid`.

`Tracked<T>` replaces `Field` and closes its loophole: `State.get()` currently hands you a value with
the staleness discarded. The new API is `fn get(&self) -> Option<(&T, Track)>` with **no** accessor
returning the bare value, so the docstring's claim becomes true rather than aspirational.

**`pid` is the partition key, and it is free.** Every telemetry record carries it unconditionally
(`chessgpu/telemetry.py:107`) and **nothing in the repo reads it.** Attribution: `game` → `(BotId,
GameId)` via each bot's `playing` list, with a sticky pid cache so `think` records land right before
and after `game` appears. Unattributed pids show as "N detached engines" — visible, not guessed at.

**Delete the entire 4-layer demultiplexer.** `_bridge`, `_orphans`, `reanchor`, the `stm` colour
disambiguation and the placement-FEN `history` set exist solely because `game` was absent from
records. It is present now (`patches/0003`). Keep one narrow fallback: a record with no `game` whose
FEN matches exactly one bot's exactly one game is attributed, otherwise `Unknown` and it contributes
to nothing. **Test that an unattributable record affects zero on-screen values** — the opposite of
today, where such a record is aggressively adopted. ~200 lines of the subtlest code in the project
deletes because a key was added upstream.

Trap: `shakmaty::Board` is `Eq + Hash` (bitboard compare, excellent) but **placement only** — no
turn, castling or ep. That is the same comparison `fusion.py` makes with `board_fen()` and it is
exactly why two games in the same opening confused it. Use `Board` as a cheap pre-filter; use
`Chess`/`Setup` when you mean "the same position".

### Config: `~/.config/sumofish/dash.toml`

Per-bot is only ever a name, a credential, a file path or a unit: `id`, `user` (**never a constant in
code** — it is hardcoded in 6 places today, including `panels.py:351` which `--user` cannot reach and
which decides which way the eval gauge reads), `token_file`/`token_var`, `unit`, `telemetry`,
`pgn_dir`, `versions`. Global is **everything rate-limited**, plus `[machine] units`, `[lab]`,
`[train]`, `[dash] fps/board/tier`.

### The API governor, and the hard scaling wall

One `sf-sources::api::Governor` task owns every outbound lichess request; no other task holds a
`reqwest::Client`. Sources send `ApiRequest` and await a oneshot. Four properties become structural:

1. **One in-flight request, paced 250ms** — the only concrete guidance lichess publishes ("Only make
   one request at a time"), with no way for a source to bypass it.
2. **Buckets keyed by endpoint, not by bot.** This is the correction to the obvious design: the
   limits are **per-IP-per-endpoint independent of credential** (measured in this repo:
   `/api/user/{name}` 429s for both authenticated *and* anonymous requests while `/api/account`
   stayed fine). So four bots polling `/api/account/playing` every 2s is 120 req/min against one
   bucket, not 4×30. Sources therefore **do not choose their own cadence** — they register a
   `PollSpec { endpoint, key, want_interval, floor_interval, priority }` and the governor issues
   permits round-robin, so `playing`'s effective interval degrades to `n_bots × base`
   automatically and *visibly*. That degradation is the honest answer to "add more bots as it grows"
   and it belongs on screen.
3. **Forbidden endpoints are unnameable.** `enum Endpoint` has no variant for `/api/stream/event` —
   one per token, and lichess-bot owns it for every token.
4. **429 = backoff only, never retry.** Every retry against a 429 renews the penalty; that cost 40
   minutes of a bot unable to create a single challenge.

**The wall, stated plainly:** max 8 concurrent `/api/stream/*` per IP, and lichess-bot at
`concurrency: 2` already holds 1 event stream + up to 2 game streams **per bot**. Two bots is 6 of 8.
Three bots and lichess-bot alone wants 9 — the bots break each other before the dashboard looks in.
Conclusion that shapes the whole design: **the dashboard streams at most the focused game and derives
everything else from local telemetry plus `/api/account/playing`.** Per §6 the local telemetry is
strictly fresher anyway. Also worth knowing before a second bot exists: 2N searching engines share
one GPU, and halving nodes-per-move costs real rating — **a second bot makes the first one weaker.**

---

## Position resolution, and the move-delay fix

**One component, `sf-model::position::Resolver`, pure, one per game.** Today the decision lives in
`watch.py::_live_position` *and* `panels.board_panel`, with a docstring asking a human to keep them in
sync — the classic setup for the eval gauge and the picture disagreeing about what is on the board.

```rust
pub enum Obs {
    EngineSearching { pid, game, pos, ply, at },   // telemetry: position AFTER opponent moved
    EngineChose     { pid, game, from, mv, at },   // telemetry done+uci: OUR move. THE FIX.
    Stream          { game, pos, last, clocks, ply, at },   // authoritative, +3 moves delayed
    Playing         { game, pos, last, at },       // /api/account/playing, no extra request
    Terminal        { game, status, winner },
}
pub struct Resolved { pos, last, ply, source, as_of, lag: Lag }
```

Rules, total and ranked:

1. **Highest ply wins; ties break by source rank** `EngineChose > EngineSearching > Playing > Stream`.
   Every `Obs` carries `game`, so cross-game contamination is impossible by construction.
2. **`EngineChose` synthesises the post-our-move position immediately** by applying `uci` to the
   record's own `fen`. This is the dead block at `fusion.py:182-191` restored as a first-class
   observation, and it fixes both halves of the complaint: no ~9s pre-move position, and our own move
   keeps its highlight even when the opponent replies before the stream catches up.
3. **Ply never regresses within a game id** — kills the reconnect-replay flicker with no
   `published_ply` bookkeeping.
4. **Disagreement is surfaced, not arbitrated away.** `sync: engine +2 ply, feed 8.7s` on screen.
   Otherwise the next person to notice the lag re-measures it and re-attributes it to this code, which
   has already happened once and cost two rounds of optimising a 400ms render under an 8,700ms delay.
5. **Move list authoritative from the stream when present, synthesised from the engine chain
   otherwise, each ply tagged with which.** Synthesised plies miss exactly one case — a game ending on
   the opponent's move, which we never search — and tagging makes that visible rather than looking
   like a lost move.

**On the delay: do not build the foundation on `/api/bot/game/stream/{gameId}`.**
`lichess-bot/lib/lichess.py:23` already opens that exact stream, for that exact game, with that exact
token. The downside is a live rated game. Foundation is telemetry-primary, which needs nothing: our
moves from `done`+`uci`, theirs from the FEN of the next search, which starts within milliseconds of
lichess-bot receiving it. Clocks in preference order: (i) `playing`'s `secondsLeft` at governor
cadence plus a local tick and resync — zero extra requests since we poll it anyway; (ii) lichess-bot's
own journal, local and free; (iii) the delayed public stream, **labelled `+3 moves` permanently**.
The undelayed bot stream is an opt-in enhancement behind `[api] bot_stream = false`, enabled only if
F4 (below) passes on a casual game.

---

## What to add: data that exists and is not shown

Ranked. Everything here is already on disk or one library call away.

| # | What | Where | Why |
|---|---|---|---|
| 1 | **EGTB move provenance.** 349 of 3,265 moves today (10.7%) came from lichess's tablebase, not the engine, with `wdl`/`dtz`/`dtm`. They produce **no `move` record at all**, so the mind panel coasts on a stale position while the board advances | bot journal | Watching the search panel through an endgame is watching a lie |
| 2 | **Watchdog restarts.** The bot was SIGKILLed and restarted twice in 30 minutes today. Because `watchdog.py` uses `systemctl kill` + `restart`, `NRestarts` stays 0, so `machine_panel` suppresses the count and **shows a green dot** | `~/.local/state/chess-gpu/watchdog.json`, journal | The single most misleading pixel on screen |
| 3 | **The lab.** 9/15 jobs, `train-136m` running, 5 queued, deadlines, the `facts{}` that gate future jobs. 9.3h of autonomous experiments, zero dashboard presence | `runs/lab/state.json`, `log.jsonl` | The largest thing happening is invisible. `lab.render()` already produces the text |
| 4 | **Which checkpoint is playing** — `boot` carries `policy_step`, `value_step`, `params`, `bins`, `batch` and the payload is discarded at `sources.py:542` | `engine.jsonl` | Promotion swaps `runs/value.pt` under a running bot |
| 5 | **`val_loss`**, the metric `best.pt` is actually selected on and which PHILOSOPHY calls the honest one. The panel plots the noisy `puzzle_acc` instead | `runs/*/log.jsonl` | Already collected, just dropped |
| 6 | **Correct training ETA.** `total = 300_000` and `/1024` are hardcoded; this run is 400k steps at batch 256, so it shows 13.5% and **eta 76.5h where the arithmetic gives ~26.5h** | `runs/*/config.json` | "When is my 40-hour run done" is the main reason to look |
| 7 | **`unique/s`, not `nps`.** PHILOSOPHY states this as an imperative; the mind panel shows `nps 1,806` beside `sims 38,720`. The number underneath is honest, the label is not | `panels.py:554` | Rename, and add `nodes/sims` as the dedup ratio |
| 8 | **Live match Elo, ±interval, LOS, LLR, SPRT state** — a 6–20h verdict arriving one game at a time. `lab.match_verdict()` is already the reader | `runs/matches/*/games.jsonl` | A number without an interval is not a result |
| 9 | **Provenance check** `Σ game.seconds ≤ job.seconds`. **All four exchange-rate rungs still violate it** (e.g. 20,733s of play credited 1,070s), and three have `code: null` | computed | PHILOSOPHY calls this the cheapest total check in the repo |
| 10 | **Deployment markers on the rating curve** — `deployed{engine, policy, value}` hashes, visibly changing across the log; `_rating_history` reads only `rating` | `logs/rating.jsonl` | The literal stated purpose of that file |
| 11 | **Per-process VRAM** (11,096 MiB trainer vs 626 MiB engine, 82% of 16GB), plus temp, power vs limit, SM/mem clocks, throttle reasons, fan | nvml | `machine_panel`'s own docstring asks "is one starving the other"; inferring it from one aggregate is the exact mistake that cost a session |
| 12 | **Correct unit list.** Polls 2 names, one of which (`chess-gpu-train.service`) is `not-found` and renders a permanent red dot; `chess-gpu-lab` — the thing actually working — is absent. Three timers unpolled. **`train_watchdog` watches the dead unit, so the 40h run is unsupervised** | `systemd/`, zbus | A permanently-red dot for a deleted unit trains you to ignore the row |
| 13 | **Budget overruns** — 9 of 324 moves overran by >0.5s, max +0.81s. `elapsed` vs `budget` is on every record | `engine.jsonl` | Flagging is the one unrecoverable failure |
| 14 | **Per-game Elo delta, opponent rating, ECO/opening** — `games.py` already extracts all 12 fields; `Results` keeps 6 | `logs/games/*.pgn` | "Lost 7 points to a 1739" is what a person wants from a results row |
| 15 | **Stockfish game verdict** — accuracy, blunder count, ACPL, ours vs theirs. `Grader` computes per-ply `{loss, grade, cp}` and only the `?!`/`??` marks reach the screen | already in memory | The only instrument that answers "is it playing well" independently of the rating |
| 16 | **Best-move instability** — the engine changes its mind in 32% of searches, up to 14 times, over ~87 think frames per move | `engine.jsonl` | The most watchable fact the feed contains, written 6×/s and thrown away |
| 17 | **The second concurrent game.** `concurrency: 2`; the demuxer spends 80 lines discarding it | `state["playing"]` | Half the bot's games happen off-screen |
| 18 | `VERSIONS.jsonl` beyond `ts` (title, layman, sha, `checkpoint_sha`, step, width, layers); `runs/lab/forward-bench.json`; `research/results.jsonl` verdicts; `Telemetry.dropped`; log-rotation notice; timer `NextElapse` | various | Cheap, one file read each |

**Engine-side additions (separate workstream, touches `chessgpu/`):** `reused` — visits inherited by
`_reroot`, returned by `MCTS.report()` and dropped by `snapshot()`, so tree reuse has zero
observability; the now-varying `c_puct_at(visits)`; batch collisions and terminal hits, which explain
the `nodes/sims` gap. Each is one key in a dict already being built. **Nothing may edit `chessgpu/`
while a match or the bot is running** — both import from the working tree and spawn per game, so an
edit lands mid-match. Use a worktree, or wait for the lab to be quiet.

---

## Build order

Each milestone leaves a runnable program with a verification that needs no chess game happening.
Cycle counts are rough; VERIFY dominates M0 and M2, GEN dominates M6.

**M0 — Falsification day. Commit no architecture until this reports.** An `xtask probe` binary that
answers and checks in `dashboard/docs/probe-results.md`. This is the cheapest milestone and two of
the seven can invalidate a settled decision.

- **F1** Does Konsole 26.04.3 accept kitty `a=T,f=100` chunked base64 PNG placed by cursor move, with
  `q=2` to suppress replies? At what chunk size? Does `a=d,d=i` delete work, or is `ESC[2J` still the
  only eraser? *Method:* write bytes to `/dev/tty` from a program in an **already correctly sized
  window** (never resize another process's pty — Lab Note), then one screenshot.
- **F2** Does it accept `f=32` raw RGBA? If yes the PNG encode leaves the hot path, at ~4x the bytes.
- **F3** Does the image survive a full ratatui redraw with `CellDiffOption::Skip` on the region?
  Verify twice: visually, and headlessly with a `Backend` wrapper that records every byte written and
  asserts zero writes inside the picture rect.
- **F4** Does a second `/api/bot/game/stream/{gameId}` with the bot's token coexist with lichess-bot's?
  *Method:* **casual game only, never rated**; tail the bot's journal for stream drops during the
  window; abort on any drop attributable to us, and write the Lab Note. `patches/0001` already makes
  the bot re-check `game_is_active` on a stream drop, which bounds the blast radius.
- **F5** Is `playing`'s `fen`/`lastMove` 3-move delayed? *Method:* 20 plies against telemetry
  timestamps. Undelayed looks like ≤ the poll interval; delayed looks like ~8.7s.
- **F6** Does usvg/resvg match rsvg for cburnett? 8 positions, pixel diff, **plus the seam assertion**
  (zero non-square-colour pixels on a scanline at 1144, nonzero at 1152).
- **F7** nvml per-process VRAM readable under the systemd sandbox; zbus reaching the **user** bus.

**M1 — Skeleton** (~10 cycles). Workspace, config, `select!` loop, `watch` shutdown, tracing to a
file, `sf-term` probe, tiers, the fit solver, and exactly one panel (`header`). Panic and signal hooks
that restore the terminal first, or a crash in a background task leaves raw mode plus graphics
garbage. *Verify:* the size-grid gate passes; `--tier standard --size 150x44 --snapshot` prints a
frame to stdout; renders, resizes and exits cleanly on `q` and Ctrl-C.

**M2 — The picture** (~10 cycles, VERIFY-dominated). `sf-board`: the ported cburnett generator, resvg,
PNG + sixel + half-block emitters, and the encoder task (newest-wins, drop stale) mirroring
`sixel.Renderer`'s proven contract. *Verify:* `board --fen … --px 1144 --png /tmp/a.png` with golden
pixel hashes; the seam test as an assertion; "never emit an image whose key's `px` differs from the
current layout's" as a unit test on the emitter.

**M3 — Telemetry and the resolver, offline first** (~8 cycles). Replay `logs/engine.jsonl.1` — 17MB
of real two-game-interleaved data already on disk — through the resolver **in a test before any live
tailing**. Then the inode-keyed tailer, then live. `replay <file> --speed 60` becomes a first-class
subcommand and **replaces `seed_demo` entirely**: a fixture built from real recorded telemetry is
strictly better than a hand-written one, and it makes `sumofish demo` a regression test you can watch.
serde_json parses 17MB in tens of milliseconds, so the 256KB backfill window becomes "index the whole
tail by pid" and the "attaching mid-game shows an empty panel" failure disappears rather than being
mitigated.

**M4 — lichess** (~10 cycles). Governor, `/api/account`, `/api/account/playing`, the game stream
(delayed or bot, per F4), finished-game export, PGN results via `pgn-reader`. *Verify:* governor test
under `tokio::time::pause()` — six synthetic bots, virtual clock, assert no endpoint bucket is ever
exceeded and no 429 is ever retried. Deterministic, instant, no network.

**M5 — Machine** (~6 cycles). nvml, zbus systemd, journal follower with cursor resume. *Verify:* diff
against `nvidia-smi --query-compute-apps` and `systemctl --user show`.

**M6 — The remaining panels, one commit each** (~15 cycles). `mind`, `moves`, `curve`, `results`,
`train`, `machine`, `tape`, plus the new `lab`, `matches`, `versions`, `api`. Each commit is one new
file plus one identifier plus generated snapshots. **The diffs are the proof the Panel abstraction
delivered** — if any commit touches a fifth file, the design failed and should be fixed then.

**M7 — Multi-bot** (~8 cycles). Second `[[bot]]`, focus switching (`Tab`, `1`-`9`), `PerBot(all)`
panels, per-bot header rows. *Verify:* run with a fixture second bot and assert total API traffic is
unchanged; run with two real bots and assert the governor's effective `playing` interval doubled and
said so on screen.

**M8 — Stockfish grader, the `Wide` second rail, cutover** (~8 cycles). The UCI driver is last because
it is the only source whose absence costs nothing. Two tasks (one owns stdin, one owns stdout) behind
one actor — writing to a child's stdin from the same future that reads its stdout is a documented
deadlock. `.kill_on_drop(true)`, or a Stockfish leaks per dashboard restart.

---

## Verification

**Layout, headlessly, at every size.** One test over the full cross product, cols 40..=320 × rows
12..=120 (~29k solves, each pure and microseconds):

- Σ of every region's children == the region's extent. *(Catches the +8 overflow directly.)*
- Every rect ≥ its variant's `min` on both axes. *(Catches clipping.)*
- No two rects intersect, over all pairs. *(Catches the picture-under-the-tape class.)*
- `picture.cells ⊆ board_region` and `picture.px ≡ 0 mod 26`.
- **Presence monotonicity:** if panel P is placed at `(c, r)` it is placed at every `(c' ≥ c,
  r' ≥ r)`. This is the flagship test. It fails at rows 50–78 today, and non-monotone presence is
  *always* a bug and otherwise invisible — `machine` is present at 49 rows, absent at 50–78, present
  at 79.
- Tier is monotone in both axes.

**Panels in isolation.** `insta` + `TestBackend`, table-driven over `(panel, variant, size)` **from
the registry itself**, so a new panel gets its snapshots by being registered, with no new test file.
Fixed `now` from `Cx` makes clocks and staleness ages deterministic.

**Colour, which `insta` cannot see** (snapshots are text-only) — three separate assertions:

1. **Palette conformance.** Walk the rendered `Buffer`, collect every distinct fg/bg colour,
   set-difference against the 24 theme hexes plus the 6 declared `PICTURE_*` colours. Any stray colour
   fails. Stronger than anything the Python has.
2. **Contrast floors.** WCAG contrast between gauge fill and ground, ≥ a floor. This encodes the
   `#32302f`-on-`#282828`-is-1.12:1 Lab Note as a test so the class cannot recur.
3. **Colour-map snapshots.** Render the buffer again as a grid of theme-colour initials and `insta`
   that as text. This gets colour into `insta` after all, and catches "the whole panel went dim".

Note the theme is **27 names over 24 distinct hexes** (`LIVE`/`COAST`/`LOST` alias `GOOD`/`ACCENT`/
`BAD`; `CHECK_RING` aliases `BAD`; `EVAL_MID` aliases `BOARD_DARK`; `PIECE_B_EDGE` aliases `FG`), and
**the six colours the picture actually uses live in `sixel.py`, not the theme** — they are lichess's
board colours, deliberately not gruvbox. Name them `PICTURE_*` and exempt them, or the first person to
enforce "keep to the theme" will gruvbox the board and destroy the one property that makes a real
image worth having.

**The position resolver.** Golden replay from real recorded telemetry:

- Snapshot the per-frame resolved position sequence against a virtual clock.
- **Assert our own move appears in the same frame as its `done` record. Write this test before the
  fix** — it is the regression test for `fusion.py:180`.
- Assert no ply regression, and zero records from game B touching game A's curve, grades or board.
- **Property test: the resolver's final state is invariant to arrival order** of observations within a
  10s window. Shuffle, re-run, compare. This is precisely the property the old demuxer lacked and why
  interleaving produced "the graph says one side is winning when it is not".

**The picture pipeline.** Golden PNG hash per `(fen, px, flip, lastmove)`; the seam assertion at
multiples of 26 and its negative at 1152; a round-trip test that parses our own kitty escape sequence
back out and checks chunking and terminators; and the byte-recording `Backend` proving zero writes
land inside the picture rect over 100 frames of neighbouring cells changing. Per the Lab Notes,
screenshots judge only how it *looks*, never whether the numbers add up.

**Sources.** Each parses bytes → `Vec<Update>` as a free function, tested against recorded fixtures.
Do not build a mock layer for things whose failure modes are network conditions. Plus an `apply()`
fuzz: random `Update` sequences must never panic and never lower a game's ply.

---

## Migration and coexistence

- Binary at `dashboard/target/release/sumofish-dash`, invoked through `bin/sumofish-dash` following
  the existing wrapper pattern so paths stay out of the systemd units.
- **`~/bin/sumofish` stays the front door.** During the build, add cases without touching defaults:
  `sumofish` → Python (unchanged; it is the working instrument for a bot playing right now),
  `sumofish next` → Rust. At cutover flip: `sumofish` → Rust, `sumofish py` → Python for one cycle,
  then delete the `py` case.
- **The Python version stays available throughout.** Losing observability on a live rated bot for a
  week is not a trade worth making. But **two dashboards double the load on the same per-IP endpoint
  buckets**, which is what already put the bot into a 40-minute no-challenge stall once. So: the Rust
  binary defaults to `--offline` until M4 (telemetry + nvml + systemd + lab only, zero lichess), and
  from M4 both programs take a pidfile lock at `$XDG_RUNTIME_DIR/sumofish-dash.lock`; the Rust one
  refuses to open lichess sources while the Python one holds it, and says so on screen.
- **At the end, delete `scripts/watch.py`, `scripts/dash/` and `tests/verify_layout.py` in one
  commit.** That last one matters: a layout gate for a deleted program is a test of nothing that still
  passes green, and PHILOSOPHY's rule is that retracted things get deleted at the point of use rather
  than annotated. `sumofish mind` becomes `sumofish-dash replay --follow`. `sumofish-lab` and
  `sumofish-games` stay Python — separate programs, out of scope, though the new `lab` and `matches`
  panels make the first much less necessary.
- `CLAUDE.md`'s Layout section grows one entry; its "The dashboard, and why it is shaped that way"
  section gets rewritten against the new design, keeping every Lab Note whose reason still applies.

## Riskiest assumptions

| # | Assumption | Falsify | If it fails |
|---|---|---|---|
| R1 | Konsole accepts kitty `a=T,f=100` direct-transfer PNG | F1 | sixel primary via `icy_sixel` (`max_colors:64, diffusion:0.0`) — a path already proven here |
| R2 | The image survives ratatui redraws under `CellDiffOption::Skip` | F3, twice | hard-reserve the region, re-emit on key change only (today's behaviour, kept) |
| R3 | Konsole's `a=d` delete works | F1 | `ESC[2J` on geometry change, exactly as today |
| R4 | A second bot game stream coexists with lichess-bot's | F4, casual only, abort on any drop | telemetry-primary + `playing` clocks; label the public stream `+3 moves`; write the Lab Note |
| R5 | `playing`'s fen/lastMove is not 3-move delayed | F5 | clocks from lichess-bot's journal (local, free), else the labelled stream |
| R6 | resvg matches rsvg for cburnett | F6, pixel diff <0.1% | keep `rsvg-convert` for the picture only, lose 54ms |
| R7 | The hand-ported `chess.svg.board()` is faithful | Extract cburnett paths ONCE via `xtask codegen` from installed python-chess, check the generated file in, then **diff our SVG string against python-chess's own output over 200 random positions** — CLAUDE.md's "exec the original and diff against it" rule | the diff names the glyph |
| R8 | ratatui's `Layout` returns rects matching the pre-filtered mins (no Fill/Min rounding surprise) | the 29k-size grid gate | do slack distribution by hand and let kasuari see only `Length` |
| R9 | nvml per-process VRAM readable; zbus reaches the user bus | F7 | keep the `nvidia-smi`/`systemctl show` subprocesses for those fields |
| R10 | Cell size is stable after startup | change the font mid-run | re-probe on every resize; cheap, probably just do it unconditionally |
| R11 | `done`+`uci` always arrives before the opponent's reply is searched | replay test, count inversions | ply-monotonicity already handles it; the highlight is lost for that ply, nothing else |
| R12 | `pid` alone partitions — no engine process reused across games | replay test asserting each pid maps to ≤1 `game` at a time | key on `(pid, game)`; a pid whose `game` changed is a new partition |

R1 and R4 are the only two that would change the architecture, which is why M0 exists as its own day.

## Out of scope, but found and live right now

Not part of this rewrite; flagged because they are true today and two of them argue for panels 2 and
12 above.

- The bot was **SIGKILLed and restarted twice in the last 30 minutes** by its own watchdog for
  120s-stuck games, and nothing on screen said so.
- **`chess-gpu-train.service` is `not-found`**, so `train_watchdog` logs "nothing to watch" every 5
  minutes and the 40-hour 136M run is unsupervised. Its state file is frozen at a previous run.
- The train panel's ETA is wrong by ~3x (hardcoded 300k steps and batch 1024 against a 400k/256 run).
- **All four exchange-rate rungs still violate `Σ game.seconds ≤ job.seconds`** in the files right
  now, three with `code: null`, and `lab.py status` plus `report.md` publish those Elo numbers
  unmarked. `runs/lab/report.md` is also stale against `state.json`.
