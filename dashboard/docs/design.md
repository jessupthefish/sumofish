I read `CLAUDE.md` (all of it, including the 429 lines of Lab Notes), `PHILOSOPHY.md`, and all 5,758 lines of the current dashboard plus `sumofish/telemetry.py`, `search_engine.py`, `tests/verify_layout.py`, `config/lichess-bot.yml`, `systemd/`, `~/bin/sumofish` and `lichess-bot/lib/lichess.py`. Three of your claims I verified directly rather than taking on trust — results in §0. Everything below is the design.

---

# 0. Verified before designing

**The 50-78 row bug is exactly that range, and the overflow is exactly 8 rows.** I reproduced `_wide_layout`'s arithmetic over rows 30..130. Overflow is continuous from 50 to 78 inclusive, no gaps, always +8 — which is `results_h`. The cause is `watch.py:289`: `self.moves_h = h - self.mind_h` in the `curve_h < 7` fallback, which omits `- self.results_h`. `rich` then silently discards the tail of the column, so `machine` disappears. Also worth knowing: presence is **non-monotone** — `machine` is present at 49 rows, absent at 50-78, present at 79. That non-monotonicity is a cheap property test and it is the one I would build the tier system's test suite around.

**`fusion.py:180` is dead code, confirmed.** `return True` at 180; lines 182-191 (the `done`+`uci` push) are unreachable. The comment for that block sits *below* the return, which is why it reads as live code.

**The theme is 27 names over 24 distinct hex values** — not 25. `LIVE/COAST/LOST` alias `GOOD/ACCENT/BAD`; `CHECK_RING` aliases `BAD`; `EVAL_MID` aliases `BOARD_DARK`; `PIECE_B_EDGE` aliases `FG`. Separately, the six colours the *picture* actually uses (`#f0d9b5`, `#b58863`, `#cdd26a`, `#aaa23a`, `#2b2724`, `#e5e0d5`) live in `sixel.py`, **not** in the theme, and are deliberately lichess's colours rather than gruvbox. Any "conform to the theme" test must exempt the picture, and the new theme module should name those six explicitly as `PICTURE_*` so a future tidy-up doesn't gruvbox the board and break the one property that makes the picture worth having.

**`lichess-bot/lib/lichess.py:23` uses `/api/bot/game/stream/{}`** with the same token, for every live game, plus one `/api/stream/event` per token. So §5's undelayed-stream plan is a direct collision, not a hypothetical one. Mitigating fact: `patches/0001` already makes the bot re-check `game_is_active` on any stream drop instead of abandoning the game, which bounds the blast radius enough to make the experiment safe to run once, on a casual game.

---

# 1. Crate and module layout

**A workspace, not a single crate.** This is the whole answer to "make the Python failures unrepresentable rather than discouraged." In one crate, `mod panels` can always `use crate::sources` — which is exactly what `panels.py:28` does (`from .sources import GATE`, a sources global read from inside a panel) and what `panels.py:103` does (`active_controls()` reads `config/lichess-bot.yml` from inside a render function). Neither is preventable by convention. In a workspace, `sf-panels/Cargo.toml` simply has no `tokio`, no `reqwest`, no `nvml`, no `zbus`, and no dependency on `sf-sources`. The build fails, every time, for everyone.

```
/home/nomad/dev/active/sumofish/dashboard/          (Cargo workspace root)
  Cargo.toml            workspace + [workspace.dependencies] pinning every version once
  clippy.toml           disallowed-methods: fs::*, process::Command, SystemTime::now,
                        Instant::now, thread::sleep  (see note below)
  crates/
    sf-model/     domain types + AppState + Update + apply() + the position resolver.
                  deps: shakmaty, serde, indexmap, jiff. NO tokio, NO ratatui, NO reqwest.
    sf-theme/     the 24 colours, named; Style helpers; contrast fn.
                  deps: ratatui (for Color/Style only).
    sf-layout/    Panel trait, PanelSpec, tiers, compositions, the fit solver, Solved.
                  deps: ratatui, sf-model, sf-theme.
    sf-panels/    one file per panel + the registry. deps: sf-layout, sf-model, sf-theme,
                  ratatui. NOTHING ELSE. This is the enforcement point.
    sf-board/     SVG generation (ported cburnett) -> usvg/resvg -> tiny-skia Pixmap ->
                  {PNG for kitty, icy_sixel, half-blocks for text}. deps: shakmaty,
                  usvg, resvg, tiny-skia, icy_sixel, png. NO tokio, NO ratatui.
    sf-term/      terminal capability probe + the kitty/sixel emitters + the byte-recording
                  test backend. deps: crossterm, ratatui.
    sf-sources/   ALL I/O. tokio tasks, the API governor, tailers, nvml, zbus, journal.
                  deps: sf-model (to build Update), tokio, reqwest, tokio-util, nvml,
                  zbus, pgn-reader. MUST NOT depend on sf-panels or sf-layout.
    sf-app/       the binary. composition root, config, the select! loop, the encoder task.
                  deps: everything.
    xtask/        codegen (cburnett path extraction), the probe harness, the size-grid gate.
```

Dependency direction is a DAG with `sf-model` at the bottom and `sf-app` at the top; **`sf-panels` and `sf-sources` are siblings that cannot see each other.** `Update` lives in `sf-model` so both can name it without either depending on the other. `cargo tree` is the enforcement, and I would add a CI check that greps `sf-panels/Cargo.toml` for the forbidden names so the violation is caught at review time too, not just at the moment someone tries.

Two residual holes and their plugs, because `std` is always in scope:

- **`std::fs` / `std::process`** — plugged with `clippy.toml`'s `disallowed-methods` plus `#![deny(clippy::disallowed_methods)]` in `sf-panels/src/lib.rs`. Not as strong as the dependency graph, but it is a build failure, and there is nothing else in `std` a panel can use to do I/O that matters.
- **the clock** — `Instant::now()` inside a panel is the sneakiest defect, because it makes a render impure and therefore unsnapshottable, and the current code does exactly this (`panels.live_clocks` calls `time.time()`, `tape_panel` calls `time.strftime`). Plug: `Cx` carries `now: Instant` and `wall: jiff::Timestamp`, and `Instant::now`/`SystemTime::now` are in the disallowed list. This is what makes ticking clocks and staleness ages snapshot-testable — you pass a fixed `now` and get a byte-identical frame.

One deliberate non-decision: I am **not** giving panels typed lenses onto sub-slices of `AppState`. It is tempting (you could prove which fields a panel reads and skip re-renders) but the failure being fixed was I/O and mutation, not over-broad reads, and at 12fps the skip is worthless. `render` takes `&AppState` behind `Cx`. Say so once and move on.

---

# 2. The Panel abstraction

```rust
// sf-layout/src/panel.rs

pub trait Panel: Send + Sync + 'static {
    /// Stable identity. Used for config, drop-order, snapshot filenames, and the
    /// per-panel view state slot in AppState. Never derived from the title.
    fn spec(&self) -> &'static PanelSpec;

    /// Does this panel have anything to say right now? Pure.
    /// This is new and it is load-bearing: today the dashboard shows
    /// "no training run active" in a 6-row box forever.
    fn relevance(&self, st: &AppState, scope: ScopeArg) -> Relevance;

    /// Pure. No I/O, no clock, no mutation, no allocation of terminal state.
    /// Called at most once per panel per frame.
    fn render(&self, v: VariantId, area: Rect, buf: &mut Buffer, cx: &Cx<'_>);

    /// Declare (do not perform) a demand for a raster image inside `area`.
    /// Only the board panel implements this. Keeps `render` pure while making
    /// the picture's geometry a pure function of (state, area) that a test can
    /// assert on without a terminal.
    fn picture(&self, _v: VariantId, _area: Rect, _cx: &Cx<'_>) -> Option<PictureRequest> {
        None
    }

    /// Keys are turned into data. The panel never mutates; sf-app applies the Action.
    fn on_key(&self, _k: KeyEvent, _cx: &Cx<'_>) -> Option<Action> { None }
}

pub struct PanelSpec {
    pub id: PanelId,                     // enum-like newtype over &'static str
    pub title: &'static str,
    pub region: RegionId,                // Board | Rail | RailB | Band | Full
    pub scope: Scope,                    // Global | PerBot(focused) | PerBot(all)
    pub weight: Weight,                  // drop order inside a region. Weight::Pinned = never
    pub variants: &'static [Variant],    // most detailed FIRST, mins strictly decreasing
}

pub struct Variant {
    pub id: VariantId,       // Full | Compact | Line | Glyph
    pub min: Size,           // hard floor. Below this the variant is not offered at all.
    pub pref: Size,          // above this, extra space is wasted on this panel
    pub grow: u16,           // Fill weight for space beyond pref. 0 = fixed height.
}
```

Panels are zero-sized types with `&'static PanelSpec`. All mutable view state (moves-list scroll, which bot has focus, whether the tape is expanded) lives in `AppState.ui: BTreeMap<PanelId, PanelView>`. That is what lets `render` take `&self` and stay a pure function, and it is what makes every panel snapshot-testable at any scroll offset without constructing a panel object in a particular state.

**The registry, and the "one file plus one line" claim.**

```rust
// sf-panels/src/lib.rs   -- the ONLY file that changes when a panel is added
sf_layout::panels! {
    board, mind, moves, curve, results, train, machine, lab, api, versions, tape,
}
```

The macro expands each identifier to `mod <ident>;` and `&<ident>::PANEL` inside `pub const ALL: &[&dyn Panel]`. Adding a panel is: create `sf-panels/src/mind.rs` (or `versions.rs`), add one identifier to that list. That is genuinely one new file and one line — the `mod` declaration comes free from the macro, which is the whole reason to have one.

**I am rejecting `inventory`/`linkme` distributed slices**, even though they would get it to zero registration lines. Three reasons: the resulting order is link-order dependent, and this design's entire safety argument rests on determinism; the requirement is that *something enumerates panels*, and a distributed slice specifically destroys the enumeration you can read; and the list order in `panels!` is itself useful data — it is the tie-break for drop order, so the one line you add is not ceremony, it is a decision you were going to have to make anyway.

Compile-time-adjacent invariants, all asserted in one test in `sf-panels`:
- `PanelId`s are unique.
- Every `variants` slice is non-empty and strictly decreasing in `min` along both axes.
- Every `spec.region` appears in at least one `Composition`, and every `Composition`'s regions are all reachable.
- For every `Composition`, `Σ min of the pinned panels ≤ composition.min`. This is what makes the solver's over-constrained branch statically unreachable, which is the requirement you flagged about kasuari.

---

# 3. Responsive tiers

**Four tiers, and they only choose a composition. Everything else is the fit loop.** The mistake in `_wide_layout`/`_narrow_layout` is not that there were two branches; it is that each branch did its own arithmetic and there was no contract between them, so a fix in one could not be checked against the other. Tiers must be small, declarative, and *not* where sizes are computed.

| Tier | Trigger | Composition | Board |
|---|---|---|---|
| `Micro` | < 60 cols or < 20 rows | one region, one panel at a time, tab-cycled | text, 8px squares, or nothing |
| `Compact` | ≥ 60 cols, ≥ 20 rows | `Column[ Board, Rail ]` — stacked | text or a small picture |
| `Standard` | ≥ 100 cols, ≥ 34 rows | `Row[ Board, Rail ]` — today's layout | picture, sized to the region |
| `Wide` | ≥ 176 cols, ≥ 44 rows | `Row[ Board, Rail, RailB ]` | picture; second rail takes the width the board currently over-eats |

Tier is `the largest tier whose Composition.min fits`, computed in one function, in one file (`sf-layout/src/tiers.rs`). A `--tier` flag forces one, for screenshots and snapshots. `Wide` is new and it is the answer to "must look good full-screen": at 200+ columns the current single 60-column rail leaves the board eating width it cannot use (a square board is height-bound past ~1150px on a 44-row window), and a second rail is where `lab`, `versions`, `api` and the match/SPRT panel go.

**Compositions are hand-written trees, max depth 3, all four in one file, each with a snapshot test. There is no general 2-D packer and there will not be one.** A packer's output is not predictable enough to memorise, and a dashboard you look at every day should have things in the same place every time. Regions, not panels, appear in the tree; panels declare `region` in their spec, and within a region the fit loop consumes them in registry order.

**The fit loop — this is where the bug class dies.**

```
fit(region, extent_main, extent_cross, panels_in_region, state) -> Placed[]

1. Drop every panel whose relevance(state) == Hidden.
2. For each survivor, pick the first (most detailed) variant whose min fits
   extent_cross. If none fits, drop the panel. (Cross-axis filtering FIRST:
   a 52-column rail cannot host a 60-column variant, and discovering that
   after solving the main axis is how you get a clipped panel.)
3. loop {
       need = candidates.iter().map(|c| c.min.main()).sum();   // iterate the SAME Vec
       if need <= extent_main { break }
       victim = candidates.iter()
            .filter(|c| c.weight != Pinned)
            .min_by_key(|c| (c.weight, c.relevance, c.registry_order));
       match victim { Some(v) => if !v.demote() { candidates.remove(v) },
                      None => unreachable!("composition floor guarantees pinned fits") }
   }
4. // The set provably fits. Only NOW is kasuari allowed to see it.
   rects = Layout::vertical(candidates.map(Constraint::Min(min)))
              .flex(Flex::Legacy).spacing(0).split(area);
   // second pass distributes slack by `grow` among panels below `pref`
5. debug_assert!(rects.len() == candidates.len());
   debug_assert!(rects.iter().map(h).sum() == area.height);
   debug_assert!(zip(rects, candidates).all(|(r,c)| r.contains_size(c.min)));
```

The specific defect at `watch.py:289` becomes unrepresentable for a structural reason worth naming: **step 3 computes `need` by summing over the same collection it is about to lay out.** You cannot forget a term in a sum over a `Vec` the way you can forget `- self.results_h` in a hand-written expression. Nothing outside `fit` is permitted to compute a size, and the only way a panel obtains a `Rect` is by being handed one from `Solved`.

`Solved` also carries `dropped: Vec<(PanelId, DropReason)>` where `DropReason` is `Hidden | CrossAxisTooSmall{need,have} | Demoted | NoRoom{need,have}`. Bind `?` to a debug overlay that prints it. "Where did my machine panel go" was unanswerable for however long that bug has been live; it should cost one keypress.

Picture geometry comes out of the same solve, once:

```rust
pub struct PictureGeom { pub cells: Rect, pub px: u32, pub cell: CellSize, pub protocol: Protocol }
// px = snap26(min(cells.width  * cell.w, cells.height * cell.h))
```

`fit` computes it for the `Board` region; the board panel receives it in `Cx` and draws the player blocks around it; the emitter receives it from `Solved`. That is one source of truth replacing the current three (`Plan.image_*`, `board_panel`'s parameters, `verify_layout.py`'s recomputation).

One invariant with a test, from the Lab Notes: **the Board region never merges borders with a neighbour and never hosts an animated cell.** `Flex`/negative spacing is available and attractive at `Compact`, but a merged border landing on an image row is the "anything styled on a row the image occupies" failure. Encode it as `Composition` validation, not a comment.

---

# 4. Multi-bot data model

```rust
pub struct AppState {
    pub bots:    IndexMap<BotId, BotState>,   // config order, stable on screen
    pub focus:   BotId,
    pub machine: MachineState,     // GPUs (nvml), units (zbus), disk, journal counters
    pub train:   TrainState,       // one GPU, so global
    pub lab:     LabState,         // runs/lab/{state.json,log.jsonl,report.md}
    pub matches: MatchState,       // runs/matches/* : elo, interval, LOS, LLR  (new)
    pub api:     ApiState,         // per-endpoint buckets, 429s, effective intervals
    pub ui:      UiState,          // per-panel view state, focus, overlays
    pub tape:    Tape,             // entries tagged Option<BotId>
    pub engines: HashMap<Pid, EngineProc>,   // ALL engine processes, attributed or not
}

pub struct BotState {
    pub cfg:     Arc<BotConfig>,
    pub account: Tracked<Account>,             // /api/account
    pub playing: Tracked<Vec<PlayingGame>>,    // /api/account/playing
    pub games:   IndexMap<GameId, GameState>,
    pub focus_game: Option<GameId>,
    pub record:  Tracked<Record>,              // version-scoped W/D/L
    pub results: Tracked<Vec<FinishedGame>>,   // from PGNs, no API cost
    pub version: Tracked<Version>,             // VERSIONS.jsonl tail
}

pub struct GameState {
    pub id: GameId, pub meta: GameMeta, pub our_colour: Color, pub tc: TimeControl,
    pub position: Resolved,                    // §5. The ONLY position anyone reads.
    pub clocks:   Clocks,                      // server value + local tick + source
    pub moves:    Vec<MoveRec>,                // authority marked per ply
    pub curve:    BTreeMap<Ply, WpWhite>,      // always White's frame. Always.
    pub sf_curve: BTreeMap<Ply, WpWhite>,      // Stockfish, whole game
    pub grades:   BTreeMap<Ply, Grade>,
    pub search:   Option<SearchSnapshot>,
    pub pid:      Option<Pid>,
}
```

`Tracked<T>` replaces `Field` and closes its one loophole: `State.get()` currently hands you a value with the staleness discarded. New API is `fn get(&self) -> Option<(&T, Track)>` and there is **no** accessor that returns the bare value. The invariant the docstring claims ("you cannot read a value without being handed how old it is") becomes true rather than aspirational.

**`pid` as the partition key.** Every telemetry record carries it unconditionally and nothing reads it, so it is free and perfect. The pipeline is:

```
engine.jsonl line -> Record{pid, game: Option<GameId>, ..}
  attribution: game -> (BotId, GameId)   via each bot's `playing` list
               pid  -> sticky cache of the above, so `think` records with the same
                       pid land on the right game even before/after `game` appears
  unattributed pids go into AppState.engines with Attribution::Unknown and are
  shown as "N detached engines" — visible, not guessed at.
```

**Delete the entire 4-layer demultiplexer.** `_bridge`, `_orphans`, `reanchor`, the `stm`-based colour disambiguation, the placement-FEN `history` set — all of it exists solely because `game` was absent from records. It is now present (patches/0003). Keep exactly one fallback, narrow and honest: a record with no `game` whose FEN matches exactly one bot's exactly one game's resolved position chain is attributed; otherwise it is `Unknown` and contributes to nothing. Do not resurrect inference. There should be a test that asserts an unattributable record affects zero on-screen values — the opposite of today's behaviour, where an unattributable record was aggressively adopted.

One trap to write down since the fallback touches it: `shakmaty::Board` is `Eq + Hash`, which is excellent, but it is **placement only** — no turn, castling or ep. That is the same comparison `fusion.py` does with `board_fen()`, and it is precisely why two games in the same opening confused it. Use `Chess`/`Setup` equality when you mean "same position", and reserve `Board` equality for the cheap pre-filter.

**Per-bot telemetry files** are supported (`CHESSGPU_TELEMETRY` per unit, config `telemetry = "..."`) but not required, because pid partitioning makes one shared file correct. If two bots name the same path, the tailer is shared and dedup is by `(pid, game)`.

**Config surface.** TOML at `etcetera`'s config dir (`~/.config/sumofish/dash.toml`), with `--config` override:

```toml
[dash]
fps = 12
board = "auto"            # auto | kitty | sixel | text | off
tier  = "auto"            # auto | micro | compact | standard | wide  (screenshots)

[api]                     # GLOBAL. The limits are per-IP, so they cannot be per-bot.
min_gap = "250ms"
game_streams = 2          # our share of the documented 8. See the wall below.
[api.endpoint."account/playing"]   rpm = 30   # per-ENDPOINT bucket, shared by all bots
[api.endpoint."account"]           rpm = 6
[api.endpoint."game/export"]       rpm = 10

[[bot]]
id = "sumofish"
user = "SumoFish"                       # never a constant in code
token_file = "~/.config/sumofish/bot.env"
token_var  = "LICHESS_BOT_TOKEN"
unit       = "sumofish-bot"
telemetry  = "/home/nomad/dev/active/sumofish/logs/engine.jsonl"
pgn_dir    = "/home/nomad/dev/active/sumofish/logs/games"
versions   = "/home/nomad/dev/active/sumofish/VERSIONS.jsonl"

[machine]
gpus  = "all"
units = ["sumofish-bot", "sumofish-train", "sumofish-lab",
         "sumofish-rating.timer", "sumofish-train-watchdog.timer"]
[lab]  state = ".../runs/lab/state.json"
[train] active = ".../runs/active.json"
```

Note what is *global* and why: everything rate-limited. Everything per-bot is a name, a credential, a file path, or a unit.

**The API budget governor.** One `sf-sources::api::Governor` task owns every outbound lichess request; no other task holds a `reqwest::Client`. Sources send `ApiRequest` on an mpsc and await a oneshot. This makes four things structural rather than hopeful:

1. **"Only make one request at a time"** (the only concrete guidance lichess publishes) plus the 250ms pace — one task, one in-flight, as `sources.GATE` does today but without the possibility of a source bypassing it.
2. **Buckets are keyed by endpoint, not by bot.** This is the correction to the obvious per-bot design: the limit is per-IP-per-endpoint independent of credential, so four bots polling `/api/account/playing` every 2s is 120 req/min against one bucket, not 4×30. So sources **do not choose their own cadence** any more. They register a `PollSpec { endpoint, key, want_interval, floor_interval, priority }` and the governor issues permits round-robin, so `playing`'s effective interval degrades to `n_bots × base` automatically and *visibly*. That degradation is the honest answer to "ability to add more bots as it grows" and it belongs on screen.
3. **Forbidden endpoints are not in the type.** `enum Endpoint` has no variant for `/api/stream/event` — one per token, and lichess-bot owns it for every token. The dashboard must never open one, and the way to guarantee that is to make it unnameable.
4. **429 handling is per-endpoint, backoff-only, never retry.** The Lab Note is explicit that every retry against a 429 renews the penalty, and it cost 40 minutes of a non-playing bot. Penalty box per endpoint with exponential backoff and a hard "no retry of the failed call".

**The hard scaling wall, stated plainly.** Max 8 concurrent `/api/stream/*` per IP. lichess-bot at `concurrency: 2` already holds 1 event stream + up to 2 game streams **per bot**. At two bots that is 6 of 8, and the dashboard gets 2. At three bots lichess-bot alone wants 9 and the bots break each other before the dashboard gets a look in. Conclusion, and it should shape the whole design: **the dashboard streams at most the focused game, and derives everything else from local telemetry plus `/api/account/playing`.** That is not a compromise forced by the budget; per §5 the local telemetry is strictly fresher anyway.

---

# 5. Position resolution and the move-delay fix

**One component, `sf-model::position::Resolver`, pure, no I/O, one per game.** It takes observations and produces the single answer everybody reads. Today that decision is made in `watch.py::_live_position` *and* in `panels.board_panel`, with a docstring asking a human to keep them in sync — the classic setup for the eval gauge and the picture disagreeing about what is on the board.

```rust
pub enum Obs {
    /// telemetry `think`/`move`: the position the engine is searching, i.e. AFTER
    /// the opponent moved. Local. Instant.
    EngineSearching { pid: Pid, game: GameId, pos: Chess, ply: Ply, at: Instant },
    /// telemetry `move` with done+uci: our own chosen move. THE FIX.
    EngineChose     { pid: Pid, game: GameId, from: Chess, mv: Move, at: Instant },
    /// lichess game stream: authoritative, carries clocks, possibly +3 moves delayed.
    Stream          { game: GameId, pos: Chess, last: Option<Move>, clocks: Clocks,
                      ply: Ply, at: Instant },
    /// /api/account/playing: fen + lastMove + secondsLeft, no extra request.
    Playing         { game: GameId, pos: Chess, last: Option<Move>, at: Instant },
    Terminal        { game: GameId, status: Status, winner: Option<Color> },
}

pub struct Resolved {
    pub pos: Chess, pub last: Option<Move>, pub ply: Ply,
    pub source: Source, pub as_of: Instant,
    /// how far behind the freshest source the authoritative one is, in plies and
    /// seconds. Currently invisible; belongs on screen.
    pub lag: Lag,
}
```

Rules, ranked and total:

1. **Highest ply wins; ties break by source rank** `EngineChose > EngineSearching > Playing > Stream`. Every observation carries `game`, so cross-game contamination is impossible by construction — the whole reason the old anchor machinery existed is gone.
2. **`EngineChose` synthesises the post-our-move position immediately** by applying `uci` to the record's own `fen`. This is the dead block at `fusion.py:182-191` restored as a first-class observation, and it fixes both halves of your finding (a): the board no longer shows the pre-move position for ~9s after SumoFish moves, and our own move gets its `last`-move highlight even when the opponent replies before the stream catches up.
3. **Ply never regresses within a game id.** Kills the reconnect-replay flicker without needing `published_ply` bookkeeping in the stream reader.
4. **Disagreement is surfaced, not arbitrated away.** When `Stream` and `EngineSearching` describe different positions at the same ply, that is a fact worth a coloured indicator (`sync: engine +2 ply, feed 8.7s`), because otherwise the next person to notice the lag will re-measure it and re-attribute it to this code, which the Lab Notes say has already happened once and cost two rounds of optimising a 400ms render under an 8,700ms delay.
5. **The move list is authoritative from the stream when present, synthesised from the engine chain otherwise, and each ply is tagged with which.** Synthesised plies are missing exactly one case — a game that ends on the opponent's move, which we never search — and tagging makes that visible instead of looking like a lost move.

**On the delay.** You are right that it is documented and deliberate, and the corollary is that `/api/stream/game/{id}` should be labelled `+3 moves` on screen permanently. But I would **not** build the design on `/api/bot/game/stream/{gameId}`, because §0 confirms lichess-bot holds that exact stream, for that exact game, with that exact token, and the downside is a live rated game. Sequence it this way instead:

- **Foundation:** telemetry-primary. With fix (2), the local channel already carries *both* sides' moves at zero latency — ours from `done`+`uci`, theirs from the FEN of the next search, which starts within milliseconds of lichess-bot receiving their move. The only things telemetry cannot give are clocks, the authoritative result, and the final move of a game we did not search.
- **Clocks, in preference order:** (i) `/api/account/playing`'s `secondsLeft` at governor cadence plus a local tick and resync — zero extra requests, since we poll it anyway; (ii) lichess-bot's own journal, which logs each move and is local and free; (iii) the delayed public stream, labelled. Which one wins is a day-1 measurement, not a design decision (§6, F3/F5).
- **The undelayed bot stream is an optional enhancement**, behind `[api] bot_stream = false`, enabled only if F4 passes cleanly. If it passes, it is strictly better and gets clocks and results too.

Everything the old fusion module did to compensate — bridge search, orphan queue, reanchor, `stm` disambiguation — deletes. That is ~200 lines of the subtlest code in the project going away because a key was added upstream and a `return` was in the wrong place.

---

# 6. Build order

Each milestone leaves a program you can run, and each has a verification you can execute without a chess game happening.

**M0 — Falsification day. No architecture is committed until this reports.** An `xtask probe` binary that answers, prints, and checks in `dashboard/docs/probe-results.md`:

- **F1** Does Konsole 26.04.3 accept kitty `a=T,f=100` chunked base64 PNG placed by cursor move, with `q=2` to suppress replies? At what chunk size? Does `a=d,d=i` delete work, or is `ESC[2J` still the only eraser? *Method:* write bytes to `/dev/tty` from a program in an already-correctly-sized window (never resize another process's pty — Lab Note), then one screenshot.
- **F2** Does it accept `f=32` raw RGBA? If yes, the PNG encode disappears from the hot path entirely, at the cost of ~4x the escape-stream bytes.
- **F3** Does the image survive a full ratatui redraw with `CellDiffOption::Skip` on the region? Verify twice: visually, and headlessly with a `Backend` wrapper that records every byte written and asserts zero writes inside the picture rect.
- **F4** Does a second `/api/bot/game/stream/{gameId}` with the bot's token coexist with lichess-bot's? *Method:* a casual game only, never rated; tail the bot's journal for stream drops during the window; abort criterion is any drop attributable to our connection, in which case the idea is dead and gets a Lab Note. Patch 0001 bounds the damage.
- **F5** Is `/api/account/playing`'s `fen`/`lastMove` subject to the 3-move delay? *Method:* 20 plies, compare arrival against telemetry timestamps; undelayed looks like ≤ the poll interval, delayed looks like ~8.7s.
- **F6** Does usvg 0.47 + resvg + tiny-skia match rsvg for cburnett? *Method:* eight positions, both paths, pixel diff; plus the seam assertion (zero non-square-colour pixels on a scanline at 1144, nonzero at 1152).
- **F7** nvml per-process VRAM readable for processes under the systemd sandbox; zbus reaching the **user** bus for `--user` units.

This is the biggest-unknown-first ordering you asked for, and it is also the cheapest milestone. Two of these seven can invalidate a settled decision, and finding out in an afternoon costs nothing.

**M1 — Skeleton.** Workspace, config load, `tokio::select!` loop, `watch` shutdown, tracing to `logs/dash.log`, `sf-term` probe, the tier system, the fit solver, and exactly one panel (`header`). *Verify:* the 78k-cell size grid test (below) passes; `sumofish-dash --tier standard --size 150x44 --snapshot` prints a frame to stdout; running it in a terminal renders, resizes and exits cleanly on `q` and Ctrl-C.

**M2 — The picture.** `sf-board`: the ported cburnett SVG generator, usvg → Pixmap, PNG + sixel + half-block emitters, and the encoder task (newest-wins, drop stale) mirroring `sixel.Renderer`'s proven design. *Verify:* `sumofish-dash board --fen … --px 1144 --png /tmp/a.png` plus golden pixel hashes; the seam test as an actual assertion; the "never emit an image whose key's px differs from the current layout's" rule as a unit test on the emitter.

**M3 — Telemetry and the resolver, offline first.** Replay `logs/engine.jsonl.1` (17MB of real two-game-interleaved data, already on disk) through the resolver in a test before any live tailing. Then the inode-keyed tailer, then live. *Verify:* the golden replay test below; `sumofish-dash replay logs/engine.jsonl.1 --speed 60` becomes a first-class subcommand and **replaces `seed_demo` entirely** — a fixture built from real recorded telemetry is strictly better than a hand-written one, and it makes `sumofish demo` a regression test you can watch.

**M4 — lichess.** Governor, `/api/account`, `/api/account/playing`, the game stream (delayed or bot, per F4), finished-game export, PGN results via `pgn-reader`. *Verify:* the governor test with `tokio::time::pause()` — six synthetic bots, virtual clock, assert no endpoint bucket is ever exceeded and no 429 is ever retried. Deterministic, instant, no network.

**M5 — Machine.** nvml (util, per-process VRAM, temp, power, clocks, throttle reasons), zbus systemd (ActiveState/SubState/NRestarts/Result/NextElapse), journal follower with cursor-resume. *Verify:* a diff test against `nvidia-smi --query-compute-apps` and `systemctl --user show` output.

**M6 — The remaining panels, one commit each.** `mind`, `moves`, `curve`, `results`, `train`, `machine`, `tape`, plus the new `lab`, `matches`, `versions`, `api`. Each commit is one new file plus one identifier in `panels!`, plus generated snapshots. The diffs are the proof that §2 delivered.

**M7 — Multi-bot.** Second `[[bot]]` block; focus switching (`Tab`, `1`-`9`); `PerBot(all)` scope panels; per-bot rows in the header. *Verify:* run with a second bot whose sources are all fixtures and assert total API traffic is unchanged; run with two real bots and assert the governor's effective `playing` interval doubled and said so on screen.

**M8 — Stockfish grader, `Wide` tier's second rail, cutover.** The hand-rolled UCI driver is last because it is the only source whose absence costs nothing.

---

# 7. Verification strategy

**Layout, headlessly, at every size.** One test that iterates the full cross product — cols 40..=320, rows 12..=120, ~29k solves, each pure and microseconds — and asserts:

- Σ of every region's children == the region's extent. (Catches the +8 overflow directly.)
- Every rect ≥ its panel's variant `min` in both axes. (Catches clipping.)
- No two rects intersect, over all pairs. (Catches the picture-under-the-tape class.)
- `picture.cells ⊆ board_region`, and `picture.px ≡ 0 mod 26`.
- **Presence monotonicity:** if panel P is placed at `(c, r)`, it is placed at every `(c', r')` with `c' ≥ c, r' ≥ r`. This is the assertion that fails at rows 50-78 today, and it is worth having as the flagship test because non-monotone presence is *always* a bug and is otherwise invisible.
- Tier is monotone in both axes.

This is the replacement for `tests/verify_layout.py`, and the difference in scope is the point: the Python gate covers the board column only, over 8 hand-chosen sizes, which is exactly why the `machine` bug is green.

**Panels in isolation.** `insta` + `TestBackend`, table-driven over `(panel, variant, size)` from the registry itself, so a new panel gets its snapshots by being registered — no new test file. Fixed `now` from `Cx` makes clocks and staleness ages deterministic.

**Colour, which `insta` cannot see.** Three separate assertions, because you are right that snapshots are text-only:

1. **Palette conformance.** Walk the rendered `Buffer`, collect every distinct fg/bg `Color`, set-difference against the 24 theme hexes plus the 6 declared `PICTURE_*` colours. Any stray colour fails. This makes theme drift a build failure across every panel at every size — which is a stronger guarantee than the Python version has ever had.
2. **Contrast floors.** For the gauge widgets, compute WCAG contrast between fill and ground and assert ≥ a floor. This encodes the `#32302f`-on-`#282828`-is-1.12:1 Lab Note as a test, so the class cannot recur.
3. **Colour-map snapshots.** Render the buffer a second time as a grid of *theme colour initials* (one char per named colour) and `insta`-snapshot that as text. This is the trick that gets colour into `insta` after all: the snapshot diff shows you a picture of where each ink went, and it catches "the whole panel went dim" which a text snapshot cannot.

**The position resolver.** Golden replay. Record once (`sumofish-dash record --out tests/fixtures/`) a slice of `engine.jsonl` with two interleaved games plus the matching stream ndjson, then:

- Snapshot the per-frame resolved position sequence against a virtual clock.
- Assert our own move appears in the same frame as its `done` record. **This is the regression test for the `fusion.py:180` bug and it should be written before the fix.**
- Assert no ply regression, and zero records from game B affecting game A's curve, grades or board.
- **Property test: the resolver's final state is invariant to arrival order** of observations within a 10-second window. Shuffle, re-run, compare. This is precisely the property the old demuxer lacked, and it is why interleaving produced "the graph says one side is winning when it is not".

**The picture pipeline.** Golden PNG hash per `(fen, px, flip, lastmove)`; the seam count assertion at multiples of 26 and its negative at 1152; a round-trip test that parses our own kitty escape sequence back out and checks the chunking and terminators; and a byte-recording `Backend` that proves zero writes land inside the picture rect over a hundred frames of neighbouring cells changing. Per the Lab Note, screenshots judge only how it *looks*, at one size, never whether numbers add up.

**Sources.** Each parses bytes → `Vec<Update>` as a free function, tested against recorded fixtures. Only the ndjson decoder and the JSON shapes get abstracted behind traits; `reqwest`, `nvml` and `zbus` are used concretely and covered by the diff tests in M4/M5. Do not build a mock layer for things whose failure modes are network conditions.

**`apply()` fuzz.** `Update` sequences generated at random must never panic and must never lower a game's ply. Cheap, and it is the function every source funnels through.

---

# 8. Migration and coexistence

- **Location:** `dashboard/` inside `/home/nomad/dev/active/sumofish`, one Cargo workspace, same git history. `CLAUDE.md`'s Layout section grows one entry. Binary at `dashboard/target/release/sumofish-dash`, invoked through `bin/sumofish-dash` following the existing wrapper pattern so paths stay out of the units.
- **`~/bin/sumofish` dispatch** keeps being the front door. During the build, add cases without touching the defaults: `sumofish` → Python (unchanged; it is the working instrument for a bot that is playing right now), `sumofish next` → Rust. At cutover, flip: `sumofish` → Rust, `sumofish py` → Python, for one release cycle. Then delete the `py` case.
- **Yes, the Python version stays available throughout** — losing observability on a live rated bot for a week is not a trade worth making. But there is a hazard you have not mentioned: **two dashboards double the load on the same per-IP endpoint buckets**, which is what already put the bot into a 40-minute no-challenge stall once. So: the Rust binary defaults to `--offline` until M4 is done (telemetry + nvml + systemd + lab only, zero lichess), and from M4 onward both programs take a pidfile lock at `$XDG_RUNTIME_DIR/sumofish-dash.lock`; the Rust one refuses to open lichess sources while the Python one holds it, and says so on screen. Cheap, and it removes a real way to break the live bot while building its replacement.
- **What happens to `scripts/dash/` at the end:** deleted, in one commit, together with `scripts/watch.py` and `tests/verify_layout.py`. That last one matters — a layout gate for a deleted program is a test of nothing that still passes green, and PHILOSOPHY's own rule is that retracted things get deleted at the point of use rather than annotated.

  Worth noting what else deletes: the Rust text fallback rasterises the **same** `Pixmap` at 8/16/24px and half-blocks it, so one rasteriser feeds all three outputs (kitty PNG, sixel, half-blocks). That removes `sprites.py` (1,481 generated lines), `make_sprites.py` (124), `ink.py` (78) and the whole two-byte luminance/alpha two-ink scheme, along with its "widening cburnett's stroke swallows the king's cross" workaround. ~1,700 lines gone for a strictly better result, and the `make_sprites` regeneration dependency on `rsvg-convert` and ImageMagick goes with it.
- **Subcommands:** `sumofish mind` becomes `sumofish-dash replay --follow`; `rating`, `log`, `train` keep shelling out for now and move in later. `sumofish-lab` and `sumofish-games` stay Python — they are separate programs and out of scope, though the new `lab` and `matches` panels make the first one much less necessary.

---

# What is worth tracking and is not, or is tracked and not shown

You asked for a survey. Everything below is data that already exists on this machine.

**Tracked, never shown.** `pid` on every telemetry record (the partition key, zero readers). The `boot` record's `policy_step`, `value_step`, `bins`, `sims`, `batch`, `params` — nothing on screen says *which checkpoint is playing*, which is the single most PHILOSOPHY-relevant fact about a live game and it is invisible. `logs/rating.jsonl`'s `deployed{engine, policy, value, current}` hashes — same story, and they are how you would know a promotion landed mid-session. `runs/lab/state.json` + `log.jsonl` + `report.md`: an entire autonomous system with per-job outcomes, decisions, verdicts and durations, with zero dashboard presence. `runs/matches/*`: Elo, ±interval, LOS, LLR — the measurement instrument's output, and PHILOSOPHY says a number without an interval is not a result, so the live match's interval and SPRT state belong on screen while it runs. `VERSIONS.jsonl`: version, title, sha, checkpoint sha, step, width, layers, ratings at cut — only `version` is read. Stockfish grades are computed for every ply and rendered only as `?!`/`?`/`??` marks; the *distribution* (accuracy %, blunders per game, ACPL) is free from the same numbers and is the only instrument here that answers "is it playing well" independently of the rating. The engine's own curve versus Stockfish's — both exist, and their disagreement is calibration data. `Telemetry.dropped` exists in the writer and is never emitted into a record at all.

**Not tracked, worth adding.** Per-process VRAM from nvml, which answers "is the trainer starving the bot" exactly instead of by inference from one aggregate — the exact mistake CLAUDE.md records as costing a whole session. Throttle reasons, SM/mem clocks, power limit versus draw. **Budget overruns:** every record has `elapsed` and `budget`, so "moves that overran their deadline" is free and it is the failure mode that loses games on a clock. Lowest clock reached per game. Systemd timer `NextElapse` (when does the next rating sample land — currently unanswerable) and unit `Result`/`ExecMainStatus` on failure. `logs/engine.jsonl` size against its 16MB rotation, and free space on the volume holding 142MB checkpoints. A per-unit count of journal ERROR/WARN lines in the last hour — the 40-minute matchmaking stall had no on-screen signal at all. Held-out loss: the lab's own job summaries carry `val 2.5653` and PHILOSOPHY says select on held-out loss, but the train panel shows loss and puzzle accuracy only. Average opponent rating and rating-weighted score, which is what makes a W/D/L record interpretable at all.

---

# Where your settled decisions look wrong, or need a caveat

**1. `ratatui-image`: do not use it. Drive kitty directly.** Four reasons, in order of weight. (a) It marks Konsole+sixel broken for graphics *clearing* — that is precisely the failure class behind eight of your Lab Notes, so you would be inheriting an unsolved problem wrapped in an abstraction. (b) Default features pull libchafa via pkg-config, a C dependency for a half-block renderer you get for free from the Pixmap you already have. (c) Its protocol selection is a runtime capability model that will not match Konsole's "experimental, direct transfer only" kitty support, so you would be overriding it anyway — and overriding a capability detector is worse than not having one. (d) `ThreadProtocol` is ~150 lines of a pattern you have already written once and proved (`sixel.Renderer`: newest-wins, drop stale, bounds lag at one render). What you *should* take from ratatui 0.30 is `CellDiffOption::Skip` itself — that is in ratatui, not in ratatui-image, and it is the actual fix. Keep ratatui-image on the shelf as an optional feature if you ever want iTerm2/WezTerm; read its source for the escape sequences.

**2. kitty primary is right; two caveats.** Direct transfer means the whole PNG is base64 in the escape stream in 4096-byte chunks — roughly 27 chunks for a 1144px board. Fine at one move per nine seconds, catastrophic if a resize ever re-transmits at frame rate, so the "emit only when the key changes, and never an image drawn for another layout" rules must survive verbatim into the Rust design (they do: the key contains `px`). And because Konsole's support is self-described experimental, the probe must be a *positive* query (`a=q`) with a config allowlist fallback — the Lab Note is "do not assume a terminal cannot do something without asking", and its mirror is equally true here. If `a=d` delete turns out broken, `ESC[2J` on a geometry change stays the eraser.

**3. `/api/bot/game/stream/{gameId}` is the riskiest single item in the plan** and §0 confirms the collision is real. Make it an opt-in enhancement gated on F4, not the foundation. Per §5 the foundation does not need it.

**4. `read_timeout` not `timeout`** is right, and there is a refinement: the *public* stream sends no keepalives (35s of silence measured, which is why 180s), but the *bot* stream does send keepalive newlines. So if F4 passes, that endpoint's read timeout can drop to ~20s and detect a genuinely dead socket nine times faster. Per-endpoint timeouts, from config.

**5. `Board` is `Eq + Hash` — true, and placement-only.** No turn, castling or ep. That is the same comparison `fusion.py` makes, and it is why two games in the same opening confused it. Use it as a cheap pre-filter; use `Chess`/`Setup` when you mean "the same position".

**6. Three gaps in the dependency list.** No PGN reader — `logs/games/*.pgn` carries the entire local game record at zero API cost and it is a multi-line format; use `pgn-reader` (same author as shakmaty, streaming) rather than hand-rolling, because PGN header parsing is a chore and PHILOSOPHY says chores stay chores. No date library — the PGN `UTCDate`/`UTCTime` → local-time conversion the Python does with `strptime` needs one; take `jiff` 0.2. And `indexmap`, for insertion-ordered per-bot and per-game maps so the screen order is config order and never a hash order.

**7. serde_json is indeed plenty fast** — and the interesting consequence is not performance, it is freedom: 17MB parses in tens of milliseconds, so the 256KB backfill window can become "index the whole tail by pid" and the "attaching mid-game shows an empty panel" failure disappears completely rather than being mitigated.

**8. `journalctl -f -o json` as a subprocess** is right, with two additions the Python lacks: use `--since` on attach so you get context (the same reasoning as the telemetry backfill Lab Note), and capture `__CURSOR` from each record so a supervised restart resumes exactly where it stopped rather than losing or duplicating the gap.

**9. "25 named colours" is 27 names over 24 hexes**, and the six colours the picture uses are not in the theme at all. Name them, and exempt them from the palette test, or the first person to enforce "keep to the theme" will gruvbox the board and destroy the one property that makes a real image worth having.

---

# Riskiest assumptions, and how to falsify each

| # | Assumption | Falsification | If it fails |
|---|---|---|---|
| R1 | Konsole accepts kitty `a=T,f=100` direct-transfer PNG, placed by cursor move | F1: write to `/dev/tty` in a pre-sized window, one screenshot | sixel primary via `icy_sixel`, `max_colors:64, diffusion:0.0` — a known-good path already proven here |
| R2 | The kitty image survives ratatui redraws under `CellDiffOption::Skip` | F3, twice: visually, and a byte-recording `Backend` asserting zero writes in the rect | the picture region becomes a hard-reserved area the buffer never touches, re-emitted on key change only (today's behaviour, kept) |
| R3 | Konsole's `a=d` delete works | F1 | `ESC[2J` on geometry change, exactly as today |
| R4 | A second bot game stream coexists with lichess-bot's | F4, casual game only, journal watched, abort on any drop | telemetry-primary + `playing` clocks; label the public stream `+3 moves`; write the Lab Note |
| R5 | `/api/account/playing` is not 3-move delayed | F5: 20 plies against telemetry timestamps | clocks from lichess-bot's journal (local, free), else the delayed stream, labelled |
| R6 | usvg/resvg matches rsvg for cburnett (no font fallback, correct fill-rule) | F6: 8 positions, pixel diff <0.1%, zero seam pixels at 1144 | keep `rsvg-convert` as a subprocess for the picture only, and lose 54ms |
| R7 | The hand-ported `chess.svg.board()` is faithful | Extract cburnett path data ONCE via `xtask codegen` from the installed python-chess, check in the generated file, then diff our generated SVG string against python-chess's own output over 200 random positions. This is CLAUDE.md's own "exec the original and diff against it" rule | fix the port; the diff tells you exactly which glyph |
| R8 | ratatui 0.30's `Layout` returns rects matching the mins we pre-filtered on (no Fill/Min rounding surprise) | the 29k-size grid test, asserting `rect ≥ min` | tighten the second (slack) pass; worst case do slack distribution by hand and let kasuari see only `Length` |
| R9 | nvml per-process VRAM is readable, and zbus reaches the **user** bus | F7: diff against `nvidia-smi --query-compute-apps` and `systemctl --user show` | keep `nvidia-smi`/`systemctl show` subprocesses for those fields; they work today |
| R10 | Cell size is stable after startup (font change, DPI change) | change the font mid-run and watch the picture misalign | re-probe on every resize; it is cheap, and I would just do this unconditionally |
| R11 | The engine's `done`+`uci` record always arrives before the opponent's reply is searched | replay test over the recorded log; count inversions | the resolver's ply-monotonicity rule already handles it; the highlight is lost for that ply, nothing else |
| R12 | pid alone partitions correctly — no engine process is ever reused across games | replay test asserting each pid maps to ≤1 `game` at a time; and check `lichess-bot`'s engine lifecycle | key on `(pid, game)` rather than `pid`, and treat a pid whose `game` changed as a new partition |

The two that would actually change the architecture are R1 and R4, which is why M0 exists and why it is one day long rather than folded into M1.

**Files most critical to implementing this plan**

- `/home/nomad/dev/active/sumofish/scripts/watch.py` — `Plan._wide_layout` / `_narrow_layout` (the geometry being replaced), `_live_position`, `_image_key`, `board_image_tick`, and the six hardcoded `USER` sites
- `/home/nomad/dev/active/sumofish/scripts/dash/fusion.py` — the dead `done`+`uci` block at line 180 and the demux machinery being deleted
- `/home/nomad/dev/active/sumofish/scripts/dash/sources.py` — `_Gate`, `Tailer`, `GameStream`, `EngineTail._backfill`, `_claims`, `Grader`: every source's proven behaviour, to be ported not reinvented
- `/home/nomad/dev/active/sumofish/sumofish/telemetry.py` — the record schema, `pid`/`game` stamping, the durable/droppable distinction to mirror in the mpsc backpressure
- `/home/nomad/dev/active/sumofish/scripts/dash/theme.py` and `/home/nomad/dev/active/sumofish/scripts/dash/sixel.py` — the 24 theme hexes, the 6 picture colours, `GRID_PX`/`snap`, `Renderer`'s newest-wins contract, and `emit`/`clear`'s ordering rules
- `/home/nomad/dev/active/sumofish/tests/verify_layout.py` — what the new size-grid gate must subsume, and the eight sizes it currently checks