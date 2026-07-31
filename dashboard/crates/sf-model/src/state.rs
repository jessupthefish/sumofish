//! `AppState` — everything on screen, owned by exactly one task.
//!
//! Per-bot where the thing belongs to a bot, global where it belongs to the
//! machine. The Python had fifteen single-slot fields plus one `curve` and one
//! `eval_smooth`, and its `State` docstring called itself "the single object every
//! source writes" -- true, and the reason a second bot was unrepresentable.
//!
//! Insertion-ordered maps throughout (`IndexMap`), so the order on screen is the
//! order in the config file and never a hash order that moves between runs.

use crate::position::{GameId, Ply, Resolved, Resolver, Source as PosSource};
use crate::tracked::{Track, Tracked};
use indexmap::IndexMap;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Config-file identity of a bot. Short, stable, filesystem-safe.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct BotId(pub String);

impl std::fmt::Display for BotId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// So `map[&"sumofish"]` works without allocating a `BotId`. Sound because the
/// derived `Hash`/`Eq` delegate to the inner `String`, whose `Borrow<str>` impl
/// hashes identically.
impl std::borrow::Borrow<str> for BotId {
    fn borrow(&self) -> &str {
        &self.0
    }
}

/// The per-bot facts that come from config rather than from an API. The lichess
/// username lives here and **never as a constant in code** -- it was hardcoded in
/// six places in the Python, including one inside `_we_are_black()` that the
/// `--user` flag could not reach and that decided which way the eval gauge read.
#[derive(Clone, Debug)]
pub struct BotConfig {
    pub id: BotId,
    pub user: String,
    pub label: String,
    pub unit: Option<String>,
    pub telemetry: std::path::PathBuf,
    pub pgn_dir: Option<std::path::PathBuf>,
    pub versions: Option<std::path::PathBuf>,
    pub rating_log: Option<std::path::PathBuf>,
    /// Whether to spend any lichess budget on this bot at all. This is the knob
    /// that makes "add more bots as it grows" affordable.
    pub watch: bool,
}

#[derive(Clone, Debug, Default)]
pub struct AppState {
    pub bots: IndexMap<BotId, BotState>,
    pub focus: Option<BotId>,
    pub machine: MachineState,
    pub train: TrainState,
    pub lab: LabState,
    pub matches: MatchState,
    pub api: ApiState,
    pub ui: UiState,
    pub tape: Tape,
    /// Every engine process seen in the telemetry, attributed or not. An
    /// unattributed one is shown as detached rather than guessed at.
    pub engines: HashMap<crate::position::Pid, EngineProc>,
}

impl AppState {
    pub fn focused(&self) -> Option<&BotState> {
        self.focus.as_ref().and_then(|id| self.bots.get(id))
    }
    pub fn focused_mut(&mut self) -> Option<&mut BotState> {
        let id = self.focus.clone()?;
        self.bots.get_mut(&id)
    }
    /// The game a panel should draw: the focused bot's focused game.
    pub fn focused_game(&self) -> Option<&GameState> {
        let b = self.focused()?;
        b.focus_game.as_ref().and_then(|g| b.games.get(g))
    }
    pub fn bot_mut(&mut self, id: &BotId) -> Option<&mut BotState> {
        self.bots.get_mut(id)
    }
}

#[derive(Clone, Debug)]
pub struct BotState {
    pub cfg: Arc<BotConfig>,
    pub account: Tracked<Account>,
    pub playing: Tracked<Vec<PlayingGame>>,
    pub games: IndexMap<GameId, GameState>,
    pub focus_game: Option<GameId>,
    pub record: Tracked<Record>,
    pub results: Tracked<Vec<FinishedGame>>,
    pub version: Tracked<Version>,
    pub rating_log: Tracked<Vec<RatingSample>>,
    /// Restarts the *watchdog* performed. Deliberately not systemd's
    /// `NRestarts`: `watchdog.py` uses `systemctl kill` + `restart` rather than
    /// letting `Restart=always` fire, so `NRestarts` stays 0 and the Python
    /// panel therefore showed a green dot through two SIGKILLs today. Verified
    /// over D-Bus in the M0 probe.
    pub watchdog: Tracked<WatchdogState>,
    /// The bot's own journal, raw and unfiltered -- what `sumofish log` used to
    /// shell out to `journalctl -f` for. Distinct from `tape`: most of what a
    /// process logs matches none of `journal::Parser`'s recognized patterns and
    /// would otherwise never appear anywhere in the dashboard at all.
    pub log: LogLines,
}

impl BotState {
    pub fn new(cfg: Arc<BotConfig>) -> Self {
        let s = Duration::from_secs;
        Self {
            cfg,
            account: Tracked::new("account", s(45)),
            playing: Tracked::new("playing", s(6)),
            games: IndexMap::new(),
            focus_game: None,
            record: Tracked::new("record", s(1800)),
            results: Tracked::new("results", s(120)),
            version: Tracked::new("version", s(600)),
            rating_log: Tracked::new("rating_log", s(120)),
            watchdog: Tracked::new("watchdog", s(60)),
            log: LogLines::default(),
        }
    }
    /// The game to draw, preferring the one we are actually thinking about.
    pub fn game(&self) -> Option<&GameState> {
        self.focus_game.as_ref().and_then(|g| self.games.get(g))
    }
}

#[derive(Clone, Debug, Default)]
pub struct Account {
    pub username: String,
    pub title: Option<String>,
    pub perfs: IndexMap<String, Perf>,
    pub games_total: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Perf {
    pub rating: i32,
    pub games: u32,
    /// Rating deviation. Dropped by the Python; a provisional 2400 and a settled
    /// 2400 are different opponents and different claims.
    pub rd: i32,
    pub provisional: bool,
    pub prog: i32,
}

/// One entry of `/api/account/playing`.
#[derive(Clone, Debug)]
pub struct PlayingGame {
    pub id: GameId,
    pub our_colour: shakmaty::Color,
    pub fen: String,
    pub last_move: Option<String>,
    pub is_my_turn: bool,
    pub seconds_left: Option<u32>,
    pub opponent: Opponent,
    pub speed: String,
    pub rated: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Opponent {
    pub username: String,
    pub title: Option<String>,
    pub rating: Option<i32>,
    /// A provisional opponent's rating is noise. The Python took title/name/rating
    /// and dropped this.
    pub provisional: bool,
    pub ai_level: Option<u32>,
}

/// Where a move came from. 10.7% of today's moves were the lichess tablebase and
/// not the engine, and they produce **no telemetry record at all**, so the search
/// panel coasts on a stale position while the board advances. Watching an endgame
/// in the Python dashboard is watching a lie.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum MoveOrigin {
    #[default]
    Unknown,
    Engine,
    /// With `wdl`/`dtz`/`dtm` when the journal gave them: the mate distance the
    /// engine structurally cannot compute.
    Tablebase,
    Book,
    /// The opponent's move.
    Them,
}

#[derive(Clone, Debug)]
pub struct MoveRec {
    pub ply: Ply,
    pub san: String,
    pub uci: String,
    pub origin: MoveOrigin,
    /// Which source we learned this ply from, so a synthesised move list is
    /// visibly synthesised rather than looking authoritative.
    pub authority: PosSource,
    pub clock: Option<Duration>,
    pub grade: Option<Grade>,
}

/// Stockfish's verdict on one move, in lichess's own bands.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Grade {
    Best,
    Good,
    Inaccuracy,
    Mistake,
    Blunder,
}

impl Grade {
    pub const fn mark(self) -> &'static str {
        match self {
            Grade::Best | Grade::Good => "",
            Grade::Inaccuracy => "?!",
            Grade::Mistake => "?",
            Grade::Blunder => "??",
        }
    }
}

#[derive(Clone, Debug)]
pub struct GameState {
    pub id: GameId,
    pub our_colour: shakmaty::Color,
    pub opponent: Opponent,
    pub rated: bool,
    pub speed: String,
    pub time_control: Option<String>,
    /// The one position. Nothing else may decide what is on the board.
    pub resolver: Resolver,
    pub clocks: Clocks,
    pub moves: Vec<MoveRec>,
    /// Win probability **always in White's frame, and displayed that way
    /// everywhere, deliberately** (2026-07-30): positive is good for White,
    /// negative is good for Black, matching every other chess tool. An earlier
    /// design converted this to "our" frame at each display site so a number was
    /// always positive when SumoFish was winning -- reverted because it silently
    /// disagreed with the classical convention every other evaluation display
    /// uses, which read as a wrong number rather than a different one. MCTS
    /// still stores values from each node's own side-to-move perspective
    /// upstream of this, so a series across alternating plies would sawtooth if
    /// stored that way -- this field is where that gets normalized to one fixed
    /// frame (White's), not "ours".
    pub curve: BTreeMap<Ply, f32>,
    /// Stockfish's whole-game curve, same frame.
    pub sf_curve: BTreeMap<Ply, f32>,
    pub search: Option<SearchSnapshot>,
    pub pid: Option<crate::position::Pid>,
    pub finished: Option<GameOutcome>,
}

impl GameState {
    pub fn new(id: GameId, our_colour: shakmaty::Color) -> Self {
        Self {
            resolver: Resolver::new(id.clone()),
            id,
            our_colour,
            opponent: Opponent::default(),
            rated: false,
            speed: String::new(),
            time_control: None,
            clocks: Clocks::default(),
            moves: Vec::new(),
            curve: BTreeMap::new(),
            sf_curve: BTreeMap::new(),
            search: None,
            pid: None,
            finished: None,
        }
    }
    pub fn resolved(&self) -> Option<&Resolved> {
        self.resolver.resolved()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Clocks {
    pub white: Option<Duration>,
    pub black: Option<Duration>,
    /// When the server values were read, so a local tick can run from them
    /// without pretending to be authoritative.
    pub as_of: Option<Instant>,
    pub source: Option<PosSource>,
    pub ticking: Option<shakmaty::Color>,
}

#[derive(Clone, Debug)]
pub struct GameOutcome {
    pub status: String,
    pub winner: Option<shakmaty::Color>,
    pub our_rating_delta: Option<i32>,
}

/// A snapshot of the search, from one telemetry record.
#[derive(Clone, Debug)]
pub struct SearchSnapshot {
    pub ply: Ply,
    pub nodes: u64,
    /// Distinct positions evaluated per second. The Python labelled this `nps`
    /// and put it beside `sims`, whose value is within 15% of it -- exactly the
    /// confusion PHILOSOPHY's "report unique/s, never raw nps" rule exists to
    /// prevent. The number underneath was always honest; the label was not.
    pub unique_per_s: u64,
    pub sims: u64,
    pub elapsed: Duration,
    pub budget: Duration,
    pub wp_white: f32,
    pub best: Option<String>,
    pub pv: Vec<String>,
    pub top: Vec<Candidate>,
    pub mate: bool,
    pub done: bool,
    /// Visits inherited from the previous move's tree by re-rooting. Computed by
    /// `MCTS.report()` and dropped by `snapshot()`, so tree reuse -- a headline
    /// optimisation -- currently has zero observability. Needs one key added
    /// engine-side; `None` until then.
    pub reused: Option<u64>,
    /// How many times the best move changed during this search. Derived here
    /// rather than upstream: it happens in 32% of searches, up to 14 times, and
    /// it is the most watchable fact the 6Hz feed contains.
    pub best_changes: u32,
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub san: String,
    pub visits: u64,
    pub q: f32,
    pub prior: f32,
}

/// A live engine process, whether or not we know whose it is.
#[derive(Clone, Debug)]
pub struct EngineProc {
    pub pid: crate::position::Pid,
    pub game: Option<GameId>,
    pub bot: Option<BotId>,
    pub last_seen: Instant,
    /// From the `boot` record, whose entire payload the Python discarded. Nothing
    /// on screen currently says which checkpoint is playing, and promotion swaps
    /// `runs/value.pt` under a running bot.
    pub boot: Option<EngineBoot>,
}

#[derive(Clone, Debug, Default)]
pub struct EngineBoot {
    /// Which search core is running: `python` or `rust`. Added to the telemetry when
    /// the Rust port was deployed to rated play on 2026-07-29, and as important as
    /// the checkpoint -- the rollback is `CHESSGPU_CORE=python` and a restart, so
    /// "which core is actually playing" is a question you need answered on screen
    /// rather than by reading a unit file.
    pub core: Option<String>,
    pub params: Option<u64>,
    pub policy_step: Option<u64>,
    pub value_step: Option<u64>,
    pub bins: Option<u32>,
    pub batch: Option<u32>,
    pub sims_cap: Option<u64>,
}

#[derive(Clone, Debug, Default)]
pub struct Record {
    pub wins: u32,
    pub draws: u32,
    pub losses: u32,
    /// What the record is scoped to, because a lifetime record averages engines
    /// that no longer exist.
    pub since_version: Option<String>,
}

#[derive(Clone, Debug)]
pub struct FinishedGame {
    pub id: Option<GameId>,
    pub opponent: String,
    pub opponent_rating: Option<i32>,
    /// From the PGN's WhiteTitle/BlackTitle header. lichess bot accounts
    /// always carry the literal title "BOT" -- this is how a human opponent
    /// is told apart from one, which matters because it changes what the
    /// result means (SumoFish is rated against both, but "beat a 2546" reads
    /// very differently depending on which kind of 2546 it was).
    pub opponent_is_bot: bool,
    pub result: String,
    pub reason: String,
    pub our_colour: Option<shakmaty::Color>,
    /// Exactly what this game cost or won. On disk in every PGN as
    /// `WhiteRatingDiff`/`BlackRatingDiff` and read by nothing.
    pub rating_delta: Option<i32>,
    pub eco: Option<String>,
    pub opening: Option<String>,
    pub plies: Option<u32>,
    pub at: Option<jiff::Timestamp>,
}

#[derive(Clone, Debug, Default)]
pub struct Version {
    pub version: String,
    pub title: String,
    pub layman: String,
    pub sha: Option<String>,
    /// The only thing on disk that answers "is the bot playing what this version
    /// shipped?". Read by nothing in the Python.
    pub checkpoint_sha: Option<String>,
    pub step: Option<u64>,
    pub at: Option<jiff::Timestamp>,
}

#[derive(Clone, Debug)]
pub struct RatingSample {
    pub at: jiff::Timestamp,
    pub perfs: IndexMap<String, Perf>,
    /// The fingerprints of what was deployed when this sample was taken. The
    /// literal stated purpose of `logs/rating.jsonl` -- "recorded alongside what
    /// changed, so a jump can be attributed rather than guessed at" -- and the
    /// Python reads only `rating` and drops it.
    pub deployed: Option<Deployed>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Deployed {
    pub engine: Option<String>,
    pub policy: Option<String>,
    pub value: Option<String>,
    pub current: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct WatchdogState {
    pub last_restart: Option<jiff::Timestamp>,
    pub restarts_today: u32,
    pub stuck_games: u32,
}

// ---------------------------------------------------------------- machine

#[derive(Clone, Debug)]
pub struct MachineState {
    pub gpus: Tracked<Vec<Gpu>>,
    pub units: Tracked<Vec<Unit>>,
    pub host: Tracked<Host>,
}

impl Default for MachineState {
    fn default() -> Self {
        let s = Duration::from_secs;
        Self {
            gpus: Tracked::new("gpus", s(2)),
            units: Tracked::new("units", s(6)),
            host: Tracked::new("host", s(30)),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Gpu {
    pub name: String,
    pub util_gpu: u32,
    pub util_mem: u32,
    pub mem_used: u64,
    pub mem_total: u64,
    pub temp_c: u32,
    pub power_w: f32,
    pub power_limit_w: f32,
    pub fan_pct: u32,
    pub clock_sm: u32,
    pub clock_mem: u32,
    pub throttle: Vec<String>,
    /// Per-process VRAM. `machine_panel`'s own docstring asks "is one starving
    /// the other"; this is that answer instead of an inference from one aggregate,
    /// which is the exact mistake CLAUDE.md records as costing a session.
    pub procs: Vec<GpuProc>,
}

#[derive(Clone, Debug)]
pub struct GpuProc {
    pub pid: u32,
    pub mem: Option<u64>,
    pub name: String,
}

#[derive(Clone, Debug)]
pub struct Unit {
    pub name: String,
    /// `loaded` / `not-found` / `masked`. A distinct state, not an error:
    /// `sumofish-train.service` is not-found and renders as a permanently red
    /// dot today, which trains you to ignore the row.
    pub load: String,
    pub active: String,
    pub sub: String,
    pub main_pid: Option<u32>,
    /// systemd's own count. Zero even after a watchdog SIGKILL -- see
    /// `BotState::watchdog`.
    pub n_restarts: u32,
    pub result: String,
    pub since: Option<jiff::Timestamp>,
    /// For timers. CLOCK_MONOTONIC microseconds upstream; converted to a
    /// duration-from-now here so no consumer can mix the two clocks.
    pub next_elapse: Option<Duration>,
    pub last_trigger: Option<jiff::Timestamp>,
}

impl Unit {
    pub fn healthy(&self) -> bool {
        self.load == "loaded" && (self.active == "active" || self.active == "activating")
    }
    pub fn missing(&self) -> bool {
        self.load == "not-found"
    }
}

#[derive(Clone, Debug, Default)]
pub struct Host {
    pub disk_free: Option<u64>,
    pub disk_total: Option<u64>,
    /// Size of the telemetry log against its 16MB rotation threshold. A rotation
    /// and a quiet engine look identical, which is exactly the failure the
    /// track model exists to prevent.
    pub telemetry_bytes: Option<u64>,
}

// ---------------------------------------------------------------- training

#[derive(Clone, Debug)]
pub struct TrainState {
    pub run: Tracked<TrainRun>,
    /// The training unit's own journal, raw -- absorbed from `sumofish train`'s
    /// `journalctl -f`. Global rather than per-bot: there is one training run,
    /// not one per bot watching it.
    pub log: LogLines,
}

impl Default for TrainState {
    fn default() -> Self {
        Self { run: Tracked::new("train", Duration::from_secs(20)), log: LogLines::default() }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TrainRun {
    pub name: String,
    pub step: u64,
    /// From the run's own `config.json`, **not a constant**. The Python hardcoded
    /// `total = 300_000` and a batch of 1024; the live run is 400k steps at batch
    /// 256, so its progress bar reads 13.5% instead of 10.1% and its ETA reads
    /// 76.5h where the arithmetic gives ~26.5h.
    pub total_steps: Option<u64>,
    pub batch_size: Option<u32>,
    pub loss: Option<f64>,
    /// Held-out loss: the metric `best.pt` is actually selected on, and the one
    /// PHILOSOPHY calls honest. Collected by the Python and then dropped, which
    /// left the panel plotting the noisy puzzle accuracy instead.
    pub val_loss: Option<f64>,
    pub puzzle_acc: Option<f64>,
    pub lr: Option<f64>,
    /// The earliest visible sign a 40-hour run is going wrong, hours before
    /// val_loss moves.
    pub grad_norm: Option<f64>,
    pub samples_per_s: Option<f64>,
    pub loss_hist: Vec<f64>,
    pub val_hist: Vec<(u64, f64)>,
    pub puzzle_hist: Vec<(u64, f64)>,
    /// Whether the training PROCESS is alive, from `runs/active.json`'s pid checked
    /// against `/proc`. Deliberately not called `watchdog_alive`: it says nothing
    /// about whether anything is supervising the run, and an earlier version of this
    /// panel conflated the two and printed UNSUPERVISED whenever training simply was
    /// not running.
    pub process_alive: bool,
}

impl TrainRun {
    /// Steps per second, from the sample rate and the run's real batch size.
    pub fn steps_per_s(&self) -> Option<f64> {
        let sps = self.samples_per_s?;
        let batch = self.batch_size? as f64;
        (batch > 0.0).then(|| sps / batch)
    }
    pub fn eta(&self) -> Option<Duration> {
        let total = self.total_steps?;
        let rate = self.steps_per_s()?;
        let left = total.saturating_sub(self.step) as f64;
        (rate > 0.0).then(|| Duration::from_secs_f64(left / rate))
    }
    pub fn progress(&self) -> Option<f64> {
        let total = self.total_steps?;
        (total > 0).then(|| (self.step as f64 / total as f64).clamp(0.0, 1.0))
    }
}

// ---------------------------------------------------------------- lab

#[derive(Clone, Debug)]
pub struct LabState {
    pub queue: Tracked<Lab>,
}

impl Default for LabState {
    fn default() -> Self {
        Self { queue: Tracked::new("lab", Duration::from_secs(30)) }
    }
}

/// The autonomous experiment queue. Nine of fifteen jobs done, 9.3 hours in, a
/// 40-hour training run in flight, and **no dashboard presence whatsoever** --
/// the largest thing happening on the machine is invisible.
#[derive(Clone, Debug, Default)]
pub struct Lab {
    pub jobs: Vec<LabJob>,
    pub current: Option<String>,
    pub started: Option<jiff::Timestamp>,
    pub waiting_since: Option<jiff::Timestamp>,
    /// The derived constants that gate later jobs, e.g. `scale_bar: 265.0` Elo.
    pub facts: BTreeMap<String, String>,
    /// `report.md` is the file `lab.py status` tells you to read, and it is
    /// currently older than `state.json`.
    pub report_stale: bool,
}

#[derive(Clone, Debug)]
pub struct LabJob {
    pub id: String,
    pub what: String,
    pub status: LabStatus,
    pub detail: Option<String>,
    pub seconds: Option<f64>,
    pub summary: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LabStatus {
    Done,
    Running,
    Pending,
    Failed,
}

// ---------------------------------------------------------------- matches

#[derive(Clone, Debug)]
pub struct MatchState {
    pub runs: Tracked<Vec<MatchRun>>,
}

impl Default for MatchState {
    fn default() -> Self {
        Self { runs: Tracked::new("matches", Duration::from_secs(30)) }
    }
}

/// A head-to-head match: a 6-to-20-hour verdict on a code change, arriving one
/// game at a time, with a defined stopping rule.
#[derive(Clone, Debug)]
pub struct MatchRun {
    pub name: String,
    pub a: String,
    pub b: String,
    pub games: u32,
    pub pairs: u32,
    pub wins: u32,
    pub draws: u32,
    pub losses: u32,
    pub elo: Option<f64>,
    /// A number without an interval is not a result. PHILOSOPHY is explicit.
    pub elo_err: Option<f64>,
    pub los: Option<f64>,
    pub llr: Option<f64>,
    pub concluded: bool,
    pub running: bool,
    /// `Σ game.seconds <= job.seconds`. Physically impossible to violate
    /// legitimately, and **all four exchange-rate rungs violate it right now**,
    /// three of them with a null code fingerprint. The cheapest total check in
    /// the repo, per PHILOSOPHY, and nothing checked it.
    pub provenance: Provenance,
    pub code: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Provenance {
    #[default]
    Unknown,
    Ok,
    /// Σ of per-game seconds exceeds the wall clock the job was credited.
    Impossible,
    /// No code fingerprint, so the result cannot be tied to an engine.
    Unfingerprinted,
}

// ---------------------------------------------------------------- api

/// What the lichess budget is doing. On screen, because the whole multi-bot story
/// is that the effective poll interval degrades as bots are added and the user
/// should be able to see that rather than infer it.
#[derive(Clone, Debug, Default)]
pub struct ApiState {
    pub endpoints: IndexMap<String, EndpointState>,
    pub in_flight: Option<String>,
    pub total_last_minute: u32,
    pub throttled: u32,
}

#[derive(Clone, Debug, Default)]
pub struct EndpointState {
    pub calls_last_minute: u32,
    pub rpm_budget: u32,
    /// What the cadence actually came out as after sharing the bucket between
    /// bots, versus what each source asked for.
    pub effective_interval: Option<Duration>,
    pub wanted_interval: Option<Duration>,
    pub last_429: Option<Instant>,
    pub penalty_until: Option<Instant>,
}

// ---------------------------------------------------------------- ui

/// Per-panel view state. Keeping it here rather than inside a panel is what lets
/// `Panel::render` take `&self` and stay a pure function, and what lets any panel
/// be snapshot-tested at any scroll offset without constructing it in a state.
#[derive(Clone, Debug, Default)]
pub struct UiState {
    pub views: BTreeMap<&'static str, PanelView>,
    pub overlay: Overlay,
    pub tier_override: Option<u8>,
    pub paused: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PanelView {
    pub scroll: u16,
    pub expanded: bool,
    pub selected: Option<usize>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Overlay {
    #[default]
    None,
    /// "Where did my machine panel go" -- answerable in one keypress instead of
    /// never.
    Layout,
    Help,
    Sources,
}

// ---------------------------------------------------------------- tape

/// The event log. Bounded, newest last.
#[derive(Clone, Debug)]
pub struct Tape {
    entries: std::collections::VecDeque<TapeEntry>,
    cap: usize,
}

impl Default for Tape {
    fn default() -> Self {
        Self { entries: std::collections::VecDeque::new(), cap: 400 }
    }
}

#[derive(Clone, Debug)]
pub struct TapeEntry {
    pub at: jiff::Timestamp,
    pub kind: TapeKind,
    /// Which bot this concerns, if any. `None` is a machine-level event.
    pub bot: Option<BotId>,
    pub text: String,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TapeKind {
    Move,
    Game,
    Engine,
    Train,
    Lab,
    Warn,
    Error,
}

impl Tape {
    pub fn push(&mut self, e: TapeEntry) {
        if self.entries.len() >= self.cap {
            self.entries.pop_front();
        }
        self.entries.push_back(e);
    }
    pub fn tail(&self, n: usize) -> impl Iterator<Item = &TapeEntry> {
        self.entries.iter().skip(self.entries.len().saturating_sub(n))
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    /// Only entries for a given bot, plus machine-level ones.
    pub fn for_bot<'a>(&'a self, bot: &'a BotId) -> impl Iterator<Item = &'a TapeEntry> {
        self.entries.iter().filter(move |e| e.bot.as_ref().is_none_or(|b| b == bot))
    }
}

/// Convenience for the many places that report a field's freshness.
pub fn worst(tracks: impl IntoIterator<Item = Track>) -> Track {
    tracks.into_iter().max().unwrap_or(Track::Lost)
}

// ---------------------------------------------------------------- raw log

/// The severity a journal line carries. Info covers everything with no explicit
/// level word, which is the common case and never worth flagging on its own.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LogLevel {
    Info,
    Warn,
    Error,
}

/// One raw, unfiltered journal line, as `journalctl -f` would show it -- as
/// opposed to `TapeEntry`, which is a curated event `journal::Parser` recognized.
/// Most of what a process logs matches none of that parser's patterns and would
/// otherwise never appear anywhere in the dashboard at all.
#[derive(Clone, Debug)]
pub struct LogEntry {
    pub at: jiff::Timestamp,
    pub level: LogLevel,
    pub text: String,
}

/// A bounded, newest-last ring of raw log lines. Same shape as `Tape`, kept
/// separate because the two serve different questions: `Tape` is "what
/// happened", this is "what did the process actually print".
#[derive(Clone, Debug)]
pub struct LogLines {
    entries: std::collections::VecDeque<LogEntry>,
    cap: usize,
}

impl Default for LogLines {
    fn default() -> Self {
        Self::new(200)
    }
}

impl LogLines {
    pub fn new(cap: usize) -> Self {
        Self { entries: std::collections::VecDeque::new(), cap }
    }
    pub fn push(&mut self, e: LogEntry) {
        if self.entries.len() >= self.cap {
            self.entries.pop_front();
        }
        self.entries.push_back(e);
    }
    /// The most recent `n` lines, oldest first -- for a scrolling display.
    pub fn tail(&self, n: usize) -> impl Iterator<Item = &LogEntry> {
        self.entries.iter().skip(self.entries.len().saturating_sub(n))
    }
    /// Newest first -- for "what is the latest problem", where scanning from the
    /// front would mean walking the whole buffer on every frame.
    pub fn iter_rev(&self) -> impl Iterator<Item = &LogEntry> {
        self.entries.iter().rev()
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// One glance at everything, for the header. Every `Tracked` field already
/// carries its own freshness; this reduces the whole `AppState` to a count of
/// how many are not `Live` right now, and the worst one among them. Pure
/// composition, no new I/O and no new polling -- it reads exactly the fields
/// every panel already reads.
///
/// A field that has never been fed at all (`Tracked::get` returns `None`) is
/// **not** counted as degraded. `lab.queue` and `matches.runs` sit at `None`
/// for an entire session in which no lab job or match ever ran, and that is a
/// feature not currently in use, not a broken one -- the same distinction
/// `Lab::relevance` and `Matches::relevance` already draw by returning
/// `Relevance::Hidden` rather than complaining. Counting it here would make the
/// header cry wolf on every normal single-bot session.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Health {
    /// How many fed-but-stale fields there are right now.
    pub degraded: u32,
    /// The worst track among them. `Track::Live` (the default) when
    /// `degraded == 0`, so a caller never has to special-case the empty count.
    pub worst: Track,
}

pub fn health(state: &AppState, now: Instant) -> Health {
    let mut h = Health::default();
    let mut note = |t: Track| {
        if t != Track::Live {
            h.degraded += 1;
            h.worst = h.worst.max(t);
        }
    };
    for t in [
        state.machine.gpus.get(now).map(|(_, t)| t),
        state.machine.units.get(now).map(|(_, t)| t),
        state.machine.host.get(now).map(|(_, t)| t),
        state.train.run.get(now).map(|(_, t)| t),
        state.lab.queue.get(now).map(|(_, t)| t),
        state.matches.runs.get(now).map(|(_, t)| t),
    ]
    .into_iter()
    .flatten()
    {
        note(t);
    }
    for b in state.bots.values() {
        for t in [
            b.account.get(now).map(|(_, t)| t),
            b.playing.get(now).map(|(_, t)| t),
            b.record.get(now).map(|(_, t)| t),
            b.results.get(now).map(|(_, t)| t),
            b.version.get(now).map(|(_, t)| t),
            b.rating_log.get(now).map(|(_, t)| t),
            b.watchdog.get(now).map(|(_, t)| t),
        ]
        .into_iter()
        .flatten()
        {
            note(t);
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tape_is_bounded_and_keeps_the_newest() {
        let mut t = Tape::default();
        for i in 0..500 {
            t.push(TapeEntry {
                at: jiff::Timestamp::UNIX_EPOCH,
                kind: TapeKind::Move,
                bot: None,
                text: format!("{i}"),
            });
        }
        assert_eq!(t.len(), 400);
        assert_eq!(t.tail(1).next().unwrap().text, "499");
    }

    #[test]
    fn log_lines_is_bounded_and_iter_rev_finds_the_latest_problem() {
        let mut l = LogLines::new(3);
        for (i, level) in
            [LogLevel::Info, LogLevel::Info, LogLevel::Warn, LogLevel::Info].into_iter().enumerate()
        {
            l.push(LogEntry { at: jiff::Timestamp::UNIX_EPOCH, level, text: format!("{i}") });
        }
        // Bounded to 3, so entry "0" (Info) was evicted.
        assert_eq!(l.len(), 3);
        assert_eq!(l.tail(1).next().unwrap().text, "3");
        let problem = l.iter_rev().find(|e| e.level != LogLevel::Info);
        assert_eq!(problem.unwrap().text, "2", "the warn is still in the buffer");
    }

    /// The training panel's arithmetic, against the run that is live right now:
    /// 400k steps at batch 256 with 968 samples/s. The Python's hardcoded 300k
    /// and /1024 give 13.5% and 76.5h; the truth is ~10% and ~26h.
    #[test]
    fn train_eta_uses_the_runs_own_config() {
        let r = TrainRun {
            step: 40_500,
            total_steps: Some(400_000),
            batch_size: Some(256),
            samples_per_s: Some(968.0),
            ..Default::default()
        };
        let pct = r.progress().unwrap() * 100.0;
        assert!((pct - 10.125).abs() < 0.01, "progress was {pct}");
        let hours = r.eta().unwrap().as_secs_f64() / 3600.0;
        assert!((hours - 26.4).abs() < 0.5, "eta was {hours}h, expected ~26.4");
    }

    #[test]
    fn a_run_without_a_config_reports_no_eta_rather_than_a_wrong_one() {
        let r = TrainRun { step: 40_500, samples_per_s: Some(968.0), ..Default::default() };
        assert!(r.eta().is_none());
        assert!(r.progress().is_none());
    }

    /// An `AppState` nobody has fed anything into is not "degraded" -- it is
    /// the state the dashboard is in for its first second, and `matches`/`lab`
    /// stay unfed for entire sessions that never run either. Every field is
    /// technically `Track::Lost` if you ask `.track()` directly, but `health`
    /// must not count a field that has never been fed at all.
    #[test]
    fn a_freshly_started_state_is_not_degraded() {
        let st = AppState::default();
        let h = health(&st, Instant::now());
        assert_eq!(h.degraded, 0, "an unfed field is unused, not broken");
        assert_eq!(h.worst, Track::Live);
    }

    /// The core contract: one field that was fed and then went stale is
    /// counted, and its track is reported as the worst.
    #[test]
    fn one_stale_bot_field_is_counted_and_reported() {
        let base = Instant::now();
        let cfg = Arc::new(BotConfig {
            id: BotId("sumofish".into()),
            user: "SumoFish".into(),
            label: String::new(),
            unit: None,
            telemetry: "/dev/null".into(),
            pgn_dir: None,
            versions: None,
            rating_log: None,
            watch: true,
        });
        let mut bot = BotState::new(cfg.clone());
        // account's interval is 45s; 6x that is well past Lost.
        bot.account.set(Account::default(), base);
        let mut st = AppState::default();
        st.bots.insert(cfg.id.clone(), bot);

        let fresh = health(&st, base);
        assert_eq!(fresh.degraded, 0, "just fed, should read live");

        let stale = health(&st, base + Duration::from_secs(300));
        assert_eq!(stale.degraded, 1, "one fed-then-abandoned field");
        assert_eq!(stale.worst, Track::Lost);
    }

    /// Two fields going bad at once is two, not one -- and the worse of the
    /// two tracks wins, not the first one seen.
    #[test]
    fn multiple_stale_fields_add_up_and_worst_wins() {
        let base = Instant::now();
        let mut st = AppState::default();
        // train.run: interval 20s. Coast in (40s, 120s], Lost past 120s.
        st.train.run.set(TrainRun::default(), base);
        // machine.host: interval 30s. Coast in (60s, 180s], Lost past 180s.
        st.machine.host.set(Host::default(), base);

        // Both merely coasting: two degraded, worst is Coast.
        let coasting = health(&st, base + Duration::from_secs(70));
        assert_eq!(coasting.degraded, 2);
        assert_eq!(coasting.worst, Track::Coast);

        // Push past train.run's 120s Lost floor but before host's 180s one:
        // only train.run is Lost, host is still Coast. Worst must be Lost even
        // though train.run was declared before host in the composition list,
        // and the count must not double-count either field.
        let mixed = health(&st, base + Duration::from_secs(150));
        assert_eq!(mixed.degraded, 2);
        assert_eq!(mixed.worst, Track::Lost);
    }

    #[test]
    fn units_distinguish_missing_from_broken() {
        let mut u = Unit {
            name: "sumofish-train.service".into(),
            load: "not-found".into(),
            active: "inactive".into(),
            sub: "dead".into(),
            main_pid: None,
            n_restarts: 0,
            result: "success".into(),
            since: None,
            next_elapse: None,
            last_trigger: None,
        };
        assert!(u.missing() && !u.healthy());
        u.load = "loaded".into();
        u.active = "active".into();
        assert!(u.healthy() && !u.missing());
    }
}
