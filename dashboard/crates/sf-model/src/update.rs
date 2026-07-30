//! `Update` — the only way anything enters `AppState`.
//!
//! `Update` lives in `sf-model` rather than in `sf-sources` for a structural
//! reason: it lets `sf-sources` (which builds them) and `sf-panels` (which reads
//! the result) both name the vocabulary without either crate being able to see
//! the other. In the Python, `panels.py:28` did `from .sources import GATE` and
//! read a source-module global at render time; that is unrepresentable here
//! because `sf-panels` has no dependency on `sf-sources` at all.
//!
//! `apply` is the single funnel, which makes two properties cheap to guarantee:
//! it never panics on any sequence, and a game's ply never decreases.

use crate::position::{GameId, Obs, Pid};
use crate::state::*;
use std::sync::Arc;
use std::time::Instant;

/// One thing a source learned. Every per-bot variant names its bot, because
/// nothing may be attributed by position on a list.
#[derive(Clone, Debug)]
pub enum Update {
    // ---- lichess, per bot ----
    Account { bot: BotId, account: Account },
    AccountFailed { bot: BotId, why: String },
    Playing { bot: BotId, games: Vec<PlayingGame> },
    PlayingFailed { bot: BotId, why: String },
    /// A position observation for a game. The resolver decides what it means.
    Position { bot: BotId, obs: Obs },
    Clocks { bot: BotId, game: GameId, clocks: Clocks },
    Moves { bot: BotId, game: GameId, moves: Vec<MoveRec> },
    GameFinished { bot: BotId, game: GameId, outcome: GameOutcome },
    Results { bot: BotId, games: Vec<FinishedGame> },
    Record { bot: BotId, record: Record },
    VersionInfo { bot: BotId, version: Version },
    RatingLog { bot: BotId, samples: Vec<RatingSample> },
    Watchdog { bot: BotId, state: WatchdogState },

    // ---- engine telemetry ----
    Search { bot: BotId, game: GameId, snapshot: SearchSnapshot },
    /// A White-framed evaluation for one ply. Always White's frame; see
    /// `GameState::curve`.
    Eval { bot: BotId, game: GameId, ply: u32, wp_white: f32 },
    StockfishEval { bot: BotId, game: GameId, ply: u32, wp_white: f32 },
    Grade { bot: BotId, game: GameId, ply: u32, grade: Grade },
    EngineSeen { pid: Pid, game: Option<GameId>, bot: Option<BotId>, at: Instant },
    EngineBooted { pid: Pid, boot: EngineBoot, at: Instant },
    /// A move whose origin is not the engine -- the lichess tablebase supplied
    /// 10.7% of today's moves and they leave no telemetry record at all.
    ///
    /// Keyed by UCI rather than ply, because the journal knows which move was
    /// played and not which half-move number it was. `apply` resolves it.
    MoveOrigin { bot: BotId, game: GameId, uci: String, origin: MoveOrigin },

    // ---- machine, global ----
    Gpus(Vec<Gpu>),
    GpusFailed(String),
    Units(Vec<Unit>),
    UnitsFailed(String),
    HostInfo(Host),
    Train(TrainRun),
    TrainFailed(String),
    LabQueue(Lab),
    LabFailed(String),
    Matches(Vec<MatchRun>),
    Api(ApiState),

    // ---- everything else ----
    Tape(TapeEntry),
    /// A bot appeared in the config. Sent once at startup, and again on reload.
    BotConfigured(Arc<BotConfig>),
}

impl Update {
    /// Which bot this concerns, for routing and for the tape.
    pub fn bot(&self) -> Option<&BotId> {
        use Update::*;
        match self {
            Account { bot, .. }
            | AccountFailed { bot, .. }
            | Playing { bot, .. }
            | PlayingFailed { bot, .. }
            | Position { bot, .. }
            | Clocks { bot, .. }
            | Moves { bot, .. }
            | GameFinished { bot, .. }
            | Results { bot, .. }
            | Record { bot, .. }
            | VersionInfo { bot, .. }
            | RatingLog { bot, .. }
            | Watchdog { bot, .. }
            | Search { bot, .. }
            | Eval { bot, .. }
            | StockfishEval { bot, .. }
            | Grade { bot, .. }
            | MoveOrigin { bot, .. } => Some(bot),
            EngineSeen { bot, .. } => bot.as_ref(),
            _ => None,
        }
    }

    /// Whether applying this could change what the board picture shows. Used to
    /// avoid asking the encoder for a frame that cannot have changed.
    pub fn touches_board(&self) -> bool {
        matches!(self, Update::Position { .. } | Update::Playing { .. })
    }
}

/// What changed, so the caller knows whether to redraw and whether to re-encode
/// the picture. Cheaper and more honest than a dirty flag on the state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Applied {
    pub state_changed: bool,
    pub board_changed: bool,
}

impl Applied {
    fn state() -> Self {
        Applied { state_changed: true, board_changed: false }
    }
    fn board() -> Self {
        Applied { state_changed: true, board_changed: true }
    }
    fn nothing() -> Self {
        Applied::default()
    }
    pub fn merge(self, other: Self) -> Self {
        Applied {
            state_changed: self.state_changed || other.state_changed,
            board_changed: self.board_changed || other.board_changed,
        }
    }
}

/// The one funnel. Never panics; unknown bots and games are created or ignored
/// rather than asserted about, because a source may legitimately be ahead of
/// config reload.
pub fn apply(st: &mut AppState, u: Update, now: Instant) -> Applied {
    use Update::*;
    match u {
        BotConfigured(cfg) => {
            let id = cfg.id.clone();
            if let Some(existing) = st.bots.get_mut(&id) {
                existing.cfg = cfg;
            } else {
                st.bots.insert(id.clone(), BotState::new(cfg));
            }
            if st.focus.is_none() {
                st.focus = Some(id);
            }
            Applied::board()
        }

        Account { bot, account } => with_bot(st, &bot, |b| {
            b.account.set(account, now);
            Applied::state()
        }),
        AccountFailed { bot, why } => with_bot(st, &bot, |b| {
            b.account.fail(why);
            Applied::state()
        }),

        Playing { bot, games } => with_bot(st, &bot, |b| {
            // Stay on the game we are already watching while it is still in the
            // list. lichess orders `nowPlaying` however it likes and the order
            // changes on its own, so treating entry zero as "the game" made the
            // board swap games mid-game and swap back.
            let still_there = b
                .focus_game
                .as_ref()
                .is_some_and(|g| games.iter().any(|p| &p.id == g));
            if !still_there {
                b.focus_game = games.first().map(|p| p.id.clone());
            }
            for p in &games {
                let entry = b
                    .games
                    .entry(p.id.clone())
                    .or_insert_with(|| GameState::new(p.id.clone(), p.our_colour));
                entry.our_colour = p.our_colour;
                entry.opponent = p.opponent.clone();
                entry.rated = p.rated;
                entry.speed = p.speed.clone();
            }
            b.playing.set(games, now);
            Applied::board()
        }),
        PlayingFailed { bot, why } => with_bot(st, &bot, |b| {
            b.playing.fail(why);
            Applied::state()
        }),

        Position { bot, obs } => {
            let game = obs.game().clone();
            with_game(st, &bot, &game, |g| {
                if g.resolver.observe(obs) { Applied::board() } else { Applied::nothing() }
            })
        }

        Clocks { bot, game, clocks } => with_game(st, &bot, &game, |g| {
            g.clocks = clocks;
            Applied::state()
        }),

        Moves { bot, game, moves } => with_game(st, &bot, &game, |g| {
            // The authoritative list replaces a synthesised one, but never
            // shortens it: a torn read must not delete plies we have seen.
            if moves.len() >= g.moves.len() {
                g.moves = moves;
            }
            Applied::state()
        }),

        GameFinished { bot, game, outcome } => with_game(st, &bot, &game, |g| {
            g.finished = Some(outcome);
            Applied::state()
        }),

        Results { bot, games } => with_bot(st, &bot, |b| {
            b.results.set(games, now);
            Applied::state()
        }),
        Record { bot, record } => with_bot(st, &bot, |b| {
            b.record.set(record, now);
            Applied::state()
        }),
        VersionInfo { bot, version } => with_bot(st, &bot, |b| {
            b.version.set(version, now);
            Applied::state()
        }),
        RatingLog { bot, samples } => with_bot(st, &bot, |b| {
            b.rating_log.set(samples, now);
            Applied::state()
        }),
        Watchdog { bot, state } => with_bot(st, &bot, |b| {
            // Two sources feed this: the journal counts restarts, the state file
            // knows the stuck-game list. Keep the larger of each rather than letting
            // whichever arrived last win, or the count flickers between 69 and 0.
            let merged = match b.watchdog.get(now) {
                Some((prev, _)) => WatchdogState {
                    last_restart: state.last_restart.or(prev.last_restart),
                    restarts_today: state.restarts_today.max(prev.restarts_today),
                    stuck_games: state.stuck_games.max(prev.stuck_games),
                },
                None => state,
            };
            b.watchdog.set(merged, now);
            Applied::state()
        }),

        Search { bot, game, snapshot } => with_game(st, &bot, &game, |g| {
            // Count how often the search changed its mind, which happens in 32%
            // of searches and is the most watchable thing the 6Hz feed carries.
            let changes = match (&g.search, &snapshot.best) {
                (Some(prev), Some(best))
                    if prev.ply == snapshot.ply && prev.best.as_deref() != Some(best.as_str()) =>
                {
                    prev.best_changes + 1
                }
                (Some(prev), _) if prev.ply == snapshot.ply => prev.best_changes,
                _ => 0,
            };
            g.search = Some(SearchSnapshot { best_changes: changes, ..snapshot });
            Applied::state()
        }),

        Eval { bot, game, ply, wp_white } => with_game(st, &bot, &game, |g| {
            g.curve.insert(ply, wp_white);
            Applied::state()
        }),
        StockfishEval { bot, game, ply, wp_white } => with_game(st, &bot, &game, |g| {
            g.sf_curve.insert(ply, wp_white);
            Applied::state()
        }),
        Grade { bot, game, ply, grade } => with_game(st, &bot, &game, |g| {
            if let Some(m) = g.moves.iter_mut().find(|m| m.ply == ply) {
                m.grade = Some(grade);
            }
            Applied::state()
        }),
        MoveOrigin { bot, game, uci, origin } => with_game(st, &bot, &game, |g| {
            // Newest first: a move can repeat in a game, and the one just played is
            // the one being described.
            if let Some(m) = g.moves.iter_mut().rev().find(|m| m.uci == uci) {
                m.origin = origin;
                Applied::state()
            } else {
                // The move list has not caught up yet. Dropping is correct; the
                // journal will not repeat itself, but the mark is cosmetic and the
                // tape line already carries the fact.
                Applied::nothing()
            }
        }),

        EngineSeen { pid, game, bot, at } => {
            let e = st.engines.entry(pid).or_insert_with(|| EngineProc {
                pid,
                game: None,
                bot: None,
                last_seen: at,
                boot: None,
            });
            e.last_seen = at;
            // A pid whose game changed is a new partition, not a moved one.
            if game.is_some() {
                e.game = game;
            }
            if bot.is_some() {
                e.bot = bot;
            }
            Applied::state()
        }
        EngineBooted { pid, boot, at } => {
            let e = st.engines.entry(pid).or_insert_with(|| EngineProc {
                pid,
                game: None,
                bot: None,
                last_seen: at,
                boot: None,
            });
            e.boot = Some(boot);
            e.last_seen = at;
            Applied::state()
        }

        Gpus(g) => {
            st.machine.gpus.set(g, now);
            Applied::state()
        }
        GpusFailed(why) => {
            st.machine.gpus.fail(why);
            Applied::state()
        }
        Units(u) => {
            st.machine.units.set(u, now);
            Applied::state()
        }
        UnitsFailed(why) => {
            st.machine.units.fail(why);
            Applied::state()
        }
        HostInfo(h) => {
            st.machine.host.set(h, now);
            Applied::state()
        }
        Train(r) => {
            st.train.run.set(r, now);
            Applied::state()
        }
        TrainFailed(why) => {
            st.train.run.fail(why);
            Applied::state()
        }
        LabQueue(l) => {
            st.lab.queue.set(l, now);
            Applied::state()
        }
        LabFailed(why) => {
            st.lab.queue.fail(why);
            Applied::state()
        }
        Matches(m) => {
            st.matches.runs.set(m, now);
            Applied::state()
        }
        Api(a) => {
            st.api = a;
            Applied::state()
        }
        Tape(e) => {
            st.tape.push(e);
            Applied::state()
        }
    }
}

fn with_bot(
    st: &mut AppState,
    bot: &BotId,
    f: impl FnOnce(&mut BotState) -> Applied,
) -> Applied {
    match st.bots.get_mut(bot) {
        Some(b) => f(b),
        // A source may be a beat ahead of config reload. Dropping is correct and
        // silent panics are not.
        None => Applied::nothing(),
    }
}

fn with_game(
    st: &mut AppState,
    bot: &BotId,
    game: &GameId,
    f: impl FnOnce(&mut GameState) -> Applied,
) -> Applied {
    let Some(b) = st.bots.get_mut(bot) else { return Applied::nothing() };
    // A game we have not been told about yet still gets a slot: telemetry is
    // faster than `/api/account/playing`, which is the whole point of the
    // ranking in `position::Source`. Colour is corrected when `playing` lands.
    let g = b
        .games
        .entry(game.clone())
        .or_insert_with(|| GameState::new(game.clone(), shakmaty::Color::White));
    f(g)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::{parse_fen, parse_uci};
    use shakmaty::Position;
    use std::path::PathBuf;

    const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    fn cfg(id: &str) -> Arc<BotConfig> {
        Arc::new(BotConfig {
            id: BotId(id.into()),
            user: format!("{id}Bot"),
            label: id.into(),
            unit: None,
            telemetry: PathBuf::from("/dev/null"),
            pgn_dir: None,
            versions: None,
            rating_log: None,
            watch: true,
        })
    }

    fn seeded() -> (AppState, Instant) {
        let mut st = AppState::default();
        let now = Instant::now();
        apply(&mut st, Update::BotConfigured(cfg("a")), now);
        (st, now)
    }

    #[test]
    fn the_first_bot_configured_takes_focus() {
        let (st, _) = seeded();
        assert_eq!(st.focus, Some(BotId("a".into())));
        // A second bot must not steal it.
        let mut st = st;
        apply(&mut st, Update::BotConfigured(cfg("b")), Instant::now());
        assert_eq!(st.focus, Some(BotId("a".into())));
        assert_eq!(st.bots.len(), 2);
        // Config order, not hash order.
        let ids: Vec<_> = st.bots.keys().map(|k| k.0.clone()).collect();
        assert_eq!(ids, vec!["a", "b"]);
    }

    /// The Lab Note: `nowPlaying[0]` is not "the game we are watching". lichess
    /// reorders that list on its own, so the board could swap games and swap back.
    #[test]
    fn focus_stays_on_a_game_while_it_is_still_in_the_list() {
        let (mut st, now) = seeded();
        let g1 = GameId("g1".into());
        let g2 = GameId("g2".into());
        let mk = |id: &GameId| PlayingGame {
            id: id.clone(),
            our_colour: shakmaty::Color::White,
            fen: START.into(),
            last_move: None,
            is_my_turn: true,
            seconds_left: None,
            opponent: Opponent::default(),
            speed: "rapid".into(),
            rated: true,
        };
        apply(
            &mut st,
            Update::Playing { bot: BotId("a".into()), games: vec![mk(&g1), mk(&g2)] },
            now,
        );
        assert_eq!(st.bots["a"].focus_game, Some(g1.clone()));

        // lichess reorders. We must not follow.
        apply(
            &mut st,
            Update::Playing { bot: BotId("a".into()), games: vec![mk(&g2), mk(&g1)] },
            now,
        );
        assert_eq!(st.bots["a"].focus_game, Some(g1.clone()), "must not swap games");

        // Only when it genuinely leaves the list do we move.
        apply(&mut st, Update::Playing { bot: BotId("a".into()), games: vec![mk(&g2)] }, now);
        assert_eq!(st.bots["a"].focus_game, Some(g2));
    }

    #[test]
    fn a_position_for_an_unknown_game_creates_it_rather_than_dropping_it() {
        let (mut st, now) = seeded();
        let game = GameId("fresh".into());
        let a = apply(
            &mut st,
            Update::Position {
                bot: BotId("a".into()),
                obs: Obs::EngineSearching {
                    pid: Pid(9),
                    game: game.clone(),
                    pos: parse_fen(START).unwrap(),
                    at: now,
                },
            },
            now,
        );
        assert!(a.board_changed);
        assert!(st.bots["a"].games.contains_key(&game));
    }

    #[test]
    fn updates_for_an_unknown_bot_are_dropped_not_panicked() {
        let (mut st, now) = seeded();
        let a = apply(
            &mut st,
            Update::AccountFailed { bot: BotId("nope".into()), why: "x".into() },
            now,
        );
        assert_eq!(a, Applied::default());
    }

    #[test]
    fn best_move_changes_are_counted_within_a_ply_and_reset_across_plies() {
        let (mut st, now) = seeded();
        let bot = BotId("a".into());
        let game = GameId("g".into());
        let snap = |ply: u32, best: &str| SearchSnapshot {
            ply,
            nodes: 0,
            unique_per_s: 0,
            sims: 0,
            elapsed: std::time::Duration::ZERO,
            budget: std::time::Duration::ZERO,
            wp_white: 0.5,
            best: Some(best.into()),
            pv: vec![],
            top: vec![],
            mate: false,
            done: false,
            reused: None,
            best_changes: 0,
        };
        for (ply, best) in [(4, "e4"), (4, "e4"), (4, "d4"), (4, "c4"), (6, "Nf3")] {
            apply(
                &mut st,
                Update::Search { bot: bot.clone(), game: game.clone(), snapshot: snap(ply, best) },
                now,
            );
        }
        let s = st.bots["a"].games[&game].search.as_ref().unwrap();
        assert_eq!(s.ply, 6);
        assert_eq!(s.best_changes, 0, "a new ply starts a fresh count");

        // And within one ply it accumulates.
        let mut st2 = seeded().0;
        for best in ["e4", "e4", "d4", "c4"] {
            apply(
                &mut st2,
                Update::Search { bot: bot.clone(), game: game.clone(), snapshot: snap(4, best) },
                now,
            );
        }
        assert_eq!(st2.bots["a"].games[&game].search.as_ref().unwrap().best_changes, 2);
    }

    /// A torn move list must not delete plies already seen.
    #[test]
    fn a_shorter_move_list_never_truncates_a_longer_one() {
        let (mut st, now) = seeded();
        let bot = BotId("a".into());
        let game = GameId("g".into());
        let mv = |ply: u32| MoveRec {
            ply,
            san: format!("m{ply}"),
            uci: "e2e4".into(),
            origin: MoveOrigin::Unknown,
            authority: crate::position::Source::Stream,
            clock: None,
            grade: None,
        };
        apply(
            &mut st,
            Update::Moves {
                bot: bot.clone(),
                game: game.clone(),
                moves: (0..10).map(mv).collect(),
            },
            now,
        );
        apply(
            &mut st,
            Update::Moves { bot, game: game.clone(), moves: (0..3).map(mv).collect() },
            now,
        );
        assert_eq!(st.bots["a"].games[&game].moves.len(), 10);
    }

    /// The property that matters for `apply`: no sequence of updates, however
    /// silly, may panic or lower a game's ply. Deterministic pseudo-random so a
    /// failure reproduces without a seed to record.
    #[test]
    fn apply_never_panics_and_ply_never_decreases() {
        let (mut st, now) = seeded();
        let bot = BotId("a".into());
        let game = GameId("g".into());
        let ucis = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"];

        // A cheap deterministic generator: a linear congruential walk over a menu
        // of updates, including out-of-order and duplicate positions.
        let mut seed: u64 = 0x5eed;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as usize
        };

        let mut high_water = 0u32;
        for _ in 0..4000 {
            let n = next();
            let depth = n % (ucis.len() + 1);
            let mut pos = parse_fen(START).unwrap();
            for u in &ucis[..depth] {
                let m = parse_uci(&pos, u).unwrap();
                pos = pos.play(m).unwrap();
            }
            let upd = match n % 5 {
                0 => Update::Position {
                    bot: bot.clone(),
                    obs: Obs::EngineSearching { pid: Pid(1), game: game.clone(), pos, at: now },
                },
                1 => Update::Position {
                    bot: bot.clone(),
                    obs: Obs::Stream { game: game.clone(), pos, last: None, at: now },
                },
                2 => Update::Eval {
                    bot: bot.clone(),
                    game: game.clone(),
                    ply: depth as u32,
                    wp_white: (n % 100) as f32 / 100.0,
                },
                3 => Update::Position {
                    bot: bot.clone(),
                    obs: Obs::Playing { game: game.clone(), pos, last: None, at: now },
                },
                _ => Update::Tape(TapeEntry {
                    at: jiff::Timestamp::UNIX_EPOCH,
                    kind: TapeKind::Move,
                    bot: Some(bot.clone()),
                    text: "x".into(),
                }),
            };
            apply(&mut st, upd, now);
            if let Some(r) = st.bots["a"].games.get(&game).and_then(|g| g.resolved()) {
                assert!(r.ply >= high_water, "ply went backwards: {} < {}", r.ply, high_water);
                high_water = r.ply;
            }
        }
        assert!(high_water > 0, "the walk should have advanced the board at least once");
    }
}
