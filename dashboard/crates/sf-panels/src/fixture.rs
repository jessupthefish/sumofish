//! A fully-populated `AppState`, for snapshots.
//!
//! Everything here is shaped like the real thing, because a snapshot of an empty
//! panel asserts nothing. The `results` reader shipped broken for a whole session
//! and nobody noticed, precisely because "nothing to show" and "silently found
//! nothing" render identically -- so these fixtures carry data in every field a
//! panel reads.
//!
//! Deterministic by construction: one base `Instant`, one fixed wall clock, no
//! randomness. That is what makes a byte-comparable snapshot possible at all, and it
//! is why `Cx` carries `now` instead of letting panels call `Instant::now`.

use indexmap::IndexMap;
use sf_model::position::{Obs, Pid, parse_fen, parse_uci};
use sf_model::state::*;
use sf_model::{AppState, BotId, GameId, Update, apply};
use shakmaty::Position;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The wall clock every fixture uses. A real-looking moment, frozen.
pub fn wall() -> jiff::Timestamp {
    "2026-07-29T21:14:07Z".parse().expect("a valid timestamp")
}

pub fn bot_id() -> BotId {
    BotId("sumofish".into())
}

pub fn game_id() -> GameId {
    GameId("RsG8Krua".into())
}

fn cfg() -> Arc<BotConfig> {
    Arc::new(BotConfig {
        id: bot_id(),
        user: "SumoFish".into(),
        label: "live v1".into(),
        unit: Some("chess-gpu-bot.service".into()),
        telemetry: "/dev/null".into(),
        pgn_dir: None,
        versions: None,
        rating_log: None,
        watch: true,
    })
}

/// A mid-game state with something in every panel.
pub fn populated(now: Instant) -> AppState {
    let mut st = AppState::default();
    let bot = bot_id();
    let game = game_id();
    apply(&mut st, Update::BotConfigured(cfg()), now);

    // --- account, from a real /api/account shape ---
    let mut perfs = IndexMap::new();
    perfs.insert(
        "rapid".into(),
        Perf { rating: 2365, games: 47, rd: 63, provisional: false, prog: 12 },
    );
    perfs.insert(
        "bullet".into(),
        Perf { rating: 1950, games: 38, rd: 69, provisional: false, prog: -4 },
    );
    apply(
        &mut st,
        Update::Account {
            bot: bot.clone(),
            account: Account {
                username: "SumoFish".into(),
                title: Some("BOT".into()),
                perfs,
                games_total: 134,
            },
        },
        now,
    );
    apply(
        &mut st,
        Update::Record {
            bot: bot.clone(),
            record: Record { wins: 7, draws: 2, losses: 4, since_version: Some("v1".into()) },
        },
        now,
    );

    // --- a game in progress: a real midgame from the bot's own log ---
    const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    let ucis = ["e2e4", "c7c5", "g1f3", "b8c6", "f1b5", "g7g6", "e1g1", "f8g7"];
    let mut pos = parse_fen(START).expect("start position");
    let mut moves = Vec::new();
    for (i, u) in ucis.iter().enumerate() {
        let mv = parse_uci(&pos, u).expect("legal");
        moves.push(MoveRec {
            ply: i as u32,
            san: ["e4", "c5", "Nf3", "Nc6", "Bb5", "g6", "O-O", "Bg7"][i].into(),
            uci: (*u).into(),
            // One tablebase move, so the origin column is exercised.
            origin: if i == 6 { MoveOrigin::Tablebase } else { MoveOrigin::Engine },
            authority: sf_model::PosSource::Stream,
            clock: None,
            // One of each bad grade, so every mark is covered.
            grade: match i {
                3 => Some(Grade::Inaccuracy),
                5 => Some(Grade::Blunder),
                _ => Some(Grade::Good),
            },
        });
        pos = pos.clone().play(mv).expect("legal");
    }
    apply(
        &mut st,
        Update::Playing {
            bot: bot.clone(),
            games: vec![PlayingGame {
                id: game.clone(),
                our_colour: shakmaty::Color::White,
                fen: String::new(),
                last_move: Some("f8g7".into()),
                is_my_turn: true,
                seconds_left: Some(742),
                opponent: Opponent {
                    username: "Ghidorah_X".into(),
                    title: Some("BOT".into()),
                    rating: Some(2288),
                    provisional: false,
                    ai_level: None,
                },
                speed: "rapid".into(),
                rated: true,
            }],
        },
        now,
    );
    apply(
        &mut st,
        Update::Position {
            bot: bot.clone(),
            obs: Obs::EngineSearching { pid: Pid(1234), game: game.clone(), pos, at: now },
        },
        now,
    );
    apply(&mut st, Update::Moves { bot: bot.clone(), game: game.clone(), moves }, now);
    apply(
        &mut st,
        Update::Clocks {
            bot: bot.clone(),
            game: game.clone(),
            clocks: Clocks {
                white: Some(Duration::from_secs(742)),
                black: Some(Duration::from_secs(688)),
                as_of: Some(now),
                source: Some(sf_model::PosSource::Playing),
                ticking: Some(shakmaty::Color::White),
            },
        },
        now,
    );

    // --- the search, mid-think ---
    apply(
        &mut st,
        Update::Search {
            bot: bot.clone(),
            game: game.clone(),
            snapshot: SearchSnapshot {
                ply: 8,
                nodes: 38708,
                unique_per_s: 3324,
                sims: 45120,
                elapsed: Duration::from_millis(21423),
                budget: Duration::from_millis(29800),
                wp_white: 0.5988,
                best: Some("Re1".into()),
                pv: ["Re1", "Nf6", "c3", "O-O", "d4"].iter().map(|s| s.to_string()).collect(),
                top: vec![
                    Candidate { san: "Re1".into(), visits: 13926, q: 0.396, prior: 0.213 },
                    Candidate { san: "c3".into(), visits: 8104, q: 0.371, prior: 0.180 },
                    Candidate { san: "d3".into(), visits: 2291, q: 0.344, prior: 0.096 },
                ],
                mate: false,
                done: false,
                reused: Some(8213),
                best_changes: 2,
            },
        },
        now,
    );
    for (ply, wp) in [(0, 0.51), (2, 0.53), (4, 0.49), (6, 0.55), (8, 0.5988)] {
        apply(
            &mut st,
            Update::Eval { bot: bot.clone(), game: game.clone(), ply, wp_white: wp },
            now,
        );
        apply(
            &mut st,
            Update::StockfishEval {
                bot: bot.clone(),
                game: game.clone(),
                ply,
                wp_white: wp - 0.02,
            },
            now,
        );
    }

    // --- the machine ---
    apply(
        &mut st,
        Update::Gpus(vec![Gpu {
            name: "NVIDIA GeForce RTX 5070 Ti".into(),
            util_gpu: 76,
            util_mem: 24,
            mem_used: 2_312_110_080,
            mem_total: 17_091_723_264,
            temp_c: 71,
            power_w: 214.0,
            power_limit_w: 300.0,
            fan_pct: 42,
            clock_sm: 2752,
            clock_mem: 13801,
            throttle: vec!["SW_POWER_CAP".into()],
            procs: vec![
                GpuProc { pid: 1345757, mem: Some(721_420_288), name: "match.py".into() },
                GpuProc { pid: 591098, mem: Some(580_911_104), name: "kwin_wayland".into() },
                // NVML reports Unavailable for processes it cannot attribute, and
                // showing 0 for one would be a lie. Covered here so the rendering of
                // that case is pinned too.
                GpuProc { pid: 4242, mem: None, name: "unattributable".into() },
            ],
        }]),
        now,
    );
    apply(
        &mut st,
        Update::Units(vec![
            unit("chess-gpu-bot.service", "loaded", "active", "running"),
            unit("chess-gpu-lab.service", "loaded", "inactive", "dead"),
            // The state the Python drew as a permanently red dot.
            unit("chess-gpu-train.service", "not-found", "inactive", "dead"),
            unit("chess-gpu-watchdog.timer", "loaded", "active", "waiting"),
        ]),
        now,
    );
    apply(
        &mut st,
        Update::Watchdog {
            bot: bot.clone(),
            state: WatchdogState {
                last_restart: Some(wall()),
                restarts_today: 3,
                stuck_games: 1,
            },
        },
        now,
    );

    // --- training ---
    apply(
        &mut st,
        Update::Train(TrainRun {
            name: "136M-sv".into(),
            step: 68000,
            total_steps: Some(400_000),
            batch_size: Some(256),
            loss: Some(2.72206),
            val_loss: Some(2.5762),
            puzzle_acc: Some(0.368),
            lr: Some(7.07e-5),
            grad_norm: Some(3.597),
            samples_per_s: Some(1088.0),
            loss_hist: (0..40).map(|i| 2.9 - i as f64 * 0.004).collect(),
            val_hist: (0..8).map(|i| (i * 5000, 2.99 - i as f64 * 0.05)).collect(),
            puzzle_hist: (0..8).map(|i| (i * 5000, 0.30 + i as f64 * 0.01)).collect(),
            process_alive: true,
        }),
        now,
    );

    // --- the lab ---
    apply(
        &mut st,
        Update::LabQueue(Lab {
            jobs: vec![
                job("ablate-causal-1", LabStatus::Done, "step 20,000  loss 2.6517"),
                job("sims-1600-800", LabStatus::Done, "300g  elo +283.8 +-41.3"),
                job("train-136m", LabStatus::Running, "running"),
                job("match-136m", LabStatus::Pending, ""),
            ],
            current: Some("train-136m".into()),
            started: Some(wall()),
            waiting_since: None,
            facts: [("scale_D".to_string(), "237.2".to_string()),
                    ("scale_bar".to_string(), "265.0".to_string())]
                .into_iter()
                .collect(),
            report_stale: true,
        }),
        now,
    );

    // --- matches, including the provenance flag ---
    apply(
        &mut st,
        Update::Matches(vec![
            MatchRun {
                name: "sims-1600-vs-800".into(),
                a: "1600".into(),
                b: "800".into(),
                games: 300,
                pairs: 150,
                wins: 180,
                draws: 40,
                losses: 80,
                elo: Some(283.8),
                elo_err: Some(41.3),
                los: Some(1.0),
                llr: None,
                concluded: true,
                running: false,
                provenance: Provenance::Impossible,
                code: None,
            },
            MatchRun {
                name: "reuse-400".into(),
                a: "on".into(),
                b: "off".into(),
                games: 300,
                pairs: 150,
                wins: 120,
                draws: 100,
                losses: 80,
                elo: Some(21.0),
                elo_err: Some(39.0),
                los: Some(0.85),
                llr: None,
                concluded: false,
                running: true,
                provenance: Provenance::Unfingerprinted,
                code: None,
            },
        ]),
        now,
    );

    // --- version, results, api, tape ---
    apply(
        &mut st,
        Update::VersionInfo {
            bot: bot.clone(),
            version: Version {
                version: "v1".into(),
                title: "the search that thinks ahead".into(),
                layman: String::new(),
                sha: Some("dc7f73c".into()),
                checkpoint_sha: Some("4156a01aacfb".into()),
                step: Some(280_000),
                at: Some(wall()),
            },
        },
        now,
    );
    apply(
        &mut st,
        Update::EngineBooted {
            pid: Pid(1234),
            boot: EngineBoot {
                core: Some("rust".into()),
                params: Some(8_911_024),
                policy_step: Some(280_000),
                value_step: Some(280_000),
                bins: Some(64),
                batch: Some(64),
                sims_cap: Some(100_000_000),
            },
            at: now,
        },
        now,
    );
    apply(
        &mut st,
        Update::EngineSeen {
            pid: Pid(1234),
            game: Some(game.clone()),
            bot: Some(bot.clone()),
            at: now,
        },
        now,
    );
    apply(
        &mut st,
        Update::Results {
            bot: bot.clone(),
            games: vec![
                finished("Z-O_AspenLabs", "win", Some(5), "normal"),
                finished("Ghidorah_X", "none", None, "abandoned"),
                finished("rorschach-bot", "loss", Some(-7), "time forfeit"),
            ],
        },
        now,
    );
    let mut endpoints = IndexMap::new();
    endpoints.insert(
        "account/playing".to_string(),
        EndpointState {
            calls_last_minute: 30,
            rpm_budget: 30,
            effective_interval: Some(Duration::from_secs(2)),
            wanted_interval: Some(Duration::from_secs(2)),
            last_429: None,
            penalty_until: None,
        },
    );
    endpoints.insert(
        "account".to_string(),
        EndpointState {
            calls_last_minute: 1,
            rpm_budget: 6,
            effective_interval: None,
            wanted_interval: Some(Duration::from_secs(10)),
            last_429: None,
            penalty_until: None,
        },
    );
    apply(
        &mut st,
        Update::Api(ApiState {
            endpoints,
            in_flight: None,
            total_last_minute: 31,
            throttled: 0,
        }),
        now,
    );
    for (kind, text) in [
        (TapeKind::Game, "game RsG8Krua vs Ghidorah_X (2288) rapid"),
        (TapeKind::Move, "Re1  wp 0.599  38708n in 21.42s"),
        (TapeKind::Engine, "telemetry log rotated"),
        (TapeKind::Error, "watchdog SIGKILLed and restarted the bot"),
    ] {
        apply(
            &mut st,
            Update::Tape(TapeEntry { at: wall(), kind, bot: Some(bot.clone()), text: text.into() }),
            now,
        );
    }
    st
}

/// The same, with a second bot, for the `bots` panel.
pub fn two_bots(now: Instant) -> AppState {
    let mut st = populated(now);
    let second = Arc::new(BotConfig {
        id: BotId("searchless".into()),
        user: "SumoFishBC".into(),
        label: "searchless control".into(),
        unit: None,
        telemetry: "/dev/null".into(),
        pgn_dir: None,
        versions: None,
        rating_log: None,
        watch: false,
    });
    apply(&mut st, Update::BotConfigured(second), now);
    st
}

/// Nothing has arrived yet: the state a dashboard shows for its first second.
pub fn empty() -> AppState {
    let mut st = AppState::default();
    apply(&mut st, Update::BotConfigured(cfg()), Instant::now());
    st
}

fn unit(name: &str, load: &str, active: &str, sub: &str) -> Unit {
    Unit {
        name: name.into(),
        load: load.into(),
        active: active.into(),
        sub: sub.into(),
        main_pid: Some(1234),
        n_restarts: 0,
        result: "success".into(),
        since: Some(wall()),
        next_elapse: Some(Duration::from_secs(41)),
        last_trigger: Some(wall()),
    }
}

fn job(id: &str, status: LabStatus, detail: &str) -> LabJob {
    LabJob {
        id: id.into(),
        what: detail.into(),
        status,
        detail: (!detail.is_empty()).then(|| detail.to_string()),
        seconds: Some(70.0),
        summary: None,
    }
}

fn finished(opponent: &str, result: &str, delta: Option<i32>, reason: &str) -> FinishedGame {
    FinishedGame {
        id: None,
        opponent: opponent.into(),
        opponent_rating: Some(2100),
        result: result.into(),
        reason: reason.into(),
        our_colour: Some(shakmaty::Color::White),
        rating_delta: delta,
        eco: Some("B45".into()),
        opening: Some("Sicilian Defense".into()),
        plies: Some(64),
        at: Some(wall()),
    }
}

