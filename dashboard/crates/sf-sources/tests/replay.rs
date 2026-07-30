//! Replay the real telemetry log through the real resolver.
//!
//! `logs/engine.jsonl` and its rotated sibling hold megabytes of genuine
//! two-game-interleaved records from a bot that has been playing for days. That is
//! a far better fixture than anything hand-written, and it is exactly the data the
//! Python's demultiplexer was built for: 468 places in one log where consecutive
//! searches alternate between two games.
//!
//! Skipped when the log is absent, so this is not a test that only passes on one
//! machine.

use sf_model::state::TapeKind;
use sf_model::{AppState, Update, apply};
use sf_sources::TelemetrySource;
use sf_sources::tailer::Batch;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

const LOGS: &[&str] =
    &["/home/nomad/dev/active/chess-gpu/logs/engine.jsonl", "/home/nomad/dev/active/chess-gpu/logs/engine.jsonl.1"];

fn load() -> Option<(String, Vec<String>)> {
    for p in LOGS {
        if let Ok(text) = std::fs::read_to_string(p) {
            let lines: Vec<String> = text.lines().map(String::from).collect();
            if lines.len() > 100 {
                return Some((p.to_string(), lines));
            }
        }
    }
    None
}

fn seeded(bot: &sf_model::BotId) -> AppState {
    let mut state = AppState::default();
    apply(
        &mut state,
        Update::BotConfigured(Arc::new(sf_model::BotConfig {
            id: bot.clone(),
            user: "SumoFish".into(),
            label: "replay".into(),
            unit: None,
            telemetry: "/dev/null".into(),
            pgn_dir: None,
            versions: None,
            rating_log: None,
            watch: true,
        })),
        Instant::now(),
    );
    state
}

fn batch(lines: Vec<String>) -> Batch {
    Batch { lines, rotated: false, truncated: false, size: 0 }
}

/// The properties that matter, over real data.
#[test]
fn the_real_log_replays_cleanly() {
    let Some((path, lines)) = load() else {
        eprintln!("SKIP: no telemetry log with enough records");
        return;
    };
    let bot = sf_model::BotId("replay".into());
    let mut src = TelemetrySource::new("/dev/null").adopt_all(bot.clone());
    let mut state = seeded(&bot);

    let now = Instant::now();
    let mut high_water: HashMap<String, u32> = HashMap::new();
    let mut chose = 0usize;
    let mut searched = 0usize;
    // Realistic chunks rather than one giant batch, so batching is exercised.
    for chunk in lines.chunks(64) {
        for u in src.absorb(batch(chunk.to_vec()), now) {
            if let Update::Position { obs, .. } = &u {
                match obs {
                    sf_model::Obs::EngineChose { .. } => chose += 1,
                    sf_model::Obs::EngineSearching { .. } => searched += 1,
                    _ => {}
                }
            }
            apply(&mut state, u, now);
        }
        // PLY NEVER REGRESSES, per game, ever.
        for (id, game) in &state.bots[&bot].games {
            if let Some(r) = game.resolved() {
                let hw = high_water.entry(id.0.clone()).or_insert(0);
                assert!(r.ply >= *hw, "game {id}: ply went backwards, {} < {hw}", r.ply);
                *hw = r.ply;
            }
        }
    }

    let b = &state.bots[&bot];
    println!(
        "replayed {} lines from {path}: {} games, {searched} searching, {chose} chose, \
         {} unattributed, {} parse errors, {} boots",
        lines.len(),
        b.games.len(),
        src.stats.unattributed,
        src.stats.parse_errors,
        src.stats.boots,
    );

    assert!(src.stats.records > 100, "the log should have produced records");
    assert!(!b.games.is_empty(), "the log should have produced at least one game");
    assert!(searched > 0, "the log should have produced searching observations");
    // THE FIX, against real data. The log contains `move` records with done+uci, and
    // every one must yield an EngineChose. Without it the board sat on the pre-move
    // position for ~9 seconds after every one of our own moves.
    assert!(chose > 0, "no EngineChose from a real log -- fusion.py:180 is back");

    // Cross-game contamination is impossible by construction; assert it anyway,
    // because it is the failure that put two games on one evaluation curve and
    // produced "a graph that says one side is winning when it is not".
    for (id, game) in &b.games {
        assert_eq!(game.resolver.foreign, 0, "game {id} was offered another game's records");
    }

    let rate = src.stats.parse_errors as f64 / src.stats.records as f64;
    assert!(rate < 0.01, "parse error rate {rate:.4} -- has the record schema changed?");
}

/// Every game the log mentions must end up with a board, or attribution is silently
/// dropping records.
#[test]
fn every_game_in_the_log_gets_a_board() {
    let Some((_, lines)) = load() else {
        eprintln!("SKIP: no telemetry log");
        return;
    };
    let mut ids: Vec<String> = lines
        .iter()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter_map(|v: serde_json::Value| v.get("game")?.as_str().map(String::from))
        .collect();
    ids.sort_unstable();
    ids.dedup();
    if ids.is_empty() {
        eprintln!("SKIP: the log carries no game ids");
        return;
    }

    let bot = sf_model::BotId("replay".into());
    let mut src = TelemetrySource::new("/dev/null").adopt_all(bot.clone());
    let mut state = seeded(&bot);
    let now = Instant::now();
    for u in src.absorb(batch(lines), now) {
        apply(&mut state, u, now);
    }

    let b = &state.bots[&bot];
    for id in &ids {
        let g = b
            .games
            .get(&sf_model::GameId(id.clone()))
            .unwrap_or_else(|| panic!("game {id} appears in the log but got no state"));
        assert!(g.resolved().is_some(), "game {id} never resolved a position");
    }
    println!("{} distinct games, every one with a board", ids.len());
}

/// The resolved state must not depend on how records happen to be batched. A
/// different chunk size is a different arrival order at the `apply` boundary.
#[test]
fn batching_does_not_change_the_outcome() {
    let Some((_, lines)) = load() else {
        eprintln!("SKIP: no telemetry log");
        return;
    };
    let sample: Vec<String> = lines.iter().take(4000).cloned().collect();
    let bot = sf_model::BotId("replay".into());

    let run = |chunk: usize| -> Vec<(String, u32)> {
        let mut src = TelemetrySource::new("/dev/null").adopt_all(bot.clone());
        let mut state = seeded(&bot);
        let now = Instant::now();
        for c in sample.chunks(chunk) {
            for u in src.absorb(batch(c.to_vec()), now) {
                apply(&mut state, u, now);
            }
        }
        let mut out: Vec<(String, u32)> = state.bots[&bot]
            .games
            .iter()
            .filter_map(|(id, g)| g.resolved().map(|r| (id.0.clone(), r.ply)))
            .collect();
        out.sort();
        out
    };

    let one = run(1);
    for chunk in [7usize, 64, 1000, 100_000] {
        assert_eq!(run(chunk), one, "chunk size {chunk} produced a different final state");
    }
}

/// A rotation must announce itself rather than looking like a quiet engine.
#[test]
fn a_rotation_is_reported() {
    let mut src = TelemetrySource::new("/dev/null");
    let ups =
        src.absorb(Batch { rotated: true, ..batch(vec![]) }, Instant::now());
    assert!(ups.iter().any(|u| matches!(u, Update::Tape(e) if e.kind == TapeKind::Engine)));
}
