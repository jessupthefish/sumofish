//! Run the parser over the machine's actual journal.
//!
//! The fixtures in `journal.rs` are lines I copied; this is every line the bot has
//! written recently, which is the only way to find out whether the markers hold up
//! against formats I have not seen. Skipped when there is no journal.

use sf_sources::journal::{Event, Parser};
use std::process::Command;

fn journal(unit: &str, since: &str) -> Option<Vec<String>> {
    let out = Command::new("journalctl")
        .args(["--user", "-u", unit, "--since", since, "--no-pager", "-o", "cat"])
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).lines().map(String::from).collect())
}

#[test]
fn the_real_bot_journal_parses() {
    let Some(lines) = journal("sumofish-bot", "-36 hours") else {
        eprintln!("SKIP: no journal");
        return;
    };
    if lines.len() < 100 {
        eprintln!("SKIP: only {} lines", lines.len());
        return;
    }
    let mut p = Parser::default();
    let (mut tb, mut engine, mut over, mut term, mut other) = (0, 0, 0, 0, 0);
    let mut sample = None;
    for l in &lines {
        match p.feed(l) {
            Some(Event::Tablebase { game, uci, wdl, dtz, dtm }) => {
                tb += 1;
                if sample.is_none() {
                    sample = Some(format!("{uci} game={game:?} wdl={wdl:?} dtz={dtz:?} dtm={dtm:?}"));
                }
                // A tablebase move with no move in it is a parse that half worked,
                // which is worse than one that did not.
                assert!(!uci.is_empty(), "a tablebase event with no move");
                assert!(uci.len() >= 4 && uci.len() <= 5, "implausible uci {uci:?}");
            }
            Some(Event::Engine) => engine += 1,
            Some(Event::GameOver(_)) => over += 1,
            Some(Event::EngineTerminated) => term += 1,
            Some(_) => other += 1,
            None => {}
        }
    }
    println!(
        "{} lines: {tb} tablebase, {engine} engine, {over} game-over, {term} terminated, {other} other",
        lines.len()
    );
    if let Some(s) = sample {
        println!("first tablebase move: {s}");
    }
    // The headline number from the survey: roughly one move in ten is not the
    // engine's. If both are zero the markers have moved and the panel would be
    // silently wrong rather than loudly broken.
    assert!(
        tb + engine > 0,
        "neither Source: marker matched in {} lines -- has lichess-bot's logging changed?",
        lines.len()
    );
    if tb + engine > 100 {
        let share = tb as f64 / (tb + engine) as f64;
        println!("tablebase share: {:.1}%", share * 100.0);
        assert!(share < 0.9, "implausible: almost every move attributed to the tablebase");
    }
}

#[test]
fn the_real_watchdog_journal_parses() {
    let Some(lines) = journal("sumofish-watchdog", "-36 hours") else {
        eprintln!("SKIP: no journal");
        return;
    };
    let mut p = Parser::default();
    let (mut restarts, mut stuck) = (0, 0);
    for l in &lines {
        match p.feed(l) {
            Some(Event::WatchdogRestart) => restarts += 1,
            Some(Event::WatchdogStuck { game, seconds }) => {
                stuck += 1;
                println!("stuck: game={game:?} for {seconds:?}s");
            }
            _ => {}
        }
    }
    println!("{} watchdog lines: {restarts} restarts, {stuck} stuck-game reports", lines.len());
    // Not asserted: a healthy day has zero of both, and that is the good case.
}
