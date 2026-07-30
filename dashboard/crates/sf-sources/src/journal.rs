//! What lichess-bot says about itself.
//!
//! This is the one source that still shells out, deliberately: there is no
//! trustworthy Rust reader for the journal's binary format, and `journalctl -o json`
//! is a documented stable interface in a way the on-disk format is not. It is one
//! long-lived pipe spawned once, not a poll, so the usual objection does not apply.
//!
//! # Why it matters more than it looks
//!
//! **10.7% of the bot's moves are not the engine's.** `online_egtb` is enabled at
//! `max_pieces: 7`, and on one day the journal recorded 349 tablebase moves against
//! 2,916 engine ones. Those moves produce **no telemetry record at all** — so the
//! search panel silently coasts on the last position it searched while the board
//! advances underneath it. Watching an endgame in the Python dashboard is watching a
//! lie, and there is nothing on screen that could tell you.
//!
//! The tablebase is also computing the thing the engine structurally cannot: `dtm`
//! is a mate distance, and PHILOSOPHY is explicit that mating requires calculation
//! this engine does not have. Attributing that to the engine flatters it.
//!
//! # Parsing a log that wraps
//!
//! lichess-bot logs through `rich`, which hard-wraps, so one logical message arrives
//! as several journal entries:
//!
//! ```text
//! INFO  Got move f3f2 from           engine_wrapper.py:1098
//!       tablebase.lichess.ovh (wdl:
//!       2, dtz: 2, dtm: 8) for game
//!       0cO0mhHX
//! INFO  Source: Lichess EGTB         engine_wrapper.py:334
//! ```
//!
//! So the reader keeps a small rolling window and reassembles. `Source: Lichess
//! EGTB` is the reliable single-line marker; the detail is behind it.

use anyhow::Result;
use serde::Deserialize;
use sf_model::state::{MoveOrigin, TapeEntry, TapeKind};
use sf_model::{BotId, GameId, Update};
use std::collections::VecDeque;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;

#[derive(Debug, Deserialize)]
struct JournalLine {
    #[serde(rename = "MESSAGE")]
    message: Option<serde_json::Value>,
    #[serde(rename = "__CURSOR")]
    cursor: Option<String>,
    #[serde(rename = "__REALTIME_TIMESTAMP")]
    realtime: Option<String>,
}

impl JournalLine {
    /// `MESSAGE` is a string, or an array of byte values when it is not valid UTF-8.
    fn text(&self) -> String {
        match &self.message {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(serde_json::Value::Array(a)) => {
                let bytes: Vec<u8> = a.iter().filter_map(|v| v.as_u64().map(|n| n as u8)).collect();
                String::from_utf8_lossy(&bytes).into_owned()
            }
            _ => String::new(),
        }
    }
    fn at(&self) -> jiff::Timestamp {
        self.realtime
            .as_deref()
            .and_then(|s| s.parse::<i64>().ok())
            .and_then(|us| jiff::Timestamp::from_microsecond(us).ok())
            .unwrap_or(jiff::Timestamp::UNIX_EPOCH)
    }
}

/// What one reassembled message told us.
#[derive(Debug, PartialEq)]
pub enum Event {
    /// A move the tablebase supplied, not the engine.
    Tablebase { game: Option<GameId>, uci: String, wdl: Option<i32>, dtz: Option<i32>, dtm: Option<i32> },
    /// A move the engine chose. Only interesting as the complement of the above.
    Engine,
    GameOver(Option<GameId>),
    Declined(String),
    Aborted(Option<GameId>),
    /// lichess-bot lost its engine. Fired 21 times in one day.
    EngineTerminated,
    /// The watchdog SIGKILLed the bot. This is the count the machine panel needs,
    /// because systemd's `NRestarts` stays at zero through a `systemctl kill`.
    WatchdogRestart,
    WatchdogStuck { game: Option<GameId>, seconds: Option<u32> },
}

/// Reassembles wrapped messages and classifies them.
pub struct Parser {
    window: VecDeque<String>,
}

impl Default for Parser {
    fn default() -> Self {
        Parser { window: VecDeque::with_capacity(8) }
    }
}

impl Parser {
    /// Feed one journal message. Returns an event when this line completes one.
    pub fn feed(&mut self, raw: &str) -> Option<Event> {
        let line = strip_rich(raw);
        if self.window.len() == 8 {
            self.window.pop_front();
        }
        self.window.push_back(line.clone());

        // The markers, checked before the expensive reassembly.
        if line.contains("Source: Lichess EGTB") {
            return Some(self.tablebase_detail());
        }
        if line.contains("Source: Engine") {
            return Some(Event::Engine);
        }
        if line.contains("[watchdog]") {
            if line.contains("restarting") {
                return Some(Event::WatchdogRestart);
            }
            if let Some(rest) = line.split("game ").nth(1) {
                if line.contains("our turn for") {
                    let game = rest.split(':').next().map(|g| GameId(g.trim().to_string()));
                    let seconds = line
                        .split("our turn for ")
                        .nth(1)
                        .and_then(|s| s.trim_end_matches('s').trim().parse().ok());
                    return Some(Event::WatchdogStuck { game, seconds });
                }
            }
        }
        if line.contains("EngineTerminatedError") || line.contains("TerminatedError") {
            return Some(Event::EngineTerminated);
        }
        if line.contains("Game over") {
            return Some(Event::GameOver(last_game_id(&self.window)));
        }
        if line.contains("Aborting") || line.contains("aborted") {
            return Some(Event::Aborted(last_game_id(&self.window)));
        }
        if line.contains("declined") || line.contains("Decline") {
            return Some(Event::Declined(squash(&line)));
        }
        None
    }

    /// Reassemble the wrapped `Got move ... from tablebase... for game ...` that
    /// precedes the `Source:` marker.
    fn tablebase_detail(&self) -> Event {
        let joined = squash(&self.window.iter().cloned().collect::<Vec<_>>().join(" "));
        // Take the LAST occurrence: the window may hold two moves' worth.
        let tail = joined.rsplit_once("Got move ").map(|(_, t)| t).unwrap_or(&joined);
        let uci = tail.split_whitespace().next().unwrap_or_default().to_string();
        Event::Tablebase {
            game: after(tail, "for game ").map(|g| GameId(g.to_string())),
            uci,
            wdl: after(tail, "wdl: ").and_then(parse_num),
            dtz: after(tail, "dtz: ").and_then(parse_num),
            dtm: after(tail, "dtm: ").and_then(parse_num),
        }
    }
}

/// The token after `key`, up to the next separator.
fn after<'a>(hay: &'a str, key: &str) -> Option<&'a str> {
    let rest = hay.split_once(key)?.1;
    let end = rest
        .find([',', ')', ' ', '\n'])
        .unwrap_or(rest.len());
    let tok = rest[..end].trim();
    (!tok.is_empty()).then_some(tok)
}

fn parse_num(s: &str) -> Option<i32> {
    s.trim_matches(|c: char| !c.is_ascii_digit() && c != '-').parse().ok()
}

/// Strip `rich`'s timestamp, level and source-file columns, which are padding rather
/// than content and which break naive matching.
fn strip_rich(s: &str) -> String {
    let mut out = s.to_string();
    // A leading "[07/28/26 18:53:05] " stamp -- and ONLY that. An earlier version
    // stripped any leading bracket, which ate the "[watchdog] " prefix that every
    // watchdog event is identified by, so restarts and stuck games both went
    // unnoticed.
    if out.starts_with('[') {
        if let Some(i) = out.find("] ") {
            let inner = &out[1..i];
            if inner.contains('/') && inner.contains(':') {
                out = out[i + 2..].to_string();
            }
        }
    }
    // A leading level word.
    for level in ["INFO ", "WARNING ", "ERROR ", "DEBUG ", "CRITICAL "] {
        let t = out.trim_start();
        if let Some(rest) = t.strip_prefix(level) {
            out = rest.to_string();
            break;
        }
    }
    // A trailing "engine_wrapper.py:334" source column.
    if let Some(i) = out.rfind("  ") {
        let tail = out[i..].trim();
        if tail.ends_with(|c: char| c.is_ascii_digit()) && tail.contains(".py:") {
            out.truncate(i);
        }
    }
    out.trim().to_string()
}

fn squash(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn last_game_id(window: &VecDeque<String>) -> Option<GameId> {
    window
        .iter()
        .rev()
        .find_map(|l| after(l, "for game ").or_else(|| after(l, "game ")))
        .map(|g| GameId(g.to_string()))
}

/// Follows one unit's journal.
pub struct JournalSource {
    unit: String,
    bot: BotId,
    child: Option<Child>,
}

impl JournalSource {
    pub fn new(unit: impl Into<String>, bot: BotId) -> Self {
        JournalSource { unit: unit.into(), bot, child: None }
    }

    /// Follow, forever, sending updates. Returns when the pipe closes or the sender
    /// is dropped.
    ///
    /// `--since` on attach rather than starting at the end: the same reasoning as the
    /// telemetry backfill. A follower with no history shows an empty panel and looks
    /// frozen, which is indistinguishable from actually frozen.
    pub async fn run(mut self, tx: mpsc::Sender<Update>, cursor: Option<String>) -> Result<()> {
        let mut cmd = Command::new("journalctl");
        cmd.args(["--user", "-u", &self.unit, "-f", "-o", "json", "--no-pager"]);
        match cursor.as_deref() {
            // A supervised restart resumes exactly where it stopped rather than
            // losing or duplicating the gap.
            Some(c) => {
                cmd.arg("--after-cursor").arg(c);
            }
            None => {
                cmd.arg("--since").arg("-30 min");
            }
        }
        cmd.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::null());
        // Or a journalctl leaks per dashboard restart.
        cmd.kill_on_drop(true);

        let mut child = cmd.spawn()?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow::anyhow!("no stdout"))?;
        self.child = Some(child);

        let mut lines = BufReader::new(stdout).lines();
        let mut parser = Parser::default();
        let mut last_uci: Option<String> = None;
        // The journal is the ONLY honest restart count: `watchdog.py` uses
        // `systemctl kill` + `restart`, so systemd's NRestarts never moves and the
        // state file records only the most recent one.
        let mut watchdog = sf_model::state::WatchdogState::default();

        while let Some(line) = lines.next_line().await? {
            let Ok(j) = serde_json::from_str::<JournalLine>(&line) else { continue };
            let text = j.text();
            if text.is_empty() {
                continue;
            }
            let Some(event) = parser.feed(&text) else { continue };
            match &event {
                Event::WatchdogRestart => {
                    watchdog.restarts_today += 1;
                    watchdog.last_restart = Some(j.at());
                }
                Event::WatchdogStuck { .. } => watchdog.stuck_games += 1,
                _ => {}
            }
            let counted = matches!(
                event,
                Event::WatchdogRestart | Event::WatchdogStuck { .. }
            );
            for u in self.to_updates(event, j.at(), &mut last_uci) {
                if tx.send(u).await.is_err() {
                    return Ok(());
                }
            }
            if counted {
                let u = Update::Watchdog { bot: self.bot.clone(), state: watchdog.clone() };
                if tx.send(u).await.is_err() {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    fn to_updates(
        &self,
        e: Event,
        at: jiff::Timestamp,
        last_uci: &mut Option<String>,
    ) -> Vec<Update> {
        let tape = |kind, text: String| {
            Update::Tape(TapeEntry { at, kind, bot: Some(self.bot.clone()), text })
        };
        match e {
            Event::Tablebase { game, uci, wdl, dtz, dtm } => {
                *last_uci = Some(uci.clone());
                let mut v = Vec::new();
                if let Some(g) = game {
                    v.push(Update::MoveOrigin {
                        bot: self.bot.clone(),
                        game: g,
                        uci: uci.clone(),
                        origin: MoveOrigin::Tablebase,
                    });
                }
                let detail = match (wdl, dtz, dtm) {
                    (Some(w), Some(z), Some(m)) => format!(" wdl {w} dtz {z} dtm {m}"),
                    _ => String::new(),
                };
                // Worth a tape line as well as the mark: it explains why the search
                // panel did not move for this ply.
                v.push(tape(TapeKind::Game, format!("{uci} from the tablebase,{detail}")));
                v
            }
            Event::Engine => vec![],
            Event::GameOver(g) => vec![tape(
                TapeKind::Game,
                format!("game {} over", g.map(|g| g.0).unwrap_or_else(|| "?".into())),
            )],
            Event::Aborted(g) => vec![tape(
                TapeKind::Warn,
                format!("game {} aborted", g.map(|g| g.0).unwrap_or_else(|| "?".into())),
            )],
            Event::Declined(what) => vec![tape(TapeKind::Game, what)],
            Event::EngineTerminated => {
                vec![tape(TapeKind::Error, "engine terminated under lichess-bot".into())]
            }
            Event::WatchdogRestart => {
                vec![tape(TapeKind::Error, "watchdog SIGKILLed and restarted the bot".into())]
            }
            Event::WatchdogStuck { game, seconds } => vec![tape(
                TapeKind::Warn,
                format!(
                    "game {} stuck: our turn for {}s",
                    game.map(|g| g.0).unwrap_or_else(|| "?".into()),
                    seconds.unwrap_or(0)
                ),
            )],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact lines from the real journal, wrapping and all.
    const EGTB: &[&str] = &[
        "[07/28/26 18:53:05] INFO     Got move f3f2 from           engine_wrapper.py:1098",
        "                             tablebase.lichess.ovh (wdl:",
        "                             2, dtz: 2, dtm: 8) for game",
        "                             0cO0mhHX",
        "                    INFO     Source: Lichess EGTB          engine_wrapper.py:334",
    ];

    #[test]
    fn a_wrapped_tablebase_move_is_reassembled() {
        let mut p = Parser::default();
        let mut last = None;
        for l in EGTB {
            last = p.feed(l);
        }
        assert_eq!(
            last,
            Some(Event::Tablebase {
                game: Some(GameId("0cO0mhHX".into())),
                uci: "f3f2".into(),
                wdl: Some(2),
                dtz: Some(2),
                // The mate distance the engine structurally cannot compute.
                dtm: Some(8),
            })
        );
    }

    /// The complement, and the reason the tablebase case matters: an engine move
    /// leaves a telemetry record and a tablebase move does not.
    #[test]
    fn an_engine_move_is_distinguished_from_a_tablebase_one() {
        let mut p = Parser::default();
        assert_eq!(
            p.feed("[07/28/26 18:52:14] INFO     Source: Engine                engine_wrapper.py:334"),
            Some(Event::Engine)
        );
    }

    /// The window holds several moves' worth of wrapped text; the marker must pick
    /// up the most recent one, not the first still in the buffer.
    #[test]
    fn two_moves_in_the_window_resolve_to_the_later_one() {
        let mut p = Parser::default();
        for l in EGTB {
            p.feed(l);
        }
        let second = [
            "INFO     Got move c1b1 from           engine_wrapper.py:1098",
            "         tablebase.lichess.ovh (wdl:",
            "         2, dtz: 4, dtm: 6) for game",
            "         0cO0mhHX",
            "INFO     Source: Lichess EGTB          engine_wrapper.py:334",
        ];
        let mut last = None;
        for l in second {
            last = p.feed(l);
        }
        match last {
            Some(Event::Tablebase { uci, dtm, .. }) => {
                assert_eq!(uci, "c1b1");
                assert_eq!(dtm, Some(6));
            }
            other => panic!("expected the second move, got {other:?}"),
        }
    }

    /// The restart count the machine panel needs. systemd's `NRestarts` stays at
    /// zero through this, which is why the Python showed a green dot through two of
    /// them in half an hour.
    #[test]
    fn a_watchdog_kill_is_recognised() {
        let mut p = Parser::default();
        assert_eq!(
            p.feed("[watchdog] restarting chess-gpu-bot.service (SIGKILL: the game loop is already dead)"),
            Some(Event::WatchdogRestart)
        );
    }

    #[test]
    fn a_stuck_game_carries_its_id_and_duration() {
        let mut p = Parser::default();
        assert_eq!(
            p.feed("[watchdog] game 8qYzuwSj: our turn for 120s"),
            Some(Event::WatchdogStuck {
                game: Some(GameId("8qYzuwSj".into())),
                seconds: Some(120)
            })
        );
    }

    #[test]
    fn rich_padding_is_stripped_without_eating_content() {
        assert_eq!(
            strip_rich("[07/28/26 18:53:05] INFO     Source: Engine                engine_wrapper.py:334"),
            "Source: Engine"
        );
        assert_eq!(strip_rich("                             0cO0mhHX"), "0cO0mhHX");
        // A message with no padding at all survives intact.
        assert_eq!(strip_rich("[watchdog] restarting"), "[watchdog] restarting");
        // And a genuine trailing number is not mistaken for a source column.
        assert_eq!(strip_rich("INFO     move: 72"), "move: 72");
    }

    #[test]
    fn a_message_that_is_a_byte_array_still_decodes() {
        let j: JournalLine =
            serde_json::from_str(r#"{"MESSAGE":[104,105],"__CURSOR":"c"}"#).unwrap();
        assert_eq!(j.text(), "hi");
    }

    #[test]
    fn an_uninteresting_line_produces_nothing() {
        let mut p = Parser::default();
        assert_eq!(p.feed("info depth 5 score cp 422 nodes 2161 nps 1719"), None);
        assert_eq!(p.feed("                    INFO     move: 72"), None);
    }

    #[test]
    fn the_window_is_bounded() {
        let mut p = Parser::default();
        for i in 0..100 {
            p.feed(&format!("line {i}"));
        }
        assert!(p.window.len() <= 8);
    }
}
