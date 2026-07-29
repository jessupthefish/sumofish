//! The engine's narration channel: `logs/engine.jsonl`.
//!
//! The engine narrates; the dashboard only listens. Every number about the search
//! comes from here, written by `search_engine.py` during the search as well as at
//! the end of it, so the viewer never imports torch, never evaluates a position and
//! therefore cannot disagree with the engine or take GPU time from it.
//!
//! # Attribution, and why `pid` is the key
//!
//! `concurrency: 2` means lichess-bot plays two games at once, spawns an engine per
//! game, and **both append to this file**. The Python built a four-layer
//! demultiplexer for that: an explicit game-id claim, a side-to-move filter, a
//! two-ply bridge search between positions, and an orphan queue with a re-anchor
//! pass. All of it existed because `game` was once absent from records.
//!
//! It is present now (`patches/0003` delivers it over UCI as `setoption name
//! GameId`), and **every record carries `pid` unconditionally** -- which nothing in
//! the Python reads. So attribution is: `pid` partitions the interleaved file into
//! per-engine streams, and `game` labels a partition. A pid whose `game` changes is
//! a new partition, not a moved one. Records that cannot be attributed contribute to
//! nothing and are counted, rather than being adopted on a guess: the old bridge
//! search could and did follow the wrong game through a shared opening.

use crate::tailer::{Batch, Tailer};
use serde::Deserialize;
use sf_model::position::{Obs, Pid, parse_fen, parse_uci};
use sf_model::state::*;
use sf_model::{BotId, GameId, Update};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// One telemetry record. Every field the engine emits, including the five in `boot`
/// that the Python parsed and threw away -- so nothing on screen could say which
/// checkpoint was playing, while promotion swaps `runs/value.pt` under a running bot.
#[derive(Debug, Deserialize)]
pub struct Record {
    pub ev: String,
    #[serde(default)]
    pub t: Option<f64>,
    #[serde(default)]
    pub pid: Option<u32>,
    #[serde(default)]
    pub game: Option<String>,
    #[serde(default)]
    pub ply: Option<u32>,
    #[serde(default)]
    pub fen: Option<String>,
    #[serde(default)]
    pub stm: Option<String>,
    #[serde(default)]
    pub wp: Option<f32>,
    #[serde(default)]
    pub wp_white: Option<f32>,
    #[serde(default)]
    pub nodes: Option<u64>,
    /// The engine calls this `nps`, and the number is honest -- it is NN forward
    /// passes over elapsed, which is unique positions evaluated. The *label* is what
    /// PHILOSOPHY forbids, because `sims` sits beside it with a value within 15% and
    /// that is exactly the confusion the rule exists to prevent. Renamed at the
    /// boundary.
    #[serde(default, rename = "nps")]
    pub unique_per_s: Option<u64>,
    #[serde(default)]
    pub sims: Option<u64>,
    #[serde(default)]
    pub elapsed: Option<f64>,
    #[serde(default)]
    pub budget: Option<f64>,
    #[serde(default)]
    pub best: Option<String>,
    #[serde(default)]
    pub pv: Vec<String>,
    /// `[[san, visits, q, prior], ...]`
    #[serde(default)]
    pub top: Vec<TopEntry>,
    #[serde(default)]
    pub mate: bool,
    #[serde(default)]
    pub done: bool,
    #[serde(default)]
    pub uci: Option<String>,
    /// Visits inherited by re-rooting the previous move's tree. Computed by
    /// `MCTS.report()` and dropped by `snapshot()`, so it is absent today; the field
    /// is here so that adding one key engine-side is all it takes.
    #[serde(default)]
    pub reused: Option<u64>,

    // --- boot only ---
    #[serde(default)]
    pub params: Option<u64>,
    #[serde(default)]
    pub policy_step: Option<u64>,
    #[serde(default)]
    pub value_step: Option<u64>,
    #[serde(default)]
    pub bins: Option<u32>,
    #[serde(default)]
    pub batch: Option<u32>,
}

/// `["b4", 13926, 0.396, 0.213]` — san, visits, q, prior.
#[derive(Debug)]
pub struct TopEntry {
    pub san: String,
    pub visits: u64,
    pub q: f32,
    pub prior: f32,
}

impl<'de> Deserialize<'de> for TopEntry {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        // A heterogeneous JSON array. Tolerant of extra elements so an engine-side
        // addition does not break parsing.
        let v = serde_json::Value::deserialize(d)?;
        let a = v.as_array().ok_or_else(|| serde::de::Error::custom("top entry is not an array"))?;
        Ok(TopEntry {
            san: a.first().and_then(|x| x.as_str()).unwrap_or_default().to_string(),
            visits: a.get(1).and_then(|x| x.as_u64()).unwrap_or(0),
            q: a.get(2).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32,
            prior: a.get(3).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32,
        })
    }
}

/// How well a record could be placed.
#[derive(Debug, Default)]
pub struct Stats {
    pub records: u64,
    pub parse_errors: u64,
    /// Records with no `game` and no way to infer one. Contribute to nothing.
    pub unattributed: u64,
    pub boots: u64,
}

/// Reads one telemetry file and turns it into `Update`s.
///
/// One per distinct path. Two bots may name the same file -- the default is one
/// shared `logs/engine.jsonl` for the whole machine -- in which case one reader
/// serves both and `pid` does the separating.
pub struct TelemetrySource {
    tailer: Tailer,
    /// Which bot owns each game, learned from each bot's `playing` list.
    game_owner: HashMap<GameId, BotId>,
    /// Sticky pid -> game, so `think` records land on the right game before and
    /// after `game` appears in the stream.
    pid_game: HashMap<Pid, GameId>,
    /// The last position each pid was searching, so a `move` record's `done`+`uci`
    /// can be applied even if its own `fen` is missing.
    pub stats: Stats,
}

impl TelemetrySource {
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        TelemetrySource {
            tailer: Tailer::new(path),
            game_owner: HashMap::new(),
            pid_game: HashMap::new(),
            stats: Stats::default(),
        }
    }

    /// Tell the reader which bot a game belongs to. Called whenever a bot's
    /// `/api/account/playing` lands. This is the whole of multi-bot attribution:
    /// `game` -> bot, then `pid` -> `game`.
    pub fn own(&mut self, game: GameId, bot: BotId) {
        self.game_owner.insert(game, bot);
    }

    pub async fn poll(&mut self, now: Instant) -> anyhow::Result<Vec<Update>> {
        let batch = self.tailer.poll().await?;
        Ok(self.absorb(batch, now))
    }

    /// Pure, so the whole thing can be tested by replaying a real log.
    pub fn absorb(&mut self, batch: Batch, now: Instant) -> Vec<Update> {
        let mut out = Vec::new();
        if batch.rotated {
            out.push(Update::Tape(TapeEntry {
                at: jiff::Timestamp::UNIX_EPOCH,
                kind: TapeKind::Engine,
                bot: None,
                // A rotation and a quiet engine are indistinguishable otherwise,
                // which is the exact failure the track model exists to prevent.
                text: "telemetry log rotated".into(),
            }));
        }
        for line in &batch.lines {
            self.stats.records += 1;
            let rec: Record = match serde_json::from_str(line) {
                Ok(r) => r,
                Err(e) => {
                    self.stats.parse_errors += 1;
                    tracing::debug!(error = %e, "unparseable telemetry record");
                    continue;
                }
            };
            self.absorb_record(rec, now, &mut out);
        }
        out
    }

    fn absorb_record(&mut self, rec: Record, now: Instant, out: &mut Vec<Update>) {
        let pid = Pid(rec.pid.unwrap_or(0));

        if rec.ev == "boot" {
            self.stats.boots += 1;
            // The payload the Python discarded. "Which checkpoint is playing" is the
            // single most PHILOSOPHY-relevant fact about a live game.
            out.push(Update::EngineBooted {
                pid,
                boot: EngineBoot {
                    params: rec.params,
                    policy_step: rec.policy_step,
                    value_step: rec.value_step,
                    bins: rec.bins,
                    batch: rec.batch,
                    sims_cap: rec.sims,
                },
                at: now,
            });
            // A boot means a new engine process, so any game this pid was on is
            // stale.
            self.pid_game.remove(&pid);
            return;
        }

        // Attribution. `game` on the record is authoritative; the sticky map covers
        // records that predate the GameId option landing in that engine process.
        let game = match rec.game.as_deref() {
            Some(g) => {
                let g = GameId(g.to_string());
                // A pid whose game changed is a NEW partition, not a moved one.
                self.pid_game.insert(pid, g.clone());
                Some(g)
            }
            None => self.pid_game.get(&pid).cloned(),
        };
        let Some(game) = game else {
            self.stats.unattributed += 1;
            return;
        };
        let Some(bot) = self.game_owner.get(&game).cloned() else {
            // We know the game but not whose it is yet. Not an error: telemetry is
            // faster than `/api/account/playing`, which is the entire point of the
            // source ranking. It will be attributed on the next poll.
            self.stats.unattributed += 1;
            return;
        };

        out.push(Update::EngineSeen {
            pid,
            game: Some(game.clone()),
            bot: Some(bot.clone()),
            at: now,
        });

        // The position being searched: after the opponent moved, and instant.
        let Some(fen) = rec.fen.as_deref() else { return };
        let Ok(pos) = parse_fen(fen) else {
            tracing::debug!(fen, "telemetry record carries an unparseable fen");
            return;
        };
        out.push(Update::Position {
            bot: bot.clone(),
            obs: Obs::EngineSearching { pid, game: game.clone(), pos: pos.clone(), at: now },
        });

        // THE FIX. A finished search has chosen, and that move is played as far as
        // the engine is concerned. In the Python this was eleven lines of dead code
        // after an unconditional `return True`, so the board sat on the pre-move
        // position for ~9 seconds until the feed caught up.
        if rec.done {
            if let Some(uci) = rec.uci.as_deref() {
                match parse_uci(&pos, uci) {
                    Ok(mv) => out.push(Update::Position {
                        bot: bot.clone(),
                        obs: Obs::EngineChose { pid, game: game.clone(), from: pos.clone(), mv, at: now },
                    }),
                    Err(e) => tracing::debug!(uci, error = %e, "chosen move does not apply"),
                }
            }
        }

        // The search snapshot.
        if let Some(ply) = rec.ply {
            out.push(Update::Search {
                bot: bot.clone(),
                game: game.clone(),
                snapshot: SearchSnapshot {
                    ply,
                    nodes: rec.nodes.unwrap_or(0),
                    unique_per_s: rec.unique_per_s.unwrap_or(0),
                    sims: rec.sims.unwrap_or(0),
                    elapsed: secs(rec.elapsed),
                    budget: secs(rec.budget),
                    wp_white: rec.wp_white.unwrap_or(0.5),
                    best: rec.best.clone(),
                    pv: rec.pv.clone(),
                    top: rec
                        .top
                        .iter()
                        .map(|t| Candidate {
                            san: t.san.clone(),
                            visits: t.visits,
                            q: t.q,
                            prior: t.prior,
                        })
                        .collect(),
                    mate: rec.mate,
                    done: rec.done,
                    reused: rec.reused,
                    best_changes: 0, // counted by `apply`, which can see the previous
                },
            });

            // The evaluation curve, **always in White's frame**. The records carry
            // both `wp` (side-to-move) and `wp_white` precisely so no consumer has
            // to remember which is which: MCTS stores values from each node's own
            // perspective, so a series across alternating plies is a sawtooth rather
            // than a trend.
            if let Some(wp_white) = rec.wp_white {
                out.push(Update::Eval { bot, game, ply, wp_white });
            }
        }
    }
}

fn secs(v: Option<f64>) -> Duration {
    Duration::from_secs_f64(v.unwrap_or(0.0).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch(lines: &[&str]) -> Batch {
        Batch {
            lines: lines.iter().map(|s| s.to_string()).collect(),
            rotated: false,
            truncated: false,
            size: 0,
        }
    }

    fn src() -> TelemetrySource {
        let mut s = TelemetrySource::new("/dev/null");
        s.own(GameId("g1".into()), BotId("a".into()));
        s
    }

    const THINK: &str = r#"{"ev":"think","t":1,"pid":42,"game":"g1","ply":2,"fen":"rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2","stm":"w","wp":0.52,"wp_white":0.52,"nodes":1000,"nps":1800,"sims":1024,"elapsed":1.5,"budget":20.0,"best":"Nf3","pv":["Nf3","Nc6"],"top":[["Nf3",900,0.52,0.3],["Bc4",100,0.48,0.2]],"mate":false,"done":false}"#;

    #[test]
    fn a_think_record_yields_a_position_a_search_and_an_eval() {
        let mut s = src();
        let ups = s.absorb(batch(&[THINK]), Instant::now());
        let kinds: Vec<&str> = ups
            .iter()
            .map(|u| match u {
                Update::EngineSeen { .. } => "seen",
                Update::Position { obs: Obs::EngineSearching { .. }, .. } => "searching",
                Update::Position { obs: Obs::EngineChose { .. }, .. } => "chose",
                Update::Search { .. } => "search",
                Update::Eval { .. } => "eval",
                _ => "other",
            })
            .collect();
        assert_eq!(kinds, vec!["seen", "searching", "search", "eval"]);
        assert_eq!(s.stats.unattributed, 0);
    }

    /// `nps` is renamed at the boundary, because the number is honest and the label
    /// is the thing PHILOSOPHY forbids.
    #[test]
    fn nps_becomes_unique_per_second() {
        let mut s = src();
        let ups = s.absorb(batch(&[THINK]), Instant::now());
        let snap = ups
            .iter()
            .find_map(|u| match u {
                Update::Search { snapshot, .. } => Some(snapshot),
                _ => None,
            })
            .unwrap();
        assert_eq!(snap.unique_per_s, 1800);
        assert_eq!(snap.sims, 1024, "and sims stays separate, so the two are comparable");
    }

    /// THE FIX, at the source boundary: a `done` record with a `uci` must produce an
    /// `EngineChose` observation, which is what puts our own move on the board
    /// immediately instead of nine seconds later.
    #[test]
    fn a_finished_search_emits_our_chosen_move() {
        let mut s = src();
        let done = THINK.replace(r#""done":false"#, r#""done":true,"uci":"g1f3""#);
        let ups = s.absorb(batch(&[&done]), Instant::now());
        assert!(
            ups.iter().any(|u| matches!(u, Update::Position { obs: Obs::EngineChose { .. }, .. })),
            "a done+uci record must yield EngineChose; this is fusion.py:180"
        );
    }

    #[test]
    fn a_done_record_without_a_uci_yields_no_chosen_move() {
        let mut s = src();
        let done = THINK.replace(r#""done":false"#, r#""done":true"#);
        let ups = s.absorb(batch(&[&done]), Instant::now());
        assert!(!ups.iter().any(|u| matches!(u, Update::Position { obs: Obs::EngineChose { .. }, .. })));
    }

    /// The boot payload the Python discarded. Nothing on screen said which
    /// checkpoint was playing.
    #[test]
    fn a_boot_record_reports_the_checkpoint() {
        let mut s = src();
        let boot = r#"{"ev":"boot","t":1,"pid":42,"params":8911024,"policy_step":280000,"value_step":280000,"bins":64,"batch":64,"sims":100000000}"#;
        let ups = s.absorb(batch(&[boot]), Instant::now());
        let boot = ups
            .iter()
            .find_map(|u| match u {
                Update::EngineBooted { boot, .. } => Some(boot),
                _ => None,
            })
            .expect("a boot update");
        assert_eq!(boot.params, Some(8911024));
        assert_eq!(boot.value_step, Some(280000));
        assert_eq!(boot.batch, Some(64));
    }

    /// The whole of multi-bot attribution: `game` labels a partition, `pid`
    /// separates the interleaved streams.
    #[test]
    fn two_games_interleaved_stay_separate() {
        let mut s = TelemetrySource::new("/dev/null");
        s.own(GameId("g1".into()), BotId("a".into()));
        s.own(GameId("g2".into()), BotId("b".into()));
        let g2 = THINK.replace(r#""game":"g1""#, r#""game":"g2""#).replace(r#""pid":42"#, r#""pid":43"#);
        let ups = s.absorb(batch(&[THINK, &g2, THINK]), Instant::now());
        let bots: Vec<&str> = ups
            .iter()
            .filter_map(|u| match u {
                Update::Position { bot, .. } => Some(bot.0.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(bots, vec!["a", "b", "a"]);
    }

    /// A record with no `game` is attributed by its pid, which is stamped
    /// unconditionally -- and which the Python never read.
    #[test]
    fn pid_attributes_a_record_with_no_game_id() {
        let mut s = src();
        s.absorb(batch(&[THINK]), Instant::now()); // teaches pid 42 -> g1
        let no_game: String = THINK.replace(r#""game":"g1","#, "");
        let ups = s.absorb(batch(&[&no_game]), Instant::now());
        assert!(ups.iter().any(|u| matches!(u, Update::Position { .. })));
        assert_eq!(s.stats.unattributed, 0);
    }

    /// An unattributable record must affect NOTHING. The opposite of the Python,
    /// where such a record was aggressively adopted and could drag the board onto
    /// another game.
    #[test]
    fn an_unattributable_record_changes_nothing() {
        let mut s = TelemetrySource::new("/dev/null");
        let ups = s.absorb(batch(&[THINK]), Instant::now());
        assert!(ups.is_empty(), "no owner means no updates at all");
        assert_eq!(s.stats.unattributed, 1);
    }

    /// A boot invalidates the pid's game, because it is a different engine process
    /// reusing the number.
    #[test]
    fn a_boot_resets_the_pid_mapping() {
        let mut s = src();
        s.absorb(batch(&[THINK]), Instant::now());
        s.absorb(batch(&[r#"{"ev":"boot","pid":42}"#]), Instant::now());
        let no_game: String = THINK.replace(r#""game":"g1","#, "");
        let ups = s.absorb(batch(&[&no_game]), Instant::now());
        assert!(ups.is_empty(), "the old mapping must not survive a boot");
    }

    #[test]
    fn a_garbage_line_is_counted_and_skipped() {
        let mut s = src();
        let ups = s.absorb(batch(&["not json at all", THINK]), Instant::now());
        assert_eq!(s.stats.parse_errors, 1);
        assert_eq!(s.stats.records, 2);
        assert!(!ups.is_empty(), "one bad line must not lose the good ones");
    }

    #[test]
    fn an_unknown_field_does_not_break_parsing() {
        let mut s = src();
        let future = THINK.replace(r#""mate":false"#, r#""mate":false,"brand_new_field":[1,2,3]"#);
        let ups = s.absorb(batch(&[&future]), Instant::now());
        assert_eq!(s.stats.parse_errors, 0, "engine-side additions must not break the reader");
        assert!(!ups.is_empty());
    }

    #[test]
    fn a_rotation_is_announced_on_the_tape() {
        let mut s = src();
        let b = Batch { rotated: true, ..batch(&[]) };
        let ups = s.absorb(b, Instant::now());
        assert!(ups.iter().any(|u| matches!(u, Update::Tape(_))));
    }

    #[test]
    fn the_top_move_ladder_survives_a_short_entry() {
        let mut s = src();
        let short = THINK.replace(r#"["Bc4",100,0.48,0.2]"#, r#"["Bc4",100]"#);
        let ups = s.absorb(batch(&[&short]), Instant::now());
        let snap = ups
            .iter()
            .find_map(|u| match u {
                Update::Search { snapshot, .. } => Some(snapshot),
                _ => None,
            })
            .unwrap();
        assert_eq!(snap.top.len(), 2);
        assert_eq!(snap.top[1].visits, 100);
        assert_eq!(snap.top[1].q, 0.0, "a missing element defaults rather than failing");
    }
}
