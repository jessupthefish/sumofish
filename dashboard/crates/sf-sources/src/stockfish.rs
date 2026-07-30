//! Stockfish's opinion of every move, ours and theirs.
//!
//! The third-party adjudicator. PHILOSOPHY is explicit that **adjudication by the
//! engine under test is a bug**, because a net that is more *confident* rather than
//! more correct banks more wins and miscalibration and strength become the same
//! number. Everything the dashboard shows about how well a move was played comes
//! from here, never from SumoFish's own evaluation.
//!
//! # Driving a UCI engine without deadlocking
//!
//! Hand-rolled: `vampirc-uci` has not been touched since 2022, and the subset needed
//! is one enum and one line parser. Three rules, all documented failure modes:
//!
//! - **Never write to stdin from the same future that reads stdout.** `tokio::process`
//!   documents the deadlock: a child that buffers input before producing output will
//!   hang forever. So one task owns each half and they meet at a channel.
//! - **`kill_on_drop(true)`**, or a Stockfish leaks per dashboard restart. A spawned
//!   child otherwise keeps running after its handle is dropped.
//! - **Wait for `readyok` before assuming a `position` was consumed**, and always
//!   `stop` and drain to `bestmove` before sending a new one. A `go` left running
//!   while the position changes puts the engine in an undefined state.
//!
//! # What a grade means
//!
//! Centipawns lost against Stockfish's best, banded with lichess's own thresholds.
//! Only the bad bands are ever drawn: marking accurate moves would be a column of
//! noise, and the eye is hunting for where the game went wrong.

use anyhow::{Context, Result};
use sf_model::state::Grade;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{mpsc, oneshot};

/// lichess's own bands, in centipawns lost.
const BANDS: [(f64, Grade); 4] = [
    (10.0, Grade::Best),
    (50.0, Grade::Good),
    (100.0, Grade::Inaccuracy),
    (300.0, Grade::Mistake),
];

pub fn grade_of(loss_cp: f64) -> Grade {
    for (threshold, g) in BANDS {
        if loss_cp < threshold {
            return g;
        }
    }
    Grade::Blunder
}

/// Stockfish's verdict on one position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Verdict {
    /// Centipawns, **from White's point of view**. Converted once, here, so no
    /// consumer has to remember whose frame it is in -- the mistake that once had
    /// the move list reading 0.97 while the chart read 0.03.
    pub cp_white: f64,
    /// Mate distance, if the position is decided.
    pub mate: Option<i32>,
    pub depth: u32,
}

impl Verdict {
    /// The same logistic the engine uses for `score cp`, inverted. Keeps the two
    /// curves on one axis so their disagreement is visible rather than an artefact.
    pub fn wp_white(&self) -> f32 {
        if let Some(m) = self.mate {
            return if m > 0 { 1.0 } else { 0.0 };
        }
        (1.0 / (1.0 + 10f64.powf(-self.cp_white / 400.0))) as f32
    }
}

enum Job {
    Analyse { fen: String, depth: u32, reply: oneshot::Sender<Result<Verdict>> },
}

/// A running Stockfish, behind one actor.
pub struct Stockfish {
    tx: mpsc::Sender<Job>,
}

impl Stockfish {
    /// Spawn and handshake. `Ok(None)` when the binary is simply absent, which is
    /// not an error: grading is the one source whose absence costs nothing.
    pub async fn spawn(binary: &Path) -> Result<Option<Stockfish>> {
        if !binary.exists() {
            tracing::info!(path = %binary.display(), "no Stockfish; move grading disabled");
            return Ok(None);
        }
        let mut child = Command::new(binary)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            // Or one leaks per dashboard restart.
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("spawn {}", binary.display()))?;

        let stdin = child.stdin.take().context("no stdin")?;
        let stdout = child.stdout.take().context("no stdout")?;
        let (tx, rx) = mpsc::channel(16);
        tokio::spawn(drive(child, stdin, BufReader::new(stdout), rx));
        Ok(Some(Stockfish { tx }))
    }

    pub async fn analyse(&self, fen: &str, depth: u32) -> Result<Verdict> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Job::Analyse { fen: fen.to_string(), depth, reply })
            .await
            .map_err(|_| anyhow::anyhow!("stockfish is gone"))?;
        rx.await.map_err(|_| anyhow::anyhow!("stockfish dropped the job"))?
    }
}

/// The one task that talks to the engine. Serialising every request here is what
/// makes the protocol rules enforceable: nothing else can interleave a `position`
/// with somebody else's `go`.
async fn drive(
    _child: Child,
    mut stdin: ChildStdin,
    mut stdout: BufReader<tokio::process::ChildStdout>,
    mut rx: mpsc::Receiver<Job>,
) {
    // Handshake. `uciok` then `readyok`, and nothing is sent until both arrive.
    if write_line(&mut stdin, "uci").await.is_err() {
        return;
    }
    if read_until(&mut stdout, "uciok", Duration::from_secs(10)).await.is_none() {
        tracing::warn!("stockfish never answered uci");
        return;
    }
    for cmd in ["setoption name Threads value 1", "setoption name Hash value 64", "isready"] {
        if write_line(&mut stdin, cmd).await.is_err() {
            return;
        }
    }
    if read_until(&mut stdout, "readyok", Duration::from_secs(10)).await.is_none() {
        return;
    }
    tracing::info!("stockfish ready");

    while let Some(job) = rx.recv().await {
        match job {
            Job::Analyse { fen, depth, reply } => {
                let r = analyse_one(&mut stdin, &mut stdout, &fen, depth).await;
                let _ = reply.send(r);
            }
        }
    }
}

async fn analyse_one(
    stdin: &mut ChildStdin,
    stdout: &mut BufReader<tokio::process::ChildStdout>,
    fen: &str,
    depth: u32,
) -> Result<Verdict> {
    write_line(stdin, &format!("position fen {fen}")).await?;
    // Confirm the position was consumed before asking it to think about it.
    write_line(stdin, "isready").await?;
    read_until(stdout, "readyok", Duration::from_secs(5))
        .await
        .context("stockfish did not acknowledge the position")?;
    write_line(stdin, &format!("go depth {depth}")).await?;

    let mut best: Option<Verdict> = None;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    let mut line = String::new();
    loop {
        line.clear();
        let read = tokio::time::timeout_at(deadline, stdout.read_line(&mut line)).await;
        match read {
            Ok(Ok(0)) => anyhow::bail!("stockfish closed"),
            Ok(Ok(_)) => {}
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => {
                // Out of time. Stop and drain, or the next `position` lands while a
                // search is running and the engine's state is undefined.
                let _ = write_line(stdin, "stop").await;
                let _ = read_until(stdout, "bestmove", Duration::from_secs(5)).await;
                anyhow::bail!("stockfish timed out at depth {depth}");
            }
        }
        let l = line.trim();
        if let Some(v) = parse_info(l) {
            best = Some(v);
        }
        if l.starts_with("bestmove") {
            break;
        }
    }
    best.context("stockfish returned no score")
}

/// `info depth 12 ... score cp -24 ...` or `score mate -3`.
///
/// The score is **side-to-move relative**, which is the trap: a series across
/// alternating plies is a sawtooth rather than a trend. The caller flips it, once,
/// into White's frame.
pub fn parse_info(line: &str) -> Option<Verdict> {
    if !line.starts_with("info ") || !line.contains(" score ") {
        return None;
    }
    let toks: Vec<&str> = line.split_whitespace().collect();
    let at = |k: &str| toks.iter().position(|t| *t == k);
    let depth: u32 = at("depth").and_then(|i| toks.get(i + 1)?.parse().ok())?;
    let si = at("score")?;
    match toks.get(si + 1)? {
        &"cp" => Some(Verdict {
            cp_white: toks.get(si + 2)?.parse::<f64>().ok()?,
            mate: None,
            depth,
        }),
        &"mate" => Some(Verdict {
            cp_white: 0.0,
            mate: Some(toks.get(si + 2)?.parse().ok()?),
            depth,
        }),
        _ => None,
    }
}

/// Flip a side-to-move score into White's frame.
pub fn to_white(v: Verdict, white_to_move: bool) -> Verdict {
    if white_to_move {
        v
    } else {
        Verdict { cp_white: -v.cp_white, mate: v.mate.map(|m| -m), depth: v.depth }
    }
}

async fn write_line(stdin: &mut ChildStdin, s: &str) -> Result<()> {
    stdin.write_all(s.as_bytes()).await?;
    stdin.write_all(b"\n").await?;
    // UCI is line-oriented over a pipe; an unflushed command is a hang.
    stdin.flush().await?;
    Ok(())
}

async fn read_until(
    stdout: &mut BufReader<tokio::process::ChildStdout>,
    needle: &str,
    timeout: Duration,
) -> Option<String> {
    let deadline = tokio::time::Instant::now() + timeout;
    let mut line = String::new();
    loop {
        line.clear();
        match tokio::time::timeout_at(deadline, stdout.read_line(&mut line)).await {
            Ok(Ok(0)) | Err(_) => return None,
            Ok(Ok(_)) => {
                if line.contains(needle) {
                    return Some(line.trim().to_string());
                }
            }
            Ok(Err(_)) => return None,
        }
    }
}

/// The default binary, from the repo layout.
pub fn default_binary(root: &Path) -> PathBuf {
    root.join("tools/stockfish/stockfish-ubuntu-x86-64-bmi2")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_centipawn_score_parses() {
        let v = parse_info("info depth 12 seldepth 18 score cp -24 nodes 100 pv e2e4").unwrap();
        assert_eq!(v.cp_white, -24.0);
        assert_eq!(v.depth, 12);
        assert_eq!(v.mate, None);
    }

    #[test]
    fn a_mate_score_parses_and_saturates_the_probability() {
        let v = parse_info("info depth 20 score mate 3 pv e2e4").unwrap();
        assert_eq!(v.mate, Some(3));
        assert_eq!(v.wp_white(), 1.0);
        let v = parse_info("info depth 20 score mate -3 pv e2e4").unwrap();
        assert_eq!(v.wp_white(), 0.0);
    }

    #[test]
    fn lines_without_a_score_are_ignored() {
        assert!(parse_info("info depth 1 currmove e2e4 currmovenumber 1").is_none());
        assert!(parse_info("bestmove e2e4 ponder e7e5").is_none());
        assert!(parse_info("readyok").is_none());
        assert!(parse_info("").is_none());
    }

    /// The trap the whole frame discipline exists for: a UCI score is
    /// side-to-move relative, so a series across alternating plies is a sawtooth
    /// rather than a trend.
    #[test]
    fn scores_are_flipped_into_whites_frame_once() {
        let raw = Verdict { cp_white: 150.0, mate: None, depth: 12 };
        assert_eq!(to_white(raw, true).cp_white, 150.0, "White to move: unchanged");
        assert_eq!(to_white(raw, false).cp_white, -150.0, "Black to move: flipped");
        let mate = Verdict { cp_white: 0.0, mate: Some(3), depth: 12 };
        assert_eq!(to_white(mate, false).mate, Some(-3));
    }

    #[test]
    fn the_probability_is_monotone_in_the_score() {
        let wp = |cp| Verdict { cp_white: cp, mate: None, depth: 1 }.wp_white();
        assert!((wp(0.0) - 0.5).abs() < 1e-6);
        assert!(wp(-500.0) < wp(-100.0));
        assert!(wp(-100.0) < wp(0.0));
        assert!(wp(0.0) < wp(100.0));
        assert!(wp(100.0) < wp(500.0));
        assert!(wp(10000.0) <= 1.0 && wp(-10000.0) >= 0.0);
    }

    /// lichess's own bands. Only the bad ones are ever drawn.
    #[test]
    fn grades_use_lichess_bands() {
        assert_eq!(grade_of(0.0), Grade::Best);
        assert_eq!(grade_of(9.9), Grade::Best);
        assert_eq!(grade_of(10.0), Grade::Good);
        assert_eq!(grade_of(49.9), Grade::Good);
        assert_eq!(grade_of(50.0), Grade::Inaccuracy);
        assert_eq!(grade_of(99.9), Grade::Inaccuracy);
        assert_eq!(grade_of(100.0), Grade::Mistake);
        assert_eq!(grade_of(299.9), Grade::Mistake);
        assert_eq!(grade_of(300.0), Grade::Blunder);
        assert_eq!(grade_of(5000.0), Grade::Blunder);
    }

    #[test]
    fn only_bad_grades_carry_a_mark() {
        assert_eq!(Grade::Best.mark(), "");
        assert_eq!(Grade::Good.mark(), "");
        assert_eq!(Grade::Inaccuracy.mark(), "?!");
        assert_eq!(Grade::Blunder.mark(), "??");
    }

    /// The binary really is where the repo says, so a missing one is a real absence
    /// rather than a wrong path.
    #[test]
    fn the_default_binary_path_matches_the_repo() {
        let p = default_binary(Path::new("/home/nomad/dev/active/chess-gpu"));
        assert!(p.ends_with("tools/stockfish/stockfish-ubuntu-x86-64-bmi2"));
    }
}

// ---------------------------------------------------------------- the grader

use sf_model::{BotId, GameId, Update};
use std::collections::HashMap;

/// Walks a game and asks Stockfish about every position, once.
///
/// Results are cached by ply, because the Python re-walked the entire move list with
/// `push_san` every four seconds even though only the Stockfish calls were memoised.
/// A finished ply's verdict never changes.
pub struct Grader {
    engine: Stockfish,
    depth: u32,
    /// game -> ply -> verdict, in White's frame throughout.
    seen: HashMap<GameId, HashMap<u32, Verdict>>,
}

impl Grader {
    pub fn new(engine: Stockfish, depth: u32) -> Self {
        Grader { engine, depth, seen: HashMap::new() }
    }

    /// Grade whatever is new in this game. `moves` is the UCI move list from the
    /// start position.
    ///
    /// Yields both the whole-game curve and the per-move grades. The Python computed
    /// both and drew only the marks, throwing away the curve that answers "is it
    /// playing well" independently of the rating.
    pub async fn grade(
        &mut self,
        bot: &BotId,
        game: &GameId,
        moves: &[String],
    ) -> Vec<Update> {
        use shakmaty::{CastlingMode, Chess, EnPassantMode, Position, fen::Fen};

        let mut out = Vec::new();
        let cache = self.seen.entry(game.clone()).or_default();
        let mut pos = Chess::default();

        for ply in 0..=moves.len() as u32 {
            let white_to_move = pos.turn() == shakmaty::Color::White;
            let verdict = match cache.get(&ply) {
                Some(v) => *v,
                None => {
                    let fen = Fen::from_position(&pos, EnPassantMode::Legal).to_string();
                    match self.engine.analyse(&fen, self.depth).await {
                        Ok(raw) => {
                            let v = to_white(raw, white_to_move);
                            cache.insert(ply, v);
                            out.push(Update::StockfishEval {
                                bot: bot.clone(),
                                game: game.clone(),
                                ply,
                                wp_white: v.wp_white(),
                            });
                            v
                        }
                        Err(e) => {
                            tracing::debug!(error = %e, ply, "stockfish declined a position");
                            break;
                        }
                    }
                }
            };

            // The grade for the move that LEFT this position, which needs the
            // verdict on both sides of it.
            if ply > 0 {
                if let Some(before) = cache.get(&(ply - 1)).copied() {
                    let mover_is_white = ply % 2 == 1;
                    // Loss from the mover's own point of view: how much worse the
                    // position got for them.
                    let delta = if mover_is_white {
                        before.cp_white - verdict.cp_white
                    } else {
                        verdict.cp_white - before.cp_white
                    };
                    out.push(Update::Grade {
                        bot: bot.clone(),
                        game: game.clone(),
                        ply: ply - 1,
                        grade: grade_of(delta.max(0.0)),
                    });
                }
            }

            let Some(uci) = moves.get(ply as usize) else { break };
            let Ok(mv) = sf_model::position::parse_uci(&pos, uci) else { break };
            pos = match pos.play(mv) {
                Ok(p) => p,
                Err(_) => break,
            };
            let _ = CastlingMode::Standard;
        }
        out
    }

    /// Forget a finished game, so a long session does not accumulate every position
    /// it has ever seen.
    pub fn forget(&mut self, game: &GameId) {
        self.seen.remove(game);
    }
}
