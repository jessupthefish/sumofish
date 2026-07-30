//! Everything the project writes to disk that nothing was reading.
//!
//! Seven readers, all cheap, all polled on their own cadence. What they surface is
//! the bulk of the survey's ranked list:
//!
//! - **The lab.** Nine of fifteen jobs, a forty-hour run in flight, and decisions
//!   taken unattended that gate what gets built next. No source in `scripts/dash/`
//!   ever opened `runs/lab/`.
//! - **Matches**, with the interval PHILOSOPHY says a number is not a result
//!   without, and the `sum(game.seconds) <= job.seconds` check that is physically
//!   impossible to violate legitimately and that nothing has ever run.
//! - **Training**, from the run's own `config.json` rather than a hardcoded 300k
//!   steps and a batch of 1024 -- which made the live run read 13.5% and 76.5h
//!   where the truth is 10.1% and ~26h.
//! - **Versions**, beyond the timestamp and the label: the checkpoint hash is the
//!   only thing on disk that answers "is the bot playing what this version shipped".
//! - **Finished games**, with the per-game rating delta that every PGN carries and
//!   nothing read.
//! - **The rating log**, with the deployment fingerprints that are the entire
//!   stated purpose of the file.
//! - **The watchdog**, whose own state file is the only honest restart count --
//!   systemd's `NRestarts` stays at zero through a `systemctl kill`.

use anyhow::{Context, Result};
use serde::Deserialize;
use sf_model::state::*;
use sf_model::{BotId, Update};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

fn ts(secs: f64) -> Option<jiff::Timestamp> {
    jiff::Timestamp::from_microsecond((secs * 1e6) as i64).ok()
}

// ---------------------------------------------------------------- lab

#[derive(Debug, Deserialize)]
struct LabJson {
    #[serde(default)]
    done: BTreeMap<String, LabDone>,
    #[serde(default)]
    facts: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    current: Option<String>,
    #[serde(default)]
    started: Option<f64>,
    #[serde(default)]
    waiting_since: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct LabDone {
    #[serde(default)]
    outcome: String,
    #[serde(default)]
    detail: Option<String>,
    #[serde(default)]
    seconds: Option<f64>,
    #[serde(default)]
    finished: Option<f64>,
    #[serde(default)]
    summary: Option<String>,
}

pub fn read_lab(state: &Path) -> Result<Lab> {
    let text = std::fs::read_to_string(state).context("read lab state")?;
    let j: LabJson = serde_json::from_str(&text).context("parse lab state")?;

    let mut jobs: Vec<LabJob> = j
        .done
        .iter()
        .map(|(id, d)| LabJob {
            id: id.clone(),
            what: d.summary.clone().unwrap_or_default(),
            status: if d.outcome == "completed" { LabStatus::Done } else { LabStatus::Failed },
            detail: d.summary.clone().or_else(|| d.detail.clone()),
            seconds: d.seconds,
            summary: d.summary.clone(),
        })
        .collect();
    // Chronological, so the queue reads in the order it happened rather than
    // alphabetically by job name.
    jobs.sort_by(|a, b| {
        let key = |x: &LabJob| j.done.get(&x.id).and_then(|d| d.finished).unwrap_or(0.0);
        key(a).partial_cmp(&key(b)).unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some(cur) = &j.current {
        jobs.push(LabJob {
            id: cur.clone(),
            what: String::new(),
            status: LabStatus::Running,
            detail: Some("running".into()),
            seconds: None,
            summary: None,
        });
    }

    // Only the scalar facts. `causal_arms` is a nested object and belongs in a
    // report, not in a two-column list.
    let facts = j
        .facts
        .iter()
        .filter_map(|(k, v)| match v {
            serde_json::Value::Number(n) => Some((k.clone(), n.to_string())),
            serde_json::Value::String(s) if s.len() <= 24 => Some((k.clone(), s.clone())),
            serde_json::Value::Bool(b) => Some((k.clone(), b.to_string())),
            _ => None,
        })
        .collect();

    // The file `lab.py status` tells you to read, against the state that writes it.
    let report = state.with_file_name("report.md");
    let report_stale = match (mtime(&report), mtime(state)) {
        (Some(r), Some(s)) => r < s,
        _ => false,
    };

    Ok(Lab {
        jobs,
        current: j.current,
        started: j.started.and_then(ts),
        waiting_since: j.waiting_since.and_then(ts),
        facts,
        report_stale,
    })
}

fn mtime(p: &Path) -> Option<SystemTime> {
    std::fs::metadata(p).ok()?.modified().ok()
}

// ---------------------------------------------------------------- matches

#[derive(Debug, Deserialize)]
struct MatchGame {
    #[serde(default)]
    result: String,
    #[serde(default)]
    seconds: f64,
    #[serde(default)]
    a_white: bool,
    #[serde(default)]
    score: Option<f64>,
    #[serde(default)]
    code: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MatchConfig {
    #[serde(default)]
    a: MatchArm,
    #[serde(default)]
    b: MatchArm,
}

#[derive(Debug, Default, Deserialize)]
struct MatchArm {
    #[serde(default)]
    label: Option<String>,
}

pub fn read_matches(dir: &Path) -> Result<Vec<MatchRun>> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else { return Ok(out) };
    for e in entries.flatten() {
        let path = e.path();
        if !path.is_dir() {
            continue;
        }
        let games_path = path.join("games.jsonl");
        let Ok(text) = std::fs::read_to_string(&games_path) else { continue };
        let games: Vec<MatchGame> = text
            .lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect();
        if games.is_empty() {
            continue;
        }
        let cfg: MatchConfig = std::fs::read_to_string(path.join("config.json"))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or(MatchConfig { a: MatchArm::default(), b: MatchArm::default() });

        let (mut w, mut d, mut l) = (0u32, 0u32, 0u32);
        for g in &games {
            match g.score {
                Some(s) if s > 0.75 => w += 1,
                Some(s) if s < 0.25 => l += 1,
                Some(_) => d += 1,
                None => match g.result.as_str() {
                    "1-0" => {
                        if g.a_white { w += 1 } else { l += 1 }
                    }
                    "0-1" => {
                        if g.a_white { l += 1 } else { w += 1 }
                    }
                    _ => d += 1,
                },
            }
        }
        let n = (w + d + l) as f64;
        let score = (w as f64 + 0.5 * d as f64) / n.max(1.0);
        let (elo, err) = elo_with_interval(score, n);

        // THE provenance check. Per-game durations summing to more than the wall
        // clock the job was credited is physically impossible, and it is how four
        // replayed match runs were caught. Cheap, total, and never run before.
        let played: f64 = games.iter().map(|g| g.seconds).sum();
        let credited = job_seconds(&path);
        let unfingerprinted = games.iter().all(|g| g.code.is_none());
        let provenance = match credited {
            Some(c) if played > c + 1.0 => Provenance::Impossible,
            _ if unfingerprinted => Provenance::Unfingerprinted,
            Some(_) => Provenance::Ok,
            None => Provenance::Unknown,
        };

        // A match written to within the last two minutes is still going.
        let running = mtime(&games_path)
            .and_then(|m| m.elapsed().ok())
            .is_some_and(|age| age < Duration::from_secs(120));

        out.push(MatchRun {
            name: path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
            a: cfg.a.label.unwrap_or_else(|| "A".into()),
            b: cfg.b.label.unwrap_or_else(|| "B".into()),
            games: (w + d + l),
            pairs: (w + d + l) / 2,
            wins: w,
            draws: d,
            losses: l,
            elo,
            elo_err: err,
            los: los(score, n),
            llr: None,
            concluded: !running,
            running,
            provenance,
            code: games.iter().find_map(|g| g.code.clone()),
        });
    }
    // Live first, then newest.
    out.sort_by_key(|m| (!m.running, m.name.clone()));
    Ok(out)
}

/// The wall clock the job was credited with, from the lab's own record.
fn job_seconds(match_dir: &Path) -> Option<f64> {
    let name = match_dir.file_name()?.to_str()?;
    let state = match_dir.parent()?.parent()?.join("lab/state.json");
    let text = std::fs::read_to_string(state).ok()?;
    let j: LabJson = serde_json::from_str(&text).ok()?;
    // Job ids and match directory names describe the same run in different words:
    // the lab calls it `sims-1600-800` and the directory is `sims-1600-vs-800`.
    // A plain containment check misses that, and a miss reports the provenance as
    // Unknown -- which is exactly the silence that let four replayed match runs go
    // unnoticed. Normalise both sides instead.
    let want = normalise(name);
    j.done
        .iter()
        .find(|(id, _)| normalise(id) == want)
        .and_then(|(_, d)| d.seconds)
}

/// Strip separators and the `vs` infix, so `sims-1600-vs-800` and `sims-1600-800`
/// are recognised as the same run.
fn normalise(s: &str) -> String {
    s.split(['-', '_', '.'])
        .filter(|part| !part.is_empty() && *part != "vs")
        .collect::<Vec<_>>()
        .join("")
        .to_ascii_lowercase()
}

/// Elo from a score, with a 95% interval. **The interval is not optional**: a point
/// estimate on its own is the thing PHILOSOPHY says is not a result.
fn elo_with_interval(score: f64, n: f64) -> (Option<f64>, Option<f64>) {
    if n < 2.0 || score <= 0.0 || score >= 1.0 {
        return (None, None);
    }
    let elo = -400.0 * ((1.0 / score) - 1.0).log10();
    // Binomial standard error on the score, pushed through the local slope of the
    // Elo curve. Approximate, and honest about being an interval rather than a
    // precise one.
    let se = (score * (1.0 - score) / n).sqrt();
    let slope = 400.0 / (score * (1.0 - score) * std::f64::consts::LN_10);
    (Some(elo), Some(1.96 * se * slope))
}

/// Likelihood the first arm is stronger, from the same normal approximation.
fn los(score: f64, n: f64) -> Option<f64> {
    if n < 2.0 {
        return None;
    }
    let se = (score * (1.0 - score) / n).sqrt().max(1e-9);
    let z = (score - 0.5) / se;
    // Standard normal CDF via erf, good to ~1e-7.
    Some(0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2)))
}

fn erf(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26.
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

// ---------------------------------------------------------------- training

#[derive(Debug, Deserialize)]
struct ActiveJson {
    run: String,
    log: PathBuf,
    #[serde(default)]
    pid: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct TrainConfig {
    #[serde(default)]
    steps: Option<u64>,
    #[serde(default)]
    batch_size: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct TrainLine {
    #[serde(default)]
    step: Option<u64>,
    #[serde(default)]
    loss: Option<f64>,
    #[serde(default)]
    lr: Option<f64>,
    #[serde(default)]
    grad_norm: Option<f64>,
    #[serde(default)]
    samples_per_s: Option<f64>,
    #[serde(default)]
    val_loss: Option<f64>,
    #[serde(default)]
    puzzle_acc: Option<f64>,
}

pub fn read_train(active: &Path) -> Result<TrainRun> {
    let text = std::fs::read_to_string(active).context("read active.json")?;
    let a: ActiveJson = serde_json::from_str(&text).context("parse active.json")?;
    let log = std::fs::read_to_string(&a.log).context("read train log")?;

    // The run's OWN config, never a constant. This is the whole difference between
    // an ETA of 26 hours and one of 76.
    let cfg: TrainConfig = a
        .log
        .parent()
        .map(|d| d.join("config.json"))
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or(TrainConfig { steps: None, batch_size: None });

    let mut run = TrainRun {
        name: a.run,
        total_steps: cfg.steps,
        batch_size: cfg.batch_size,
        ..Default::default()
    };
    for line in log.lines() {
        let Ok(l) = serde_json::from_str::<TrainLine>(line) else { continue };
        if let (Some(step), Some(loss)) = (l.step, l.loss) {
            run.step = run.step.max(step);
            run.loss = Some(loss);
            run.loss_hist.push(loss);
            run.lr = l.lr.or(run.lr);
            run.grad_norm = l.grad_norm.or(run.grad_norm);
            run.samples_per_s = l.samples_per_s.or(run.samples_per_s);
        }
        // Eval records carry no `loss`, so they are a separate shape on the same
        // stream.
        if let Some(v) = l.val_loss {
            run.val_loss = Some(v);
            run.val_hist.push((l.step.unwrap_or(0), v));
        }
        if let Some(p) = l.puzzle_acc {
            run.puzzle_acc = Some(p);
            run.puzzle_hist.push((l.step.unwrap_or(0), p));
        }
    }
    // A moving window; the sparkline only ever draws its last `width` values anyway.
    if run.loss_hist.len() > 400 {
        run.loss_hist.drain(..run.loss_hist.len() - 400);
    }

    // Is anything actually watching this? `train_watchdog` watches
    // `chess-gpu-train.service`, which does not exist, so a forty-hour run started
    // by the lab has no stall detector at all.
    run.watchdog_alive = a.pid.is_some_and(|p| Path::new(&format!("/proc/{p}")).exists());
    run.watchdog_unit = Some("chess-gpu-train-watchdog.timer".into());
    Ok(run)
}

// ---------------------------------------------------------------- versions

#[derive(Debug, Deserialize)]
struct VersionJson {
    version: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    layman: String,
    #[serde(default)]
    sha: Option<String>,
    #[serde(default)]
    checkpoint_sha: Option<String>,
    #[serde(default)]
    step: Option<u64>,
    #[serde(default)]
    ts: Option<f64>,
}

pub fn read_version(path: &Path) -> Result<Version> {
    let text = std::fs::read_to_string(path).context("read VERSIONS.jsonl")?;
    let last = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .next_back()
        .context("no versions recorded")?;
    let v: VersionJson = serde_json::from_str(last).context("parse version")?;
    Ok(Version {
        version: v.version,
        title: v.title,
        layman: v.layman,
        sha: v.sha,
        checkpoint_sha: v.checkpoint_sha,
        step: v.step,
        at: v.ts.and_then(ts),
    })
}

// ---------------------------------------------------------------- results

/// Finished games, from the PGNs lichess-bot writes. Zero API cost.
///
/// The Python kept six of the twelve fields `games.py` already extracts. The one
/// that matters most is the per-game rating delta: "lost 7 points to a 1739" is what
/// a person actually wants from a results row, and it is in every PGN.
pub fn read_results(dir: &Path, user: &str, since: Option<jiff::Timestamp>) -> Result<Vec<FinishedGame>> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else { return Ok(out) };
    for e in entries.flatten() {
        let path = e.path();
        if path.extension().and_then(|s| s.to_str()) != Some("pgn") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else { continue };
        for block in text.split("\n\n\n") {
            if let Some(g) = parse_pgn_headers(block, user) {
                if since.is_none_or(|s| g.at.is_none_or(|a| a >= s)) {
                    out.push(g);
                }
            }
        }
    }
    // Newest first, and capped: forty games is more history than a panel can show.
    out.sort_by_key(|g| std::cmp::Reverse(g.at.map(|t| t.as_second()).unwrap_or(0)));
    out.truncate(40);
    Ok(out)
}

fn parse_pgn_headers(block: &str, user: &str) -> Option<FinishedGame> {
    let mut h: BTreeMap<&str, &str> = BTreeMap::new();
    for line in block.lines() {
        let line = line.trim();
        let rest = line.strip_prefix('[')?.strip_suffix(']').unwrap_or(line);
        let (k, v) = rest.split_once(' ')?;
        h.insert(k, v.trim().trim_matches('"'));
    }
    let white = *h.get("White")?;
    let black = *h.get("Black")?;
    let we_are_white = white.eq_ignore_ascii_case(user);
    if !we_are_white && !black.eq_ignore_ascii_case(user) {
        return None;
    }
    let result = match (*h.get("Result")?, we_are_white) {
        ("1-0", true) | ("0-1", false) => "win",
        ("0-1", true) | ("1-0", false) => "loss",
        ("1/2-1/2", _) => "draw",
        // `*` is aborted or abandoned, which is not a draw and must not be counted
        // as one.
        _ => "none",
    };
    let at = match (h.get("UTCDate"), h.get("UTCTime")) {
        (Some(d), Some(t)) => format!("{}T{}Z", d.replace('.', "-"), t).parse().ok(),
        _ => None,
    };
    Some(FinishedGame {
        id: h.get("GameId").map(|s| sf_model::GameId(s.to_string())),
        opponent: if we_are_white { black.into() } else { white.into() },
        opponent_rating: h
            .get(if we_are_white { "BlackElo" } else { "WhiteElo" })
            .and_then(|s| s.parse().ok()),
        result: result.into(),
        reason: h.get("Termination").map(|s| s.to_lowercase()).unwrap_or_default(),
        our_colour: Some(if we_are_white { shakmaty::Color::White } else { shakmaty::Color::Black }),
        rating_delta: h
            .get(if we_are_white { "WhiteRatingDiff" } else { "BlackRatingDiff" })
            .and_then(|s| s.trim_start_matches('+').parse().ok()),
        eco: h.get("ECO").map(|s| s.to_string()),
        opening: h.get("Opening").map(|s| s.to_string()),
        plies: None,
        at,
    })
}

// ---------------------------------------------------------------- rating log

#[derive(Debug, Deserialize)]
struct RatingJson {
    #[serde(default)]
    ts: Option<f64>,
    #[serde(default)]
    ratings: indexmap::IndexMap<String, crate::lichess::PerfJson>,
    #[serde(default)]
    deployed: Option<DeployedJson>,
}

#[derive(Debug, Deserialize)]
struct DeployedJson {
    #[serde(default)]
    engine: Option<String>,
    #[serde(default)]
    policy: Option<String>,
    #[serde(default)]
    value: Option<String>,
    #[serde(default)]
    current: Option<String>,
}

pub fn read_rating_log(path: &Path) -> Result<Vec<RatingSample>> {
    let text = std::fs::read_to_string(path).context("read rating log")?;
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let start = lines.len().saturating_sub(500);
    Ok(lines[start..]
        .iter()
        .filter_map(|l| serde_json::from_str::<RatingJson>(l).ok())
        .filter_map(|r| {
            Some(RatingSample {
                at: r.ts.and_then(ts)?,
                perfs: r
                    .ratings
                    .into_iter()
                    .map(|(k, p)| {
                        (
                            k,
                            Perf {
                                rating: p.rating,
                                games: p.games,
                                rd: p.rd,
                                provisional: p.prov,
                                prog: p.prog,
                            },
                        )
                    })
                    .collect(),
                // The whole stated purpose of this file: a jump can be attributed
                // rather than guessed at. The Python read only `rating`.
                deployed: r.deployed.map(|d| Deployed {
                    engine: d.engine,
                    policy: d.policy,
                    value: d.value,
                    current: d.current,
                }),
            })
        })
        .collect())
}

// ---------------------------------------------------------------- watchdog

#[derive(Debug, Deserialize)]
struct WatchdogJson {
    #[serde(default)]
    last_restart: Option<f64>,
    #[serde(default)]
    our_turn_since: BTreeMap<String, f64>,
}

/// The only honest restart count. systemd's `NRestarts` stays at zero because
/// `watchdog.py` uses `systemctl kill` + `restart` rather than letting
/// `Restart=always` fire -- so the Python panel showed a green dot through two
/// SIGKILLs in half an hour.
pub fn read_watchdog(path: &Path) -> Result<WatchdogState> {
    let text = std::fs::read_to_string(path).context("read watchdog state")?;
    let w: WatchdogJson = serde_json::from_str(&text).context("parse watchdog state")?;
    let last = w.last_restart.and_then(ts);
    let today = jiff::Timestamp::now().as_second() - 86_400;
    Ok(WatchdogState {
        last_restart: last,
        restarts_today: u32::from(last.is_some_and(|t| t.as_second() > today)),
        stuck_games: w.our_turn_since.len() as u32,
    })
}

// ---------------------------------------------------------------- the source

/// Polls every file-backed reader on one slow cadence. They are all a few
/// milliseconds of `read_to_string` and a parse, so one task is plenty.
pub struct FileSource {
    pub bot: BotId,
    pub user: String,
    pub lab_state: Option<PathBuf>,
    pub matches_dir: Option<PathBuf>,
    pub train_active: Option<PathBuf>,
    pub versions: Option<PathBuf>,
    pub pgn_dir: Option<PathBuf>,
    pub rating_log: Option<PathBuf>,
    pub watchdog: Option<PathBuf>,
}

impl FileSource {
    pub fn poll(&self) -> Vec<Update> {
        let mut out = Vec::new();
        // A missing or unreadable file is reported as a source failure, never a
        // panic and never silence: a panel that has gone blank must say why.
        macro_rules! feed {
            ($path:expr, $read:expr, $ok:expr, $err:expr) => {
                if let Some(p) = $path.as_deref() {
                    match $read(p) {
                        Ok(v) => out.push($ok(v)),
                        Err(e) => out.push($err(format!("{}: {e}", p.display()))),
                    }
                }
            };
        }
        feed!(self.lab_state, read_lab, Update::LabQueue, Update::LabFailed);
        feed!(self.train_active, read_train, Update::Train, Update::TrainFailed);
        if let Some(p) = self.matches_dir.as_deref() {
            if let Ok(m) = read_matches(p) {
                out.push(Update::Matches(m));
            }
        }
        if let Some(p) = self.versions.as_deref() {
            if let Ok(v) = read_version(p) {
                out.push(Update::VersionInfo { bot: self.bot.clone(), version: v });
            }
        }
        if let Some(p) = self.pgn_dir.as_deref() {
            if let Ok(g) = read_results(p, &self.user, None) {
                let (w, d, l) = (
                    g.iter().filter(|x| x.result == "win").count() as u32,
                    g.iter().filter(|x| x.result == "draw").count() as u32,
                    g.iter().filter(|x| x.result == "loss").count() as u32,
                );
                out.push(Update::Record {
                    bot: self.bot.clone(),
                    record: Record { wins: w, draws: d, losses: l, since_version: None },
                });
                out.push(Update::Results { bot: self.bot.clone(), games: g });
            }
        }
        if let Some(p) = self.rating_log.as_deref() {
            if let Ok(s) = read_rating_log(p) {
                out.push(Update::RatingLog { bot: self.bot.clone(), samples: s });
            }
        }
        if let Some(p) = self.watchdog.as_deref() {
            if let Ok(w) = read_watchdog(p) {
                out.push(Update::Watchdog { bot: self.bot.clone(), state: w });
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elo_always_comes_with_an_interval() {
        let (e, err) = elo_with_interval(0.6, 300.0);
        assert!(e.unwrap() > 0.0, "a winning score is positive Elo");
        assert!(err.unwrap() > 0.0, "an interval is never optional");
        // A degenerate score has no defined Elo, and must report neither rather
        // than an infinity.
        assert_eq!(elo_with_interval(1.0, 300.0), (None, None));
        assert_eq!(elo_with_interval(0.0, 300.0), (None, None));
        assert_eq!(elo_with_interval(0.6, 1.0), (None, None));
    }

    /// More games must mean a tighter interval, or the number is not measuring
    /// anything.
    #[test]
    fn the_interval_narrows_with_more_games() {
        let (_, few) = elo_with_interval(0.55, 24.0);
        let (_, many) = elo_with_interval(0.55, 2400.0);
        assert!(many.unwrap() < few.unwrap() / 5.0);
        // A 24-game match is worth about +-200 Elo, which is the point CLAUDE.md
        // makes about every historical claim in the repo.
        assert!(few.unwrap() > 100.0, "24 games cannot see a small difference");
    }

    #[test]
    fn los_is_a_probability_and_points_the_right_way() {
        assert!(los(0.5, 100.0).unwrap() > 0.49 && los(0.5, 100.0).unwrap() < 0.51);
        assert!(los(0.7, 300.0).unwrap() > 0.99);
        assert!(los(0.3, 300.0).unwrap() < 0.01);
    }

    const PGN: &str = r#"[Event "rated bullet game"]
[Site "https://lichess.org/wP3dAROt"]
[White "Aravan84529"]
[Black "SumoFish"]
[Result "0-1"]
[GameId "wP3dAROt"]
[UTCDate "2026.07.29"]
[UTCTime "00:53:17"]
[WhiteElo "2252"]
[BlackElo "2163"]
[BlackRatingDiff "+7"]
[ECO "B45"]
[Opening "Sicilian Defense"]
[TimeControl "60+2"]
[Termination "Normal"]"#;

    #[test]
    fn a_pgn_yields_the_fields_the_python_dropped() {
        let g = parse_pgn_headers(PGN, "SumoFish").expect("our game");
        assert_eq!(g.result, "win", "we were Black and Black won");
        assert_eq!(g.opponent, "Aravan84529");
        assert_eq!(g.opponent_rating, Some(2252));
        // The one that matters: what the game actually cost.
        assert_eq!(g.rating_delta, Some(7));
        assert_eq!(g.eco.as_deref(), Some("B45"));
        assert_eq!(g.reason, "normal");
        assert!(g.at.is_some());
    }

    #[test]
    fn a_game_we_did_not_play_is_not_ours() {
        assert!(parse_pgn_headers(PGN, "SomebodyElse").is_none());
    }

    /// `*` is aborted or abandoned. Counting it as a draw would inflate the draw
    /// rate, which is a statistic this project has already been misled by once.
    #[test]
    fn an_aborted_game_is_neither_a_win_a_loss_nor_a_draw() {
        let aborted = PGN.replace(r#"[Result "0-1"]"#, r#"[Result "*"]"#);
        let g = parse_pgn_headers(&aborted, "SumoFish").unwrap();
        assert_eq!(g.result, "none");
    }

    #[test]
    fn a_rating_delta_can_be_negative() {
        let lost = PGN
            .replace(r#"[Result "0-1"]"#, r#"[Result "1-0"]"#)
            .replace(r#"[BlackRatingDiff "+7"]"#, r#"[BlackRatingDiff "-7"]"#);
        let g = parse_pgn_headers(&lost, "SumoFish").unwrap();
        assert_eq!(g.result, "loss");
        assert_eq!(g.rating_delta, Some(-7));
    }

    /// The real names, from the real files. `sims-1600-vs-800` is the directory and
    /// `sims-1600-800` is the lab's id for the same run; failing to connect them
    /// reports the provenance as Unknown, which is the silence that let four
    /// replayed runs go unnoticed.
    #[test]
    fn lab_job_ids_and_match_directory_names_are_reconciled() {
        assert_eq!(normalise("sims-1600-vs-800"), normalise("sims-1600-800"));
        assert_eq!(normalise("sims-3200-vs-1600"), normalise("sims-3200-1600"));
        assert_eq!(normalise("forward-bench"), normalise("forward_bench"));
        // And it must not collapse genuinely different runs together.
        assert_ne!(normalise("sims-400-vs-200"), normalise("sims-800-vs-400"));
        assert_ne!(normalise("reuse-400"), normalise("sims-400-200"));
        assert_ne!(normalise("smoke-harness"), normalise("smoke-pairs"));
    }

    /// The inequality PHILOSOPHY calls the cheapest total check in the repo: it is
    /// physically impossible to play 9,507 seconds of chess in the 5 seconds a job
    /// was credited.
    #[test]
    fn more_play_than_wall_clock_is_flagged_as_impossible() {
        let credited = Some(5.0);
        let played = 9507.7;
        assert!(played > credited.unwrap() + 1.0);
        // And a legitimate run is not flagged.
        assert!(!(2897.3 > 3000.0 + 1.0));
    }

    #[test]
    fn a_block_that_is_not_a_pgn_is_skipped_not_panicked() {
        assert!(parse_pgn_headers("", "SumoFish").is_none());
        assert!(parse_pgn_headers("just some text", "SumoFish").is_none());
        assert!(parse_pgn_headers("[White \"a\"]", "SumoFish").is_none());
    }
}
