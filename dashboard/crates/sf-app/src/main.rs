//! `sumofish-dash` — the composition root.
//!
//! One task owns `AppState`. Every source sends `Update`s down one bounded channel
//! and the owner applies them; nothing locks, and the render function takes `&AppState`
//! for free. That is the shape every ratatui and tokio reference recommends, and the
//! reason is concrete: with a lock you either hold it across the whole render, which
//! blocks writers, or you render a torn state.

mod config;
mod render;
mod run;

use anyhow::Result;
use std::path::PathBuf;

/// Parsed argv. Hand-rolled rather than clap: six flags do not earn a dependency,
/// and the ones that matter are the ones that make a frame reproducible.
#[derive(Debug, Default)]
pub struct Args {
    pub config: Option<PathBuf>,
    /// Force a tier, for screenshots and snapshots.
    pub tier: Option<String>,
    /// Render one frame at a fixed size to stdout and exit. No terminal, no
    /// alternate screen, no clock: the output is byte-reproducible.
    pub snapshot: Option<(u16, u16)>,
    /// Force a graphics protocol instead of probing.
    pub board: Option<String>,
    /// Do not open any lichess connection. The default until the Python dashboard
    /// is retired, because two dashboards double the load on the same per-IP
    /// endpoint buckets -- which is what put the bot into a 40-minute
    /// no-challenge stall once.
    pub offline: bool,
    /// Replay a telemetry file instead of tailing the live one.
    pub replay: Option<PathBuf>,
    pub speed: f64,
    /// Force the `?` overlay on in a snapshot, so the four responsive tiers can be
    /// checked without a terminal: `--snapshot WxH --overlay`.
    pub overlay: bool,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let _guard = init_tracing()?;

    let cfg = config::Config::load(args.config.as_deref())?;

    if let Some((w, h)) = args.snapshot {
        // A frame, on stdout, with a fixed clock. Used by `cargo insta` and by
        // anything that wants to look at the layout without a terminal.
        let frame = render::snapshot(&cfg, &args, w, h)?;
        print!("{frame}");
        return Ok(());
    }

    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(run::run(cfg, args))
}

fn parse_args() -> Result<Args> {
    let mut a = Args { speed: 1.0, ..Args::default() };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut it = argv.iter();
    while let Some(arg) = it.next() {
        let (key, inline) = match arg.split_once('=') {
            Some((k, v)) => (k, Some(v.to_string())),
            None => (arg.as_str(), None),
        };
        let mut value = || -> Result<String> {
            match inline.clone() {
                Some(v) => Ok(v),
                None => it.next().cloned().ok_or_else(|| anyhow::anyhow!("{key} needs a value")),
            }
        };
        match key {
            "--config" => a.config = Some(PathBuf::from(value()?)),
            "--tier" => a.tier = Some(value()?),
            "--board" => a.board = Some(value()?),
            "--replay" => a.replay = Some(PathBuf::from(value()?)),
            "--speed" => a.speed = value()?.parse().unwrap_or(1.0),
            "--offline" => a.offline = true,
            "--overlay" => a.overlay = true,
            "--snapshot" => {
                let v = value()?;
                let (w, h) = v
                    .split_once('x')
                    .ok_or_else(|| anyhow::anyhow!("--snapshot wants WxH, got {v}"))?;
                a.snapshot = Some((w.parse()?, h.parse()?));
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown flag {other}\n\n{USAGE}"),
        }
    }
    Ok(a)
}

const USAGE: &str = "\
sumofish-dash — the SumoFish dashboard

  --config PATH        config file (default: $XDG_CONFIG_HOME/sumofish/dash.toml)
  --tier NAME          force micro|compact|standard|wide
  --board NAME         force kitty|sixel|text|off instead of probing
  --snapshot WxH       render one frame to stdout and exit
  --overlay            force the ? help overlay on (with --snapshot)
  --replay PATH        replay a telemetry log instead of tailing the live one
  --speed N            replay speed multiplier (default 1)
  --offline            open no lichess connection
";

/// Logs go to a **file**. The TUI owns stdout, and a single `println!` from a
/// background task corrupts the frame.
fn init_tracing() -> Result<tracing_appender::non_blocking::WorkerGuard> {
    use tracing_subscriber::EnvFilter;
    let dir = std::env::var("XDG_STATE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()))
                .join(".local/state")
        })
        .join("sumofish-dash");
    std::fs::create_dir_all(&dir).ok();
    let appender = tracing_appender::rolling::Builder::new()
        .rotation(tracing_appender::rolling::Rotation::DAILY)
        .filename_prefix("dash")
        .filename_suffix("log")
        .max_log_files(5)
        .build(&dir)?;
    // The guard must outlive the program or buffered lines are lost on exit. The
    // classic bug is `let (w, _) = non_blocking(..)`, which drops it immediately.
    let (writer, guard) = tracing_appender::non_blocking(appender);
    tracing_subscriber::fmt()
        .with_writer(writer)
        .with_ansi(false)
        .with_target(false)
        .with_file(true)
        .with_line_number(true)
        .with_env_filter(
            EnvFilter::try_from_env("SUMOFISH_DASH_LOG").unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
    tracing::info!(dir = %dir.display(), "logging to file; stdout belongs to the TUI");
    Ok(guard)
}
