//! Configuration.
//!
//! Two rules decide what goes where:
//!
//! - **Per-bot is only ever a name, a credential, a file path, or a unit.** The
//!   lichess username in particular: it was a hardcoded constant in six places in
//!   the Python, one of them inside `_we_are_black()` where the `--user` flag could
//!   not reach it, and that function decides which way the evaluation gauge reads.
//! - **Everything rate-limited is global**, because lichess's limits are per-IP and
//!   per-endpoint independent of credential. Four bots polling
//!   `/api/account/playing` every two seconds is 120 requests a minute against one
//!   bucket, not four buckets of 30.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub dash: Dash,
    #[serde(default)]
    pub api: Api,
    #[serde(default, rename = "bot")]
    pub bots: Vec<Bot>,
    #[serde(default)]
    pub machine: Machine,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dash {
    /// Redraws per second. The engine narrates at ~6.7Hz and a chess clock needs
    /// tenths only under ten seconds, so twelve is generous.
    #[serde(default = "default_fps")]
    pub fps: u32,
    /// `auto` | `kitty` | `sixel` | `text` | `off`.
    #[serde(default = "default_board")]
    pub board: String,
    /// `auto` or a tier name, for screenshots and snapshots.
    #[serde(default = "default_auto")]
    pub tier: String,
}

fn default_fps() -> u32 {
    12
}
fn default_board() -> String {
    "auto".into()
}
fn default_auto() -> String {
    "auto".into()
}

impl Default for Dash {
    fn default() -> Self {
        Dash { fps: default_fps(), board: default_board(), tier: default_auto() }
    }
}

/// Global, because the limits are.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Api {
    /// Floor between any two lichess calls. lichess publishes no numeric limits at
    /// all -- "various strategies", and "some limits may require longer" than the
    /// usual minute -- so "Only make one request at a time" is the only concrete
    /// guidance there is, and this is it.
    #[serde(default = "default_min_gap_ms")]
    pub min_gap_ms: u64,
    /// Our share of the documented eight concurrent streams per IP. lichess-bot
    /// already holds one event stream plus up to two game streams **per bot**, so at
    /// two bots it is using six of the eight.
    #[serde(default = "default_streams")]
    pub game_streams: u32,
    /// Never open a bot game stream unless this is set and F4 has passed.
    /// lichess-bot holds that exact stream with that exact token, and the downside
    /// of a collision is a live rated game.
    #[serde(default)]
    pub bot_stream: bool,
}

fn default_min_gap_ms() -> u64 {
    250
}
fn default_streams() -> u32 {
    1
}

impl Default for Api {
    fn default() -> Self {
        Api {
            min_gap_ms: default_min_gap_ms(),
            game_streams: default_streams(),
            bot_stream: false,
        }
    }
}

impl Api {
    pub fn min_gap(&self) -> Duration {
        Duration::from_millis(self.min_gap_ms)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bot {
    pub id: String,
    /// The lichess account. **Never a constant in code.** Verified against
    /// `/api/account` at startup rather than trusted, because the token and this
    /// string are two independent assertions and nothing reconciled them before.
    pub user: String,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub token_file: Option<PathBuf>,
    #[serde(default = "default_token_var")]
    pub token_var: String,
    #[serde(default)]
    pub unit: Option<String>,
    pub telemetry: PathBuf,
    #[serde(default)]
    pub pgn_dir: Option<PathBuf>,
    #[serde(default)]
    pub versions: Option<PathBuf>,
    #[serde(default)]
    pub rating_log: Option<PathBuf>,
    /// Whether to spend any lichess budget on this bot. The knob that makes "add
    /// more bots as it grows" affordable.
    #[serde(default = "yes")]
    pub watch: bool,
}

fn default_token_var() -> String {
    "LICHESS_BOT_TOKEN".into()
}
fn yes() -> bool {
    true
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Machine {
    /// The units to watch. Note what the Python polled: exactly two names, one of
    /// which (`chess-gpu-train.service`) does not exist and therefore rendered a
    /// permanently red dot, while `chess-gpu-lab` -- the unit actually doing the
    /// work -- was absent entirely.
    #[serde(default = "default_units")]
    pub units: Vec<String>,
    #[serde(default)]
    pub lab_state: Option<PathBuf>,
    #[serde(default)]
    pub train_active: Option<PathBuf>,
    #[serde(default)]
    pub matches_dir: Option<PathBuf>,
}

fn default_units() -> Vec<String> {
    [
        "chess-gpu-bot.service",
        "chess-gpu-lab.service",
        "chess-gpu-train.service",
        "chess-gpu-watchdog.timer",
        "chess-gpu-train-watchdog.timer",
        "chess-gpu-rating.timer",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

impl Config {
    /// Load from an explicit path, else the XDG config dir, else a default that
    /// watches the one bot this repo has.
    pub fn load(explicit: Option<&std::path::Path>) -> Result<Config> {
        let path = match explicit {
            Some(p) => Some(p.to_path_buf()),
            None => default_path(),
        };
        let Some(path) = path.filter(|p| p.exists()) else {
            tracing::info!("no config file; using the built-in default");
            return Ok(Config::builtin());
        };
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("read {}", path.display()))?;
        let cfg: Config = toml::from_str(&text)
            .with_context(|| format!("parse {}", path.display()))?;
        cfg.validate()?;
        tracing::info!(path = %path.display(), bots = cfg.bots.len(), "config loaded");
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        let mut ids: Vec<&str> = self.bots.iter().map(|b| b.id.as_str()).collect();
        ids.sort_unstable();
        let n = ids.len();
        ids.dedup();
        anyhow::ensure!(ids.len() == n, "duplicate bot id in config");
        anyhow::ensure!(self.dash.fps >= 1 && self.dash.fps <= 60, "fps must be 1..=60");
        Ok(())
    }

    /// What this repo looks like today, so the thing runs with no config at all.
    pub fn builtin() -> Config {
        let root = PathBuf::from("/home/nomad/chess-gpu");
        Config {
            dash: Dash::default(),
            api: Api::default(),
            machine: Machine {
                units: default_units(),
                lab_state: Some(root.join("runs/lab/state.json")),
                train_active: Some(root.join("runs/active.json")),
                matches_dir: Some(root.join("runs/matches")),
            },
            bots: vec![Bot {
                id: "sumofish".into(),
                user: "SumoFish".into(),
                label: Some("live".into()),
                token_file: Some(
                    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/home/nomad".into()))
                        .join(".config/chess-gpu/bot.env"),
                ),
                token_var: default_token_var(),
                unit: Some("chess-gpu-bot.service".into()),
                telemetry: root.join("logs/engine.jsonl"),
                pgn_dir: Some(root.join("logs/games")),
                versions: Some(root.join("VERSIONS.jsonl")),
                rating_log: Some(root.join("logs/rating.jsonl")),
                watch: true,
            }],
        }
    }
}

fn default_path() -> Option<PathBuf> {
    use etcetera::BaseStrategy;
    etcetera::choose_base_strategy()
        .ok()
        .map(|s| s.config_dir().join("sumofish/dash.toml"))
}

/// Read a token by variable name from an env-file, never printing it.
///
/// The value is returned but never logged, never displayed and never included in
/// an error message: everything typed into this process's output could end up in a
/// transcript.
pub fn read_token(file: &std::path::Path, var: &str) -> Option<String> {
    // An env var wins, which is how systemd supplies it.
    if let Ok(v) = std::env::var(var) {
        if !v.is_empty() {
            return Some(v);
        }
    }
    let text = std::fs::read_to_string(file).ok()?;
    for line in text.lines() {
        let line = line.trim().trim_start_matches("export ");
        if let Some(rest) = line.strip_prefix(&format!("{var}=")) {
            let v = rest.trim().trim_matches('"').trim_matches('\'');
            if !v.is_empty() {
                return Some(v.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_builtin_config_is_valid_and_watches_one_bot() {
        let c = Config::builtin();
        c.validate().unwrap();
        assert_eq!(c.bots.len(), 1);
        assert_eq!(c.bots[0].user, "SumoFish");
        assert!(!c.api.bot_stream, "the undelayed bot stream stays off until F4 passes");
    }

    #[test]
    fn duplicate_bot_ids_are_rejected() {
        let toml = r#"
            [[bot]]
            id = "a"
            user = "A"
            telemetry = "/x"
            [[bot]]
            id = "a"
            user = "B"
            telemetry = "/y"
        "#;
        let c: Config = toml::from_str(toml).unwrap();
        assert!(c.validate().is_err());
    }

    #[test]
    fn an_unknown_key_is_an_error_rather_than_silently_ignored() {
        // A typo in a config file that silently does nothing is worse than one that
        // refuses to start.
        let toml = r#"
            [dash]
            fpss = 30
        "#;
        assert!(toml::from_str::<Config>(toml).is_err());
    }

    #[test]
    fn a_minimal_bot_block_fills_in_the_rest() {
        let toml = r#"
            [[bot]]
            id = "x"
            user = "X"
            telemetry = "/tmp/x.jsonl"
        "#;
        let c: Config = toml::from_str(toml).unwrap();
        assert_eq!(c.bots[0].token_var, "LICHESS_BOT_TOKEN");
        assert!(c.bots[0].watch);
        assert_eq!(c.dash.fps, 12);
        assert_eq!(c.api.min_gap_ms, 250);
    }

    #[test]
    fn tokens_are_read_by_name_from_an_env_file() {
        let dir = std::env::temp_dir().join(format!("sf-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("bot.env");
        std::fs::write(&f, "# a comment\nexport LICHESS_BOT_TOKEN=\"abc123\"\nOTHER=x\n").unwrap();
        assert_eq!(read_token(&f, "LICHESS_BOT_TOKEN").as_deref(), Some("abc123"));
        assert_eq!(read_token(&f, "NOT_THERE"), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn fps_is_bounded() {
        let mut c = Config::builtin();
        c.dash.fps = 0;
        assert!(c.validate().is_err());
        c.dash.fps = 120;
        assert!(c.validate().is_err());
    }
}
