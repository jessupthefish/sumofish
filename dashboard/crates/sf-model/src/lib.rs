//! The SumoFish dashboard's domain layer.
//!
//! This crate is the bottom of the dependency graph and it is deliberately
//! starved: no tokio, no ratatui, no reqwest, no nvml, no zbus. That is what
//! makes `sf-panels` and `sf-sources` siblings that cannot see each other while
//! still sharing a vocabulary.
//!
//! Three things live here and nowhere else:
//!
//! - [`position::Resolver`] decides which position is on the screen. Exactly one
//!   place, because in the Python two places decided and a docstring asked a
//!   human to keep them in sync.
//! - [`tracked::Tracked`] carries a value together with how stale it is, with no
//!   accessor that hands over the value alone.
//! - [`update::apply`] is the single funnel through which anything enters
//!   [`state::AppState`].

pub mod position;
pub mod state;
pub mod tracked;
pub mod update;

pub use position::{GameId, Obs, Pid, Ply, Resolved, Resolver, Source as PosSource};
pub use state::{AppState, BotConfig, BotId, BotState, GameState};
pub use tracked::{Track, Tracked, ago};
pub use update::{Applied, Update, apply};
