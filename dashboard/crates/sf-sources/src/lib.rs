//! Every source of data, and all of the I/O.
//!
//! Sources produce [`sf_model::Update`]s and send them down one bounded channel to
//! the single task that owns the state. They never touch `AppState` and they cannot
//! see a panel: this crate does not depend on `sf-panels` or `sf-layout`, so the
//! Python's `from .sources import GATE` inside a render function is unrepresentable.

pub mod files;
pub mod governor;
pub mod journal;
pub mod lichess;
pub mod machine;
pub mod tailer;
pub mod telemetry;

pub use files::FileSource;
pub use governor::{Api, Endpoint, Governor};
pub use journal::JournalSource;
pub use lichess::{HttpFetcher, LichessSource};
pub use machine::{GpuSource, UnitsSource};
pub use tailer::{Batch, Tailer};
pub use telemetry::TelemetrySource;
