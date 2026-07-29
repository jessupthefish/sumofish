//! Terminal capabilities and the picture emitters.
//!
//! Everything terminal-specific lives here: what the terminal can do, and the
//! escape sequences that exploit it. `sf-board` produces pixels and knows nothing
//! about any of this.

pub mod caps;
pub mod emit;

pub use caps::{Caps, Graphics, probe};
pub use emit::{ImageKey, Painter, clear, emit};
