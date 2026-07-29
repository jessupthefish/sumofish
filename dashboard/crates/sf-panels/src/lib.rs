//! The panel registry.
//!
//! **This is the only file that changes when a panel is added.** Create
//! `src/<name>.rs` with a `pub static PANEL`, add the identifier below. The macro
//! emits the `mod` declaration and the slice entry.
//!
//! Compare the Python: one new panel was eight edit sites across five files
//! (`panels.py`, `sources.py`, `state.py`, `watch.py` twice -- the `Plan`
//! geometry chain *and* the `draw()` dispatch -- and then `verify_layout.py` in a
//! follow-up commit to catch the clipping bug the first one shipped). Nothing
//! enumerated panels, so nothing could be added by declaration.
//!
//! The list order is meaningful: it is the tie-break for which of two
//! equal-weight panels is dropped first when a region runs out of room.

#![deny(clippy::disallowed_methods)]

sf_layout::panels! {
    header,
}

#[cfg(test)]
mod registry_tests;
