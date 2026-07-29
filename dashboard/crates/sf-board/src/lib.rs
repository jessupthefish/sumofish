//! Position to pixels. No terminal, no scheduling, no async.
//!
//! `sf-term` turns the [`Raster`] into escape sequences; whose thread does the
//! encoding is the app's business.

pub mod cburnett;
pub mod raster;
pub mod svg;

pub use raster::{HalfCell, Raster, Renderer, halfblocks, seam_count};
pub use svg::{Colors, Request, VIEWBOX, board_svg, square_from_name};
