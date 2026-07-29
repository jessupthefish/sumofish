//! Layout: the `Panel` contract, the tiers, and the one allocator.
//!
//! Nothing outside this crate may compute a rect. A panel receives one from
//! [`Solved`] and that is the only way it can obtain one.

pub mod alloc;
pub mod panel;
pub mod solve;
pub mod tiers;
pub mod widgets;

pub use alloc::{Alloc, allocate};
pub use panel::{
    Action, Cx, GRID_PX, Panel, PanelId, PanelSpec, PictureGeom, PictureRequest, RegionId,
    Relevance, Scope, Size, Variant, VariantId, Weight, snap26,
};
pub use solve::{CellSize, DropReason, Placed, Solved, check_registry, picture_geom, size, solve};
pub use tiers::{Axis, Composition, Tier, composition, tier_for};
