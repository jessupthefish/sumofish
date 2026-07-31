//! Size tiers and the four compositions.
//!
//! Tiers choose a composition and **nothing else**. They never compute a size.
//!
//! The Python's mistake was not that it had two branches (`_wide_layout` and
//! `_narrow_layout`); it was that each branch did its own arithmetic with no
//! contract between them, so `_narrow_layout` never set `results_h` and `draw()`
//! needed `getattr(plan, "results_h", 0)` to paper over it. Here a composition is
//! a declaration: a tree of regions with extents, plus a mapping from the region
//! a panel *asks* for to the region this tier actually has.
//!
//! That mapping is what keeps panel presence monotone in the terminal size. If a
//! panel could be exclusive to one tier -- present at `Micro`, gone at `Compact`
//! -- growing the window would make it vanish, which is the same class of
//! surprise as the 8-row bug. Instead every composition accepts every panel and
//! folds the regions it lacks into the ones it has.

use crate::alloc::Alloc;
use crate::panel::{RegionId, Size};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Tier {
    Micro,
    Compact,
    Standard,
    Wide,
}

impl Tier {
    pub const ALL: [Tier; 4] = [Tier::Micro, Tier::Compact, Tier::Standard, Tier::Wide];
    pub const fn name(self) -> &'static str {
        match self {
            Tier::Micro => "micro",
            Tier::Compact => "compact",
            Tier::Standard => "standard",
            Tier::Wide => "wide",
        }
    }
    pub fn parse(s: &str) -> Option<Tier> {
        Tier::ALL.into_iter().find(|t| t.name() == s)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Axis {
    Horizontal,
    Vertical,
}

/// A region: an extent along its parent's split axis, and the axis its own panels
/// stack along.
#[derive(Clone, Copy, Debug)]
pub struct RegionSpec {
    pub id: RegionId,
    /// How panels inside it are stacked.
    pub stack: Axis,
    /// Its size along the parent split's axis.
    pub extent: Alloc,
}

/// A composition tree. Depth is at most three by construction, and there are
/// exactly four of them, each with a snapshot test. There is no general 2-D
/// packer and there will not be one: a packer's output is not predictable enough
/// to memorise, and a dashboard you look at every day should put things in the
/// same place every time.
#[derive(Clone, Copy, Debug)]
pub enum Node {
    Region(RegionSpec),
    Split { axis: Axis, extent: Alloc, children: &'static [Node] },
}

impl Node {
    pub fn extent(&self) -> Alloc {
        match self {
            Node::Region(r) => r.extent,
            Node::Split { extent, .. } => *extent,
        }
    }
}

pub struct Composition {
    pub tier: Tier,
    /// The smallest terminal this composition is used for, and the floor the
    /// pinned panels must fit inside. Asserted by a registry test, which is what
    /// makes the solver's "nothing left to drop" branch unreachable.
    pub min: Size,
    pub root: Node,
    /// Which region a panel actually lands in at this tier. Every composition
    /// accepts every region; the ones it lacks fold inward.
    pub remap: fn(RegionId) -> RegionId,
}

// ---------------------------------------------------------------- extents
//
// The numbers, and where they come from.

/// What the right-hand panels need to be *correct* rather than merely present.
/// The Python's `MIN_WIDE_COLS` comment records the incident: using one number
/// both for the width the dashboard asks the terminal for and the width below
/// which it gives up meant that at 149 columns -- one short of the 150 it asked
/// for, on a window already as wide as the monitor gets -- it fell back to a
/// 46-column text board, a fifty-row empty search panel and no evaluation panel.
const RAIL_MIN: u16 = 52;
/// Where the ladder's bar comes down to half of `GAUGE_MAX`, which is the first
/// thing in that column that visibly loses by being narrower. Past this, extra
/// width is worth more to the board.
const RAIL_MAX: u16 = 66;
/// The narrowest board worth drawing at all.
const BOARD_MIN: u16 = 44;
/// The header. Three rows, full width, and the picture's origin is computed from
/// this rather than from a magic absolute screen row.
const HEADER_H: u16 = 3;
/// The footer's keybinding hint bar. One row, full width, present at every tier
/// including `Micro` -- it is the answer to "how would you know `?` exists",
/// which is the whole reason it cannot itself be hidden behind `?`.
const FOOTER_H: u16 = 1;
const fn footer(extent: Alloc) -> Node {
    Node::Region(RegionSpec { id: RegionId::Footer, stack: Axis::Vertical, extent })
}

/// Slack split between the board and the rail. The board wins, because on a 16:9
/// window a square board is width-bound long before it is height-bound, so width
/// converts to board area and the rail's extra columns mostly do not.
const BOARD_FILL: u16 = 3;
const RAIL_FILL: u16 = 1;

const fn band(extent: Alloc) -> Node {
    Node::Region(RegionSpec { id: RegionId::Band, stack: Axis::Vertical, extent })
}

static MICRO: Composition = Composition {
    tier: Tier::Micro,
    min: Size::new(24, 8),
    root: Node::Split {
        axis: Axis::Vertical,
        extent: Alloc::grow(8, 1),
        children: &[
            // Even at Micro the header stays: without it there is nothing on
            // screen that says which bot you are looking at.
            Node::Region(RegionSpec {
                id: RegionId::Band,
                stack: Axis::Vertical,
                extent: Alloc::fixed(HEADER_H),
            }),
            Node::Region(RegionSpec {
                id: RegionId::Full,
                stack: Axis::Vertical,
                extent: Alloc::grow(4, 1),
            }),
            footer(Alloc::fixed(FOOTER_H)),
        ],
    },
    // Everything folds into one region and the panel with the most to say gets the
    // screen -- except the header and the footer, which keep their own bands.
    // Folding the header in too leaves the band empty and the whole screen shifted
    // down by three blank rows; folding the footer in would put it in direct
    // competition with whatever panel currently owns the screen, and it would lose
    // most of the time -- exactly the discoverability problem this exists to fix.
    remap: |r| match r {
        RegionId::Band => RegionId::Band,
        RegionId::Footer => RegionId::Footer,
        _ => RegionId::Full,
    },
};

static COMPACT: Composition = Composition {
    tier: Tier::Compact,
    min: Size::new(60, 20),
    root: Node::Split {
        axis: Axis::Vertical,
        extent: Alloc::grow(20, 1),
        children: &[
            Node::Region(RegionSpec {
                id: RegionId::Band,
                stack: Axis::Vertical,
                extent: Alloc::fixed(HEADER_H),
            }),
            // Stacked rather than side by side: at 60 columns a board and a rail
            // would both be too narrow to be correct.
            Node::Region(RegionSpec {
                id: RegionId::Board,
                stack: Axis::Vertical,
                extent: Alloc::grow(8, 2),
            }),
            Node::Region(RegionSpec {
                id: RegionId::Rail,
                stack: Axis::Vertical,
                extent: Alloc::grow(6, 3),
            }),
            footer(Alloc::fixed(FOOTER_H)),
        ],
    },
    remap: |r| match r {
        RegionId::RailB | RegionId::Full => RegionId::Rail,
        other => other,
    },
};

static STANDARD: Composition = Composition {
    tier: Tier::Standard,
    min: Size::new(100, 34),
    root: Node::Split {
        axis: Axis::Vertical,
        extent: Alloc::grow(34, 1),
        children: &[
            band(Alloc::fixed(HEADER_H)),
            // 30, not 31: one row of the old floor now goes to the footer instead,
            // so the declared tier minimum (100x34) does not move.
            Node::Split {
                axis: Axis::Horizontal,
                extent: Alloc::grow(30, 1),
                children: &[
                    Node::Region(RegionSpec {
                        id: RegionId::Board,
                        stack: Axis::Vertical,
                        extent: Alloc::grow(BOARD_MIN, BOARD_FILL),
                    }),
                    Node::Region(RegionSpec {
                        id: RegionId::Rail,
                        stack: Axis::Vertical,
                        extent: Alloc::new(RAIL_MIN, RAIL_MAX, RAIL_FILL),
                    }),
                ],
            },
            footer(Alloc::fixed(FOOTER_H)),
        ],
    },
    remap: |r| match r {
        RegionId::RailB | RegionId::Full => RegionId::Rail,
        other => other,
    },
};

static WIDE: Composition = Composition {
    tier: Tier::Wide,
    min: Size::new(176, 44),
    root: Node::Split {
        axis: Axis::Vertical,
        extent: Alloc::grow(44, 1),
        children: &[
            band(Alloc::fixed(HEADER_H)),
            // 40, not 41: one row of the old floor now goes to the footer instead,
            // so the declared tier minimum (176x44) does not move.
            Node::Split {
                axis: Axis::Horizontal,
                extent: Alloc::grow(40, 1),
                children: &[
                    Node::Region(RegionSpec {
                        id: RegionId::Board,
                        stack: Axis::Vertical,
                        extent: Alloc::grow(BOARD_MIN, BOARD_FILL),
                    }),
                    Node::Region(RegionSpec {
                        id: RegionId::Rail,
                        stack: Axis::Vertical,
                        extent: Alloc::new(RAIL_MIN, RAIL_MAX, RAIL_FILL),
                    }),
                    // The second rail is the answer to "must look good
                    // full-screen": past ~1150px a square board is height-bound
                    // on a 44-row window, so a single rail leaves the board
                    // eating width it cannot convert into board.
                    Node::Region(RegionSpec {
                        id: RegionId::RailB,
                        stack: Axis::Vertical,
                        extent: Alloc::new(46, 60, RAIL_FILL),
                    }),
                ],
            },
            footer(Alloc::fixed(FOOTER_H)),
        ],
    },
    remap: |r| match r {
        RegionId::Full => RegionId::Rail,
        other => other,
    },
};

pub fn composition(tier: Tier) -> &'static Composition {
    match tier {
        Tier::Micro => &MICRO,
        Tier::Compact => &COMPACT,
        Tier::Standard => &STANDARD,
        Tier::Wide => &WIDE,
    }
}

/// The largest tier whose composition fits. One function, one file.
pub fn tier_for(w: u16, h: u16) -> Tier {
    Tier::ALL
        .into_iter()
        .rev()
        .find(|t| {
            let m = composition(*t).min;
            w >= m.w && h >= m.h
        })
        .unwrap_or(Tier::Micro)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_is_monotone_in_both_axes() {
        for w in 0..=320u16 {
            for h in 0..=140u16 {
                let t = tier_for(w, h);
                if w < 320 {
                    assert!(tier_for(w + 1, h) >= t, "tier shrank going wider at {w}x{h}");
                }
                if h < 140 {
                    assert!(tier_for(w, h + 1) >= t, "tier shrank going taller at {w}x{h}");
                }
            }
        }
    }

    #[test]
    fn the_thresholds_are_where_they_are_declared() {
        assert_eq!(tier_for(20, 6), Tier::Micro);
        assert_eq!(tier_for(59, 40), Tier::Micro, "too narrow for compact");
        assert_eq!(tier_for(200, 19), Tier::Micro, "too short for compact");
        assert_eq!(tier_for(60, 20), Tier::Compact);
        assert_eq!(tier_for(99, 40), Tier::Compact, "one column short of standard");
        assert_eq!(tier_for(100, 34), Tier::Standard);
        assert_eq!(tier_for(175, 60), Tier::Standard, "one column short of wide");
        assert_eq!(tier_for(176, 44), Tier::Wide);
        assert_eq!(tier_for(240, 90), Tier::Wide);
    }

    /// Every composition must accept every region, or a panel could be exclusive
    /// to a tier and would vanish when the window grew.
    #[test]
    fn every_composition_accepts_every_region() {
        for tier in Tier::ALL {
            let c = composition(tier);
            let mut present = Vec::new();
            collect(&c.root, &mut present);
            for r in RegionId::ALL {
                let mapped = (c.remap)(r);
                assert!(
                    present.contains(&mapped),
                    "{}: region {} remaps to {}, which the tree does not contain",
                    tier.name(),
                    r.name(),
                    mapped.name()
                );
            }
        }
    }

    /// A composition's own minimum must be big enough for its tree's minimums, or
    /// `tier_for` would choose a tier that cannot be laid out.
    #[test]
    fn composition_minimums_are_self_consistent() {
        for tier in Tier::ALL {
            let c = composition(tier);
            let (w, h) = tree_min(&c.root);
            assert!(c.min.w >= w, "{}: declared min width {} < tree's {w}", tier.name(), c.min.w);
            assert!(c.min.h >= h, "{}: declared min height {} < tree's {h}", tier.name(), c.min.h);
        }
    }

    fn collect(n: &Node, out: &mut Vec<RegionId>) {
        match n {
            Node::Region(r) => out.push(r.id),
            Node::Split { children, .. } => children.iter().for_each(|c| collect(c, out)),
        }
    }

    /// Minimum (width, height) a node needs. Along a split's axis the children's
    /// minimums add; across it they take the maximum.
    fn tree_min(n: &Node) -> (u16, u16) {
        match n {
            Node::Region(r) => match r.stack {
                Axis::Vertical => (r.extent.min, 1),
                Axis::Horizontal => (1, r.extent.min),
            },
            Node::Split { axis, children, .. } => {
                let parts: Vec<(u16, u16)> = children.iter().map(tree_min).collect();
                match axis {
                    Axis::Vertical => (
                        parts.iter().map(|p| p.0).max().unwrap_or(1),
                        children.iter().map(|c| c.extent().min).sum(),
                    ),
                    Axis::Horizontal => (
                        children.iter().map(|c| c.extent().min).sum(),
                        parts.iter().map(|p| p.1).max().unwrap_or(1),
                    ),
                }
            }
        }
    }
}
