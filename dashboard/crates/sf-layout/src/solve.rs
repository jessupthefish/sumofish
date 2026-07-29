//! The solver: regions from the composition, panels from the registry, and one
//! answer that every consumer reads.
//!
//! The only code permitted to compute a size. Nothing outside this module may
//! derive a rect, and the only way a panel gets one is by being handed it in
//! [`Solved`]. That is what replaces the Python's triplicated geometry --
//! `Plan.image_*`, `board_panel`'s eight positional parameters, and
//! `verify_layout.py` recomputing the same arithmetic a third time, with the test
//! docstring admitting "nothing in either one can notice a disagreement".

use crate::alloc::{Alloc, allocate};
use crate::panel::{
    Cx, Panel, PanelId, PictureGeom, Relevance, RegionId, Size, Variant, VariantId, Weight, snap26,
};
use crate::tiers::{Axis, Composition, Node, RegionSpec, Tier, composition, tier_for};
use ratatui::layout::Rect;

/// A panel that made it onto the screen.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Placed {
    pub id: PanelId,
    pub region: RegionId,
    pub variant: VariantId,
    pub rect: Rect,
}

/// Why a panel is not on the screen. Bound to a debug overlay, because "where did
/// my machine panel go" was unanswerable for however long that bug was live and
/// it should cost one keypress.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DropReason {
    /// Nothing to say right now.
    Hidden,
    /// The region is too narrow (or short) for even the smallest variant.
    CrossAxisTooSmall { need: u16, have: u16 },
    /// Room ran out along the stacking axis and this was the cheapest to lose.
    NoRoom { need: u16, have: u16 },
    /// This tier has no region for it. Should never happen -- every composition
    /// accepts every region -- and is here so that a violation is visible rather
    /// than silent.
    NoRegion,
}

#[derive(Clone, Debug, Default)]
pub struct Solved {
    pub tier: Tier,
    pub placed: Vec<Placed>,
    pub dropped: Vec<(PanelId, DropReason)>,
    pub regions: Vec<(RegionId, Rect)>,
    /// Where the picture goes. Computed once, here, from the board panel's own
    /// declared demand.
    pub picture: Option<PictureGeom>,
}

impl Default for Tier {
    fn default() -> Self {
        Tier::Standard
    }
}

impl Solved {
    pub fn rect_of(&self, id: PanelId) -> Option<Rect> {
        self.placed.iter().find(|p| p.id == id).map(|p| p.rect)
    }
    pub fn region(&self, id: RegionId) -> Option<Rect> {
        self.regions.iter().find(|(r, _)| *r == id).map(|(_, rect)| *rect)
    }
    pub fn is_placed(&self, id: PanelId) -> bool {
        self.placed.iter().any(|p| p.id == id)
    }
}

/// Terminal cell size in pixels, probed once and re-probed on resize. Needed to
/// convert the board region's cells into an image size.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CellSize {
    pub w: u16,
    pub h: u16,
}

impl Default for CellSize {
    /// Konsole at Hack 10 on this machine, measured. Only used when the terminal
    /// declines to answer the query; a wrong guess makes the picture the wrong
    /// shape, not broken.
    fn default() -> Self {
        CellSize { w: 8, h: 15 }
    }
}

/// Lay out `area`. Pure: same inputs, same output, no terminal involved, which is
/// what lets the whole size grid be swept in a unit test.
pub fn solve(
    area: Rect,
    panels: &[&dyn Panel],
    cx: &Cx<'_>,
    cell: CellSize,
    tier_override: Option<Tier>,
) -> Solved {
    let tier = tier_override.unwrap_or_else(|| tier_for(area.width, area.height));
    let comp = composition(tier);

    let mut regions = Vec::new();
    split(&comp.root, area, &mut regions);

    let mut out = Solved { tier, placed: Vec::new(), dropped: Vec::new(), regions, picture: None };

    // Bucket panels by the region this tier actually puts them in, keeping
    // registry order within a bucket -- that order is the documented tie-break.
    for (region_id, rect) in out.regions.clone() {
        let spec = region_spec(&comp.root, region_id);
        let mut candidates: Vec<Candidate> = Vec::new();
        for (order, p) in panels.iter().enumerate() {
            let s = p.spec();
            if (comp.remap)(s.region) != region_id {
                continue;
            }
            let rel = p.relevance(cx);
            if rel == Relevance::Hidden {
                out.dropped.push((s.id, DropReason::Hidden));
                continue;
            }
            candidates.push(Candidate {
                id: s.id,
                weight: s.weight,
                relevance: rel,
                order,
                variants: s.variants,
                chosen: 0,
            });
        }
        let stack = spec.map(|s| s.stack).unwrap_or(Axis::Vertical);
        fit(rect, stack, candidates, region_id, &mut out);
    }

    // The picture's geometry comes out of the same solve, once. The board panel
    // declares which cells it wants; we turn that into pixels.
    if let Some(p) = out.placed.iter().find(|p| p.region == RegionId::Board) {
        if let Some(panel) = panels.iter().find(|x| x.spec().id == p.id) {
            if let Some(req) = panel.picture(p.variant, p.rect, cx) {
                out.picture = Some(picture_geom(req.cells, cell));
            }
        }
    }
    out
}

/// Cells to pixels. Snapped to the 26-pixel grid so the board never grows a
/// hairline it does not have.
pub fn picture_geom(cells: Rect, cell: CellSize) -> PictureGeom {
    let by_width = cells.width as u32 * cell.w as u32;
    let by_height = cells.height as u32 * cell.h as u32;
    let px = snap26(by_width.min(by_height));
    PictureGeom { cells, px, cell_w: cell.w, cell_h: cell.h }
}

struct Candidate {
    id: PanelId,
    weight: Weight,
    relevance: Relevance,
    order: usize,
    variants: &'static [Variant],
    chosen: usize,
}

impl Candidate {
    fn variant(&self) -> &'static Variant {
        &self.variants[self.chosen]
    }
    /// Step down one level of detail. False when there is nothing less detailed.
    fn demote(&mut self) -> bool {
        if self.chosen + 1 < self.variants.len() {
            self.chosen += 1;
            true
        } else {
            false
        }
    }
}

/// The fit loop. This is where the 8-row bug class dies.
fn fit(
    rect: Rect,
    stack: Axis,
    mut candidates: Vec<Candidate>,
    region: RegionId,
    out: &mut Solved,
) {
    if rect.width == 0 || rect.height == 0 {
        for c in candidates {
            out.dropped.push((c.id, DropReason::NoRoom { need: 1, have: 0 }));
        }
        return;
    }
    let (main_extent, cross_extent) = match stack {
        Axis::Vertical => (rect.height, rect.width),
        Axis::Horizontal => (rect.width, rect.height),
    };
    let cross_of = |v: &Variant| match stack {
        Axis::Vertical => v.min.w,
        Axis::Horizontal => v.min.h,
    };
    let main_of = |v: &Variant| match stack {
        Axis::Vertical => (v.min.h, v.pref.h),
        Axis::Horizontal => (v.min.w, v.pref.w),
    };

    // Step 2: cross axis FIRST. A 52-column rail cannot host a 60-column variant,
    // and discovering that after solving the main axis is how you get a panel
    // that is placed and then clipped.
    candidates.retain(|c| {
        let ok = c.variants.iter().any(|v| cross_of(v) <= cross_extent);
        if !ok {
            let need = c.variants.iter().map(cross_of).min().unwrap_or(u16::MAX);
            out.dropped.push((c.id, DropReason::CrossAxisTooSmall { need, have: cross_extent }));
        }
        ok
    });
    for c in candidates.iter_mut() {
        c.chosen = c
            .variants
            .iter()
            .position(|v| cross_of(v) <= cross_extent)
            .expect("retain kept only candidates with a fitting variant");
    }

    // Step 3: demote, then drop, until the set provably fits. `need` is a sum over
    // the same Vec that is about to be laid out -- you cannot forget a term in
    // that the way you can forget `- self.results_h` in a hand-written expression.
    //
    // **Demotion comes before dropping, and `Pinned` does not exempt a panel from
    // it.** `Pinned` means "never dropped", not "never shrunk". Conflating the two
    // is a real bug the size grid caught within seconds of a second pinned panel
    // existing: at 24x10 both `header` and `board` picked their largest
    // cross-fitting variant, their minimums no longer fit the height, neither was
    // demotable, and the fallback dropped `board` -- so the board was present at 23
    // columns and gone at 24. Growing the window removed a panel, which is the
    // whole class this gate exists to forbid.
    let cheapest = |cs: &[Candidate], want_demotable: bool| -> Option<usize> {
        cs.iter()
            .enumerate()
            .filter(|(_, c)| {
                if want_demotable {
                    c.chosen + 1 < c.variants.len()
                } else {
                    c.weight != Weight::Pinned
                }
            })
            .min_by_key(|(_, c)| (c.weight, c.relevance, std::cmp::Reverse(c.order)))
            .map(|(i, _)| i)
    };

    loop {
        let need: u32 = candidates.iter().map(|c| main_of(c.variant()).0 as u32).sum();
        if need <= main_extent as u32 {
            break;
        }
        // Showing less beats showing nothing, so try every demotion first. A
        // demotion must re-satisfy the cross axis too.
        if let Some(i) = cheapest(&candidates, true) {
            let c = &mut candidates[i];
            let mut demoted = false;
            while c.demote() {
                if cross_of(c.variant()) <= cross_extent {
                    demoted = true;
                    break;
                }
            }
            if demoted {
                continue;
            }
            // It ran out of variants that fit across; it can only be dropped now.
            let c = candidates.remove(i);
            out.dropped.push((c.id, DropReason::NoRoom { need: need as u16, have: main_extent }));
            continue;
        }
        // Nothing left to shrink. Drop the cheapest droppable panel.
        if let Some(i) = cheapest(&candidates, false) {
            let c = candidates.remove(i);
            out.dropped.push((c.id, DropReason::NoRoom { need: need as u16, have: main_extent }));
            continue;
        }
        // Every panel is pinned and at its smallest, and they still do not fit.
        // `check_registry` forbids this at declaration time; reaching it means a
        // composition's floor is wrong. Drop rather than clip -- a missing panel is
        // honest and a clipped one is a lie -- and leave the reason behind.
        let need_now = need as u16;
        match candidates.pop() {
            Some(c) => {
                out.dropped.push((c.id, DropReason::NoRoom { need: need_now, have: main_extent }));
            }
            None => return,
        }
    }

    // Step 4: the set provably fits, so allocate.
    let allocs: Vec<Alloc> = candidates
        .iter()
        .map(|c| {
            let v = c.variant();
            let (min, pref) = main_of(v);
            if v.grow == 0 {
                Alloc::fixed(min)
            } else {
                Alloc::new(min, pref.max(min), v.grow)
            }
        })
        .collect();
    let sizes = allocate(main_extent, &allocs)
        .expect("the drop loop guarantees the minimums fit; this is the invariant");

    let mut cursor = match stack {
        Axis::Vertical => rect.y,
        Axis::Horizontal => rect.x,
    };
    for (c, size) in candidates.iter().zip(sizes) {
        let r = match stack {
            Axis::Vertical => Rect { x: rect.x, y: cursor, width: rect.width, height: size },
            Axis::Horizontal => Rect { x: cursor, y: rect.y, width: size, height: rect.height },
        };
        cursor += size;
        out.placed.push(Placed { id: c.id, region, variant: c.variant().id, rect: r });
    }
}

/// Divide an area among a composition's children, recursively.
fn split(node: &Node, area: Rect, out: &mut Vec<(RegionId, Rect)>) {
    match node {
        Node::Region(r) => out.push((r.id, area)),
        Node::Split { axis, children, .. } => {
            let extent = match axis {
                Axis::Vertical => area.height,
                Axis::Horizontal => area.width,
            };
            let allocs: Vec<Alloc> = children.iter().map(|c| c.extent()).collect();
            // If even the minimums do not fit, hand each child its share of what
            // there is rather than returning nothing: the tier gate should have
            // prevented this, and a visibly cramped screen beats a blank one.
            let sizes = allocate(extent, &allocs).unwrap_or_else(|| {
                let n = children.len().max(1) as u16;
                let each = extent / n;
                let mut v = vec![each; children.len()];
                if let Some(last) = v.last_mut() {
                    *last += extent - each * n;
                }
                v
            });
            let mut cursor = match axis {
                Axis::Vertical => area.y,
                Axis::Horizontal => area.x,
            };
            for (child, size) in children.iter().zip(sizes) {
                let sub = match axis {
                    Axis::Vertical => {
                        Rect { x: area.x, y: cursor, width: area.width, height: size }
                    }
                    Axis::Horizontal => {
                        Rect { x: cursor, y: area.y, width: size, height: area.height }
                    }
                };
                cursor += size;
                split(child, sub, out);
            }
        }
    }
}

fn region_spec(node: &Node, id: RegionId) -> Option<RegionSpec> {
    match node {
        Node::Region(r) if r.id == id => Some(*r),
        Node::Region(_) => None,
        Node::Split { children, .. } => children.iter().find_map(|c| region_spec(c, id)),
    }
}

/// Assertions every registry must satisfy. Called from a test in `sf-panels`,
/// where the registry lives, so a badly declared panel cannot ship.
pub fn check_registry(panels: &[&dyn Panel]) -> Result<(), String> {
    let mut ids: Vec<&str> = panels.iter().map(|p| p.spec().id.0).collect();
    ids.sort_unstable();
    let n = ids.len();
    ids.dedup();
    if ids.len() != n {
        return Err("duplicate PanelId".into());
    }

    for p in panels {
        let s = p.spec();
        if s.variants.is_empty() {
            return Err(format!("{}: no variants", s.id));
        }
        // Most detailed first, minimums strictly decreasing on both axes, so the
        // "first that fits" scan is correct and demotion always makes progress.
        for w in s.variants.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            if b.min.w > a.min.w || b.min.h > a.min.h {
                return Err(format!(
                    "{}: variant {} is not smaller than {} ({}x{} vs {}x{})",
                    s.id,
                    b.id.name(),
                    a.id.name(),
                    b.min.w,
                    b.min.h,
                    a.min.w,
                    a.min.h
                ));
            }
            if b.min.w == a.min.w && b.min.h == a.min.h {
                return Err(format!(
                    "{}: variants {} and {} have identical minimums, so one is unreachable",
                    s.id,
                    a.id.name(),
                    b.id.name()
                ));
            }
        }
        for v in s.variants {
            if v.pref.w < v.min.w || v.pref.h < v.min.h {
                return Err(format!("{}: variant {} prefers less than its min", s.id, v.id.name()));
            }
        }
    }

    // The floor that makes the solver's over-constrained branch unreachable: at
    // every tier, the pinned panels' smallest variants must fit that tier's
    // declared minimum size.
    for tier in Tier::ALL {
        let comp: &Composition = composition(tier);
        let mut regions = Vec::new();
        split(&comp.root, Rect { x: 0, y: 0, width: comp.min.w, height: comp.min.h }, &mut regions);
        for (region_id, rect) in regions {
            let spec = region_spec(&comp.root, region_id);
            let stack = spec.map(|s| s.stack).unwrap_or(Axis::Vertical);
            let (main, cross) = match stack {
                Axis::Vertical => (rect.height, rect.width),
                Axis::Horizontal => (rect.width, rect.height),
            };
            let pinned: Vec<&'static crate::panel::PanelSpec> = panels
                .iter()
                .map(|p| p.spec())
                .filter(|s| s.weight == Weight::Pinned && (comp.remap)(s.region) == region_id)
                .collect();
            let need: u32 = pinned
                .iter()
                .map(|s| {
                    let v = s.variants.last().expect("checked non-empty");
                    match stack {
                        Axis::Vertical => v.min.h as u32,
                        Axis::Horizontal => v.min.w as u32,
                    }
                })
                .sum();
            if need > main as u32 {
                return Err(format!(
                    "tier {} region {}: pinned panels need {need} but the tier's floor gives {main}",
                    tier.name(),
                    region_id.name()
                ));
            }
            // The CROSS axis is checked for *every* panel, not just the pinned
            // ones, and this is what keeps panel presence monotone in the terminal
            // size.
            //
            // Without it there is a real hole at every tier boundary. A rail is
            // full-width at `Compact` but only `RAIL_MIN` columns once `Standard`
            // splits the screen, so a panel needing 60 columns would be placed at
            // 99 columns wide and *dropped* at 100 -- widening the window makes it
            // vanish, which is the same species of surprise as the 8-row bug.
            //
            // The rule that closes it: every panel must have at least one variant
            // narrow enough for the narrowest region it can ever land in. That is
            // a real constraint on panel authors (a `Glyph` variant is often the
            // answer) and it is the price of never having a panel disappear when
            // the window grows.
            for s in panels.iter().map(|p| p.spec()).filter(|s| (comp.remap)(s.region) == region_id)
            {
                let narrowest = s
                    .variants
                    .iter()
                    .map(|v| match stack {
                        Axis::Vertical => v.min.w,
                        Axis::Horizontal => v.min.h,
                    })
                    .min()
                    .expect("checked non-empty");
                if narrowest > cross {
                    return Err(format!(
                        "tier {} region {}: panel {} has no variant narrower than {narrowest} \
                         across, but the region is only {cross} there. Add a smaller variant, or \
                         it will vanish when the window is made WIDER.",
                        tier.name(),
                        region_id.name(),
                        s.id
                    ));
                }
            }
        }
    }
    Ok(())
}

/// A `Size` helper for panel declarations, so the specs read as data.
pub const fn size(w: u16, h: u16) -> Size {
    Size::new(w, h)
}
