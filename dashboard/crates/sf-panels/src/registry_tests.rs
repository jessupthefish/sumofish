//! The layout gate. This is the replacement for `tests/verify_layout.py`, and the
//! difference in scope is the whole point.
//!
//! The Python gate checked the **board column only**, at eight hand-chosen
//! terminal sizes. That is exactly why the `machine` panel silently disappearing
//! on every terminal between 50 and 78 rows tall has been green this whole time:
//! the right-hand column's vertical budget was never asserted at all.
//!
//! This sweeps the full cross product of plausible terminal sizes and asserts six
//! properties, of which the last is the one that would have caught that bug on the
//! day it shipped.

use crate::ALL;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use sf_layout::{CellSize, Cx, PanelId, RegionId, Solved, Tier, check_registry, solve};
use sf_model::AppState;
use sf_model::state::{BotConfig, BotId, BotState};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

const COLS: std::ops::RangeInclusive<u16> = 20..=320;
const ROWS: std::ops::RangeInclusive<u16> = 6..=140;

fn fixture() -> AppState {
    let mut st = AppState::default();
    let cfg = Arc::new(BotConfig {
        id: BotId("sumofish".into()),
        user: "SumoFish".into(),
        label: "live v1".into(),
        unit: Some("chess-gpu-bot".into()),
        telemetry: "/dev/null".into(),
        pgn_dir: None,
        versions: None,
        rating_log: None,
        watch: true,
    });
    st.bots.insert(cfg.id.clone(), BotState::new(cfg.clone()));
    st.focus = Some(cfg.id.clone());
    st
}

fn cx<'a>(st: &'a AppState, now: Instant) -> Cx<'a> {
    Cx {
        state: st,
        now,
        wall: jiff::Timestamp::UNIX_EPOCH,
        bot: None,
        frame: 0,
        picture: None,
    }
}

fn solve_at(st: &AppState, w: u16, h: u16, now: Instant) -> Solved {
    solve(
        Rect { x: 0, y: 0, width: w, height: h },
        ALL,
        &cx(st, now),
        CellSize { w: 8, h: 15 },
        None,
    )
}

/// Every declaration invariant, including the two that make the solver's
/// pathological branches unreachable rather than merely unlikely.
#[test]
fn the_registry_is_well_formed() {
    if let Err(e) = check_registry(ALL) {
        panic!("registry invariant violated: {e}");
    }
}

/// Property 1: the parts sum to the whole, in every region, at every size.
/// This is the assertion the Python never had, and it fails directly on the
/// `results_h` bug.
#[test]
fn every_region_is_exactly_filled_by_its_panels() {
    let st = fixture();
    let now = Instant::now();
    for w in COLS {
        for h in ROWS {
            let s = solve_at(&st, w, h, now);
            let mut by_region: HashMap<RegionId, Vec<Rect>> = HashMap::new();
            for p in &s.placed {
                by_region.entry(p.region).or_default().push(p.rect);
            }
            for (region, rect) in &s.regions {
                let Some(rects) = by_region.get(region) else { continue };
                // Panels stack along one axis, so their extents along it must add
                // up to the region's, exactly.
                let used: u32 = rects.iter().map(|r| r.height as u32).sum();
                let vertical = rects.iter().all(|r| r.width == rect.width);
                if vertical {
                    assert_eq!(
                        used, rect.height as u32,
                        "{w}x{h} tier {} region {}: panels total {used} rows but the region has {}",
                        s.tier.name(),
                        region.name(),
                        rect.height
                    );
                }
            }
        }
    }
}

/// Property 2: no panel is ever given less than the variant it was placed at
/// declared it needs. This is what "a clipped panel is a lie" means as a test.
#[test]
fn no_panel_is_ever_clipped() {
    let st = fixture();
    let now = Instant::now();
    let spec_of: HashMap<PanelId, &'static sf_layout::PanelSpec> =
        ALL.iter().map(|p| (p.spec().id, p.spec())).collect();
    for w in COLS {
        for h in ROWS {
            let s = solve_at(&st, w, h, now);
            for p in &s.placed {
                let spec = spec_of[&p.id];
                let v = spec
                    .variants
                    .iter()
                    .find(|v| v.id == p.variant)
                    .expect("placed at a variant it declares");
                assert!(
                    p.rect.width >= v.min.w && p.rect.height >= v.min.h,
                    "{w}x{h}: {} placed at {}x{} but its {} variant needs {}x{}",
                    p.id,
                    p.rect.width,
                    p.rect.height,
                    v.id.name(),
                    v.min.w,
                    v.min.h
                );
            }
        }
    }
}

/// Property 3: no two panels overlap. The Python had no such check, and the
/// picture-under-the-tape family of bugs is exactly this.
#[test]
fn no_two_panels_overlap() {
    let st = fixture();
    let now = Instant::now();
    for w in COLS {
        for h in ROWS {
            let s = solve_at(&st, w, h, now);
            for (i, a) in s.placed.iter().enumerate() {
                for b in &s.placed[i + 1..] {
                    assert!(
                        !intersects(a.rect, b.rect),
                        "{w}x{h}: {} {:?} overlaps {} {:?}",
                        a.id,
                        a.rect,
                        b.id,
                        b.rect
                    );
                }
            }
        }
    }
}

/// Property 4: everything placed is inside the terminal. A rect that runs off the
/// bottom is how rows nobody paints appear, and whatever was in them survives
/// every frame afterwards.
#[test]
fn nothing_is_placed_outside_the_screen() {
    let st = fixture();
    let now = Instant::now();
    for w in COLS {
        for h in ROWS {
            let s = solve_at(&st, w, h, now);
            for p in &s.placed {
                assert!(
                    p.rect.x + p.rect.width <= w && p.rect.y + p.rect.height <= h,
                    "{w}x{h}: {} at {:?} runs off the screen",
                    p.id,
                    p.rect
                );
            }
        }
    }
}

/// Property 5: the picture is inside the board region and always on the 26-pixel
/// grid. Both were comments in the Python, recomputed in three places.
#[test]
fn the_picture_stays_on_the_grid_and_inside_its_region() {
    let st = fixture();
    let now = Instant::now();
    for w in COLS {
        for h in ROWS {
            let s = solve_at(&st, w, h, now);
            let Some(pic) = s.picture else { continue };
            assert_eq!(pic.px % sf_layout::GRID_PX, 0, "{w}x{h}: {}px is off-grid", pic.px);
            let board = s.region(RegionId::Board).expect("a picture implies a board region");
            assert!(
                contains(board, pic.cells),
                "{w}x{h}: picture {:?} escapes the board region {board:?}",
                pic.cells
            );
        }
    }
}

/// Property 6, the flagship. **Presence must be monotone in both axes**: if a
/// panel is on screen at some size, it is on screen at every larger size.
///
/// Non-monotone presence is always a bug and is otherwise invisible. In the Python
/// the `machine` panel is present at 49 rows, absent from 50 to 78, and present
/// again at 79 -- and every existing test stayed green through all of it. A
/// property test over the grid catches the whole class, not just the instance.
#[test]
fn panel_presence_is_monotone_in_both_axes() {
    let st = fixture();
    let now = Instant::now();
    let ws: Vec<u16> = COLS.collect();
    let hs: Vec<u16> = ROWS.collect();

    // present[panel][wi][hi]
    let mut present: HashMap<PanelId, Vec<Vec<bool>>> = ALL
        .iter()
        .map(|p| (p.spec().id, vec![vec![false; hs.len()]; ws.len()]))
        .collect();
    for (wi, &w) in ws.iter().enumerate() {
        for (hi, &h) in hs.iter().enumerate() {
            let s = solve_at(&st, w, h, now);
            for p in &s.placed {
                present.get_mut(&p.id).expect("registry panel")[wi][hi] = true;
            }
        }
    }

    for (id, grid) in &present {
        for wi in 0..ws.len() {
            for hi in 0..hs.len() {
                if !grid[wi][hi] {
                    continue;
                }
                if wi + 1 < ws.len() {
                    assert!(
                        grid[wi + 1][hi],
                        "{id} is present at {}x{} but vanishes at {}x{} -- widening the window \
                         must never remove a panel",
                        ws[wi],
                        hs[hi],
                        ws[wi + 1],
                        hs[hi]
                    );
                }
                if hi + 1 < hs.len() {
                    assert!(
                        grid[wi][hi + 1],
                        "{id} is present at {}x{} but vanishes at {}x{} -- growing the window \
                         must never remove a panel",
                        ws[wi],
                        hs[hi],
                        ws[wi],
                        hs[hi + 1]
                    );
                }
            }
        }
    }
}

/// A dropped panel always says why, so the debug overlay can answer "where did my
/// machine panel go" instead of leaving it a mystery for however long the bug
/// lives.
#[test]
fn every_absent_panel_has_a_recorded_reason() {
    let st = fixture();
    let now = Instant::now();
    for w in [20u16, 40, 60, 100, 150, 200, 320] {
        for h in [6u16, 12, 20, 34, 44, 60, 94, 140] {
            let s = solve_at(&st, w, h, now);
            for p in ALL {
                let id = p.spec().id;
                if s.is_placed(id) {
                    continue;
                }
                assert!(
                    s.dropped.iter().any(|(d, _)| *d == id),
                    "{w}x{h}: {id} is neither placed nor explained"
                );
            }
        }
    }
}

/// Rendering must not panic at any size, including degenerate ones. A dashboard
/// that dies on a narrow window is worse than one that shows less.
#[test]
fn rendering_never_panics_at_any_size() {
    let st = fixture();
    let now = Instant::now();
    for w in [0u16, 1, 2, 5, 19, 20, 40, 79, 80, 100, 175, 176, 320] {
        for h in [0u16, 1, 2, 5, 6, 11, 12, 19, 20, 33, 34, 43, 44, 140] {
            let area = Rect { x: 0, y: 0, width: w, height: h };
            let s = solve_at(&st, w, h, now);
            let mut buf = Buffer::empty(area);
            let c = Cx { picture: s.picture, ..cx(&st, now) };
            for p in &s.placed {
                let panel =
                    ALL.iter().find(|x| x.spec().id == p.id).expect("placed panel is in registry");
                panel.render(p.variant, p.rect, &mut buf, &c);
            }
        }
    }
}

/// Forcing a tier must work at any size, because that is how screenshots and
/// snapshots are taken.
#[test]
fn a_forced_tier_is_honoured_and_still_safe() {
    let st = fixture();
    let now = Instant::now();
    for tier in Tier::ALL {
        for (w, h) in [(20u16, 6u16), (100, 34), (320, 140)] {
            let area = Rect { x: 0, y: 0, width: w, height: h };
            let s = solve(area, ALL, &cx(&st, now), CellSize { w: 8, h: 15 }, Some(tier));
            assert_eq!(s.tier, tier);
            for p in &s.placed {
                assert!(p.rect.x + p.rect.width <= w && p.rect.y + p.rect.height <= h);
            }
        }
    }
}

fn intersects(a: Rect, b: Rect) -> bool {
    a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height
}

fn contains(outer: Rect, inner: Rect) -> bool {
    inner.x >= outer.x
        && inner.y >= outer.y
        && inner.x + inner.width <= outer.x + outer.width
        && inner.y + inner.height <= outer.y + outer.height
}
