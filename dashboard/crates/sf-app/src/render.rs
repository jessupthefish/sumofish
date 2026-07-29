//! Turning state into a frame.
//!
//! Two callers: the live loop, and `--snapshot`, which renders one frame at a fixed
//! size with a fixed clock and prints it. The second exists because a frame that
//! can only be produced by a terminal cannot be diffed, and every layout question
//! then needs a screenshot -- which the Lab Notes are emphatic about: a screenshot
//! judges how something *looks*, never whether the numbers add up.

use crate::config::Config;
use crate::Args;
use anyhow::Result;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use sf_layout::{CellSize, Cx, PictureRequest, Solved, Tier, solve};
use sf_model::AppState;
use std::time::Instant;

/// Lay out and draw. Returns the solve, so the caller can find the picture's
/// geometry without recomputing it.
pub fn draw(
    buf: &mut Buffer,
    area: Rect,
    state: &AppState,
    now: Instant,
    wall: jiff::Timestamp,
    frame: u64,
    cell: CellSize,
    tier: Option<Tier>,
) -> (Solved, Option<PictureRequest>) {
    let base = Cx { state, now, wall, bot: None, frame, picture: None };
    let solved = solve(area, sf_panels::ALL, &base, cell, tier);

    // The picture geometry is known now, so panels can draw around it.
    let cx = Cx { picture: solved.picture, ..base };

    let mut request = None;
    for placed in &solved.placed {
        let Some(panel) = sf_panels::ALL.iter().find(|p| p.spec().id == placed.id) else {
            continue;
        };
        panel.render(placed.variant, placed.rect, buf, &cx);
        if request.is_none() {
            request = panel.picture(placed.variant, placed.rect, &cx);
        }
    }

    // Reserve the picture's cells so ratatui's diff leaves them alone. With kitty
    // graphics in Konsole this is belt and braces -- the M0 probe wrote text across
    // the image's own rows and it survived -- but the sixel fallback needs it, and
    // ratatui would otherwise write spaces over the region on the next frame.
    if let Some(pic) = solved.picture {
        skip_cells(buf, pic.cells);
    }
    (solved, request)
}

/// Mark a rect as "do not touch" for the buffer diff.
fn skip_cells(buf: &mut Buffer, rect: Rect) {
    let area = buf.area;
    for y in rect.y..rect.y.saturating_add(rect.height).min(area.y + area.height) {
        for x in rect.x..rect.x.saturating_add(rect.width).min(area.x + area.width) {
            if let Some(cell) = buf.cell_mut((x, y)) {
                cell.set_diff_option(ratatui::buffer::CellDiffOption::Skip);
            }
        }
    }
}

/// One frame, as text, with everything that varies pinned.
pub fn snapshot(cfg: &Config, args: &Args, w: u16, h: u16) -> Result<String> {
    let state = crate::run::seed_state(cfg);
    let area = Rect { x: 0, y: 0, width: w, height: h };
    let mut buf = Buffer::empty(area);
    let tier = args.tier.as_deref().and_then(Tier::parse);
    // A fixed clock, so the frame is byte-reproducible. This is the whole reason
    // `Cx` carries `now` instead of letting panels call `Instant::now`.
    let now = Instant::now();
    let (solved, _) = draw(
        &mut buf,
        area,
        &state,
        now,
        jiff::Timestamp::UNIX_EPOCH,
        0,
        CellSize { w: 8, h: 15 },
        tier,
    );

    let mut out = String::new();
    out.push_str(&format!(
        "# tier={} panels={} dropped={}\n",
        solved.tier.name(),
        solved.placed.len(),
        solved.dropped.len()
    ));
    for p in &solved.placed {
        out.push_str(&format!(
            "# {:<8} {:<8} {:>3},{:<3} {}x{}\n",
            p.id.0,
            p.variant.name(),
            p.rect.x,
            p.rect.y,
            p.rect.width,
            p.rect.height
        ));
    }
    for (id, why) in &solved.dropped {
        out.push_str(&format!("# dropped {id}: {why:?}\n"));
    }
    if let Some(pic) = solved.picture {
        out.push_str(&format!("# picture {}px at {:?}\n", pic.px, pic.cells));
    }
    out.push('\n');
    for y in 0..h {
        for x in 0..w {
            out.push_str(buf.cell((x, y)).map(|c| c.symbol()).unwrap_or(" "));
        }
        out.push('\n');
    }
    Ok(out)
}
