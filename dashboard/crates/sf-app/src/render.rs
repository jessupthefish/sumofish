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
use ratatui::text::Span;
use sf_layout::{CellSize, Cx, DropReason, PictureRequest, Solved, Tier, solve};
use sf_model::AppState;
use sf_model::state::Overlay;
use sf_theme::Styles;
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

    // The overlay replaces the frame rather than layering on top of it: at `Micro`
    // there is no room to show both a panel's content and a legible registry
    // table over it, and a toggle should show one clean thing, not two competing
    // ones. No picture is requested while it is up, so the run loop's paint branch
    // is a no-op even though `solved.picture` is still `Some` underneath.
    if state.ui.overlay != Overlay::None {
        draw_overlay(buf, area, &solved);
        return (solved, None);
    }

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

/// The `?` overlay: every panel this build knows about, self-reported (name,
/// keybind, one-line purpose) via `PanelSpec` rather than a hardcoded list here --
/// so a panel absorbed in a later step is covered by being registered, the same
/// contract the registry already gives `check_registry` and the snapshot tests.
/// Paired with *why* anything is missing right now, from `Solved::dropped`: design
/// doc's own words for the gap this closes are "where did my machine panel go".
/// Also carries a permanent tip pointing at `sumofish demo`, the offline replay
/// mode that exists to make this exact overlay (and the rest of the layout)
/// checkable without a live game -- previously discoverable only by reading
/// `~/bin/sumofish`'s comments.
fn draw_overlay(buf: &mut Buffer, area: Rect, solved: &Solved) {
    let mut y = area.y;
    // Every row goes through `fit`, not a plain `put`: this is a list (of keys, of
    // panels), and the same rule as the footer applies -- a row clipped mid-word
    // must say so, or it reads as though nothing was cut. `fit` needs owned spans
    // since a truncated span is a new, shorter one.
    let mut line = |buf: &mut Buffer, spans: Vec<Span<'static>>| {
        if y < area.y + area.height {
            sf_layout::widgets::put(buf, area, y, sf_layout::widgets::fit(spans, area.width));
            y += 1;
        }
    };

    // Prelude is fixed: title, blank, "keys", the key line, the demo-mode tip,
    // blank, "panels". Two more rows per registered panel after that. Known
    // before the loop runs, so a vertical shortfall can be declared up front
    // rather than discovered by the reader counting how many panels went
    // missing off the bottom.
    let prelude = 7u16;
    let needed = prelude + 2 * sf_panels::ALL.len() as u16;
    let short = needed.saturating_sub(area.height);

    line(
        buf,
        vec![Span::styled(
            format!(
                " sumofish -- help   tier {} at {}x{}   press ? to close ",
                solved.tier.name(),
                area.width,
                area.height
            ),
            Styles::title(),
        )],
    );
    if short > 0 {
        line(
            buf,
            vec![Span::styled(
                format!("{short} row(s) below do not fit -- widen or grow the terminal to see them"),
                Styles::ink(sf_theme::WARM),
            )],
        );
    } else {
        line(buf, vec![]);
    }

    line(buf, vec![Span::styled("keys", Styles::title())]);
    let mut keyline: Vec<Span<'static>> = Vec::new();
    for (i, (key, what)) in sf_panels::footer::KEYS.iter().enumerate() {
        if i > 0 {
            keyline.push(Span::styled("   ", Styles::faint()));
        }
        keyline.push(Span::styled(*key, Styles::accent()));
        keyline.push(Span::styled(format!(" {what}"), Styles::dim()));
    }
    line(buf, keyline);
    // `sumofish demo` exists precisely so the layout below is checkable without
    // a live game -- the launcher script's own comment calls it "a regression
    // test you can watch" -- but nothing in the running TUI itself said so
    // before this line. A permanent line here, not a state-gated one: it is
    // useful whether or not a game happens to be running right now, and it
    // stays reachable with a keypress instead of requiring a read of the shell
    // script that wraps this binary.
    line(
        buf,
        vec![
            Span::styled("tip  ", Styles::title()),
            Span::styled("no live game? ", Styles::dim()),
            Span::styled("sumofish demo", Styles::accent()),
            Span::styled(" replays recorded telemetry so you can see the full layout anytime.", Styles::dim()),
        ],
    );
    line(buf, vec![]);

    line(buf, vec![Span::styled("panels", Styles::title())]);
    for p in sf_panels::ALL {
        let spec = p.spec();
        let region = format!("{:<6}", spec.region.name());
        let status = match solved.rect_of(spec.id) {
            Some(rect) => {
                let variant = solved
                    .placed
                    .iter()
                    .find(|pl| pl.id == spec.id)
                    .map(|pl| pl.variant.name())
                    .unwrap_or("?");
                format!("{region} {}@{variant} {}x{}", spec.id, rect.width, rect.height)
            }
            None => match solved.dropped.iter().find(|(id, _)| *id == spec.id) {
                Some((_, why)) => format!("{region} {} -- {}", spec.id, drop_text(why)),
                None => format!("{region} {} -- not evaluated this frame", spec.id),
            },
        };
        let key = spec.keybind.unwrap_or("");
        // A trailing literal space after each padded field, not folded into the
        // width, so a title or key longer than the column (`"lichess budget"`,
        // `"Tab / 1-9"`) still gets a separator instead of running straight into
        // the next column -- `{:<14}` alone only pads short content, it does not
        // truncate long content, so without this the two visually merge.
        line(
            buf,
            vec![
                Span::styled(format!("{key:<9} "), Styles::accent()),
                Span::styled(format!("{:<14} ", spec.title), Styles::body()),
                Span::styled(spec.help, Styles::dim()),
            ],
        );
        line(buf, vec![Span::styled(format!("          {status}"), Styles::faint())]);
    }
}

/// A short, human sentence for why a panel is not on screen. Mirrors what
/// `DropReason`'s own variants already say, in `render::snapshot`'s `{:?}` form --
/// this is the prose version, for the overlay a person is actually reading.
fn drop_text(reason: &DropReason) -> String {
    match reason {
        DropReason::Hidden => "nothing to show right now".to_string(),
        DropReason::CrossAxisTooSmall { need, have } => {
            format!("region is {have} cols wide, needs {need}")
        }
        DropReason::NoRoom { need, have } => {
            format!("no room: {have} available, {need} needed")
        }
        DropReason::NoRegion => "no region for it at this tier (this is a bug)".to_string(),
    }
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
    let mut state = crate::run::seed_state(cfg);
    if args.overlay {
        state.ui.overlay = Overlay::Layout;
    }
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
