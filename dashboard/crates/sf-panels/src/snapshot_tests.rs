//! What each panel actually draws, pinned.
//!
//! The gap this closes: fourteen panels had zero coverage of their *output*. The
//! layout gate proves every panel gets a rect it fits in; it says nothing about what
//! goes in the rect. The `results` reader shipped broken for a whole session and
//! rendered an empty panel, which is indistinguishable from a panel with nothing to
//! say — and nothing in the suite looked.
//!
//! Three assertions, because `insta` alone is not enough:
//!
//! 1. **Text snapshots**, table-driven off the registry, so a new panel is covered
//!    by being registered rather than by someone remembering to add a test.
//! 2. **Palette conformance.** `insta` captures text only, so a panel could go
//!    entirely the wrong colour and every snapshot would still pass. Every colour
//!    rendered must be one the theme declares.
//! 3. **Colour-map snapshots**, which is how colour gets into `insta` after all: the
//!    same buffer rendered as a grid of one-character ink names. The diff shows where
//!    each ink went, and catches "the whole panel went dim" — which a text snapshot
//!    cannot.

use crate::{ALL, fixture};
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;
use sf_layout::{Cx, PanelId, VariantId};
use sf_model::AppState;
use std::collections::BTreeSet;
use std::time::Instant;

/// A representative rect per variant. Generous enough that panels draw their
/// content rather than their degraded form, except where the variant IS the
/// degraded form.
fn area_for(v: VariantId) -> Rect {
    let (w, h) = match v {
        VariantId::Full => (62, 16),
        VariantId::Compact => (40, 8),
        VariantId::Line => (40, 1),
        VariantId::Glyph => (12, 1),
    };
    Rect { x: 0, y: 0, width: w, height: h }
}

fn draw(id: PanelId, v: VariantId, st: &AppState, now: Instant) -> Buffer {
    let panel = ALL.iter().find(|p| p.spec().id == id).expect("a registered panel");
    let area = area_for(v);
    let mut buf = Buffer::empty(area);
    let cx = Cx {
        state: st,
        now,
        wall: fixture::wall(),
        bot: None,
        frame: 0,
        picture: None,
    };
    panel.render(v, area, &mut buf, &cx);
    buf
}

fn as_text(buf: &Buffer) -> String {
    let a = buf.area;
    (0..a.height)
        .map(|y| {
            (0..a.width)
                .map(|x| buf.cell((x, y)).map(|c| c.symbol()).unwrap_or(" "))
                .collect::<String>()
                .trim_end()
                .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Every panel, every variant, with a fully-populated state.
#[test]
fn panels_render_their_content() {
    let now = Instant::now();
    let st = fixture::two_bots(now);
    for p in ALL {
        let spec = p.spec();
        for v in spec.variants {
            let buf = draw(spec.id, v.id, &st, now);
            insta::assert_snapshot!(
                format!("{}__{}", spec.id.0, v.id.name()),
                as_text(&buf)
            );
        }
    }
}

/// And with nothing to show, because "waiting" states are what you look at when
/// something is wrong and they are the ones nobody checks.
#[test]
fn panels_render_their_empty_state() {
    let now = Instant::now();
    let st = fixture::empty();
    for p in ALL {
        let spec = p.spec();
        let v = spec.variants[0];
        let buf = draw(spec.id, v.id, &st, now);
        insta::assert_snapshot!(format!("empty__{}", spec.id.0), as_text(&buf));
    }
}

/// Every colour on screen must be one the theme declares.
///
/// This is the assertion `insta` cannot make. A panel that went entirely the wrong
/// colour, or that hardcoded an RGB instead of taking one from `sf_theme`, passes
/// every text snapshot.
#[test]
fn every_rendered_colour_is_in_the_palette() {
    let now = Instant::now();
    let allowed: BTreeSet<(u8, u8, u8)> = sf_theme::ALL
        .iter()
        // The picture's six are lichess's board colours, deliberately not gruvbox,
        // and are exempt by name rather than by accident -- see sf_theme::picture.
        .chain(sf_theme::picture::ALL)
        .map(|i| i.rgb)
        .collect();

    let mut strays: BTreeSet<(String, (u8, u8, u8))> = BTreeSet::new();
    for st in [fixture::two_bots(now), fixture::empty()] {
        for p in ALL {
            let spec = p.spec();
            for v in spec.variants {
                let buf = draw(spec.id, v.id, &st, now);
                for cell in buf.content() {
                    for c in [cell.fg, cell.bg] {
                        match c {
                            Color::Reset => {}
                            Color::Rgb(r, g, b) if allowed.contains(&(r, g, b)) => {}
                            Color::Rgb(r, g, b) => {
                                strays.insert((spec.id.0.to_string(), (r, g, b)));
                            }
                            other => {
                                // An indexed or named colour cannot be checked
                                // against the palette and will not match the rest of
                                // the screen on a differently-themed terminal.
                                strays.insert((
                                    format!("{} (non-rgb {other:?})", spec.id.0),
                                    (0, 0, 0),
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        strays.is_empty(),
        "colours outside the theme: {strays:?}\n\
         Every ink must come from sf_theme, or the screen stops being one system."
    );
}

/// Colour, in a form `insta` can diff.
///
/// One character per named ink, so the snapshot is a picture of where each colour
/// went. This is what catches a panel losing its emphasis or going uniformly dim,
/// which a text snapshot passes without noticing.
#[test]
fn panels_use_the_inks_they_meant_to() {
    let now = Instant::now();
    let st = fixture::two_bots(now);
    for p in ALL {
        let spec = p.spec();
        let v = spec.variants[0];
        let buf = draw(spec.id, v.id, &st, now);
        let a = buf.area;
        let map = (0..a.height)
            .map(|y| {
                (0..a.width)
                    .map(|x| {
                        buf.cell((x, y)).map(|c| ink_initial(c.fg)).unwrap_or(' ')
                    })
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n");
        insta::assert_snapshot!(format!("inks__{}", spec.id.0), map);
    }
}

/// One stable character per ink. Uppercase for the loud ones, so a glance at the map
/// shows where the eye is being pulled.
fn ink_initial(c: Color) -> char {
    let Color::Rgb(r, g, b) = c else { return '.' };
    let rgb = (r, g, b);
    for (ink, ch) in [
        (sf_theme::ACCENT, 'A'),
        (sf_theme::BAD, 'X'),
        (sf_theme::GOOD, 'G'),
        (sf_theme::WARM, 'W'),
        (sf_theme::INFO, 'I'),
        (sf_theme::COOL, 'C'),
        (sf_theme::PLUM, 'P'),
        (sf_theme::EVAL_FILL, 'E'),
        (sf_theme::FG, 'f'),
        (sf_theme::DIM, 'd'),
        (sf_theme::FAINT, '-'),
        (sf_theme::BG, ' '),
    ] {
        if ink.rgb == rgb {
            return ch;
        }
    }
    '?'
}

/// A panel must never draw outside the rect it was given. The layout gate proves the
/// rects do not overlap; this proves the panels respect them.
#[test]
fn no_panel_draws_outside_its_rect() {
    let now = Instant::now();
    let st = fixture::two_bots(now);
    for p in ALL {
        let spec = p.spec();
        for v in spec.variants {
            // Render into a buffer LARGER than the rect, then require the margin to
            // be untouched.
            let inner = area_for(v.id);
            let outer = Rect { x: 0, y: 0, width: inner.width + 6, height: inner.height + 3 };
            let mut buf = Buffer::empty(outer);
            let cx = Cx {
                state: &st,
                now,
                wall: fixture::wall(),
                bot: None,
                frame: 0,
                picture: None,
            };
            p.render(v.id, inner, &mut buf, &cx);
            for y in 0..outer.height {
                for x in 0..outer.width {
                    let inside = x < inner.width && y < inner.height;
                    if inside {
                        continue;
                    }
                    let cell = buf.cell((x, y)).expect("in bounds");
                    assert_eq!(
                        cell.symbol(),
                        " ",
                        "{} ({}) wrote {:?} at ({x},{y}), outside its {}x{} rect",
                        spec.id,
                        v.id.name(),
                        cell.symbol(),
                        inner.width,
                        inner.height
                    );
                }
            }
        }
    }
}
