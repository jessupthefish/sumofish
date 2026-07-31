//! The keybinding hint bar: the answer to "how would you know `?` exists".
//!
//! Everything else in this dashboard is discoverable by looking -- a panel that
//! goes quiet says so, and a dropped panel now has a reason (see `?`, the help
//! overlay). Keybindings are the one thing that cannot be inferred by looking at
//! the screen, so unlike the overlay this cannot itself be gated behind a
//! keypress: it is pinned, full-width, one row, present at every tier including
//! `Micro`.
//!
//! Region `Footer` exists only for this panel. It could not reuse `Band` (the
//! header's region): two `Band` nodes in the same composition tree -- one at the
//! top for the header, one at the bottom for this -- would be indistinguishable to
//! the solver, since panels are bucketed into a region purely by matching
//! `RegionId`, and both instances would collect every `Band`-declared panel as
//! candidates for both rects.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::text::Span;
use sf_layout::widgets::{fit, put};
use sf_layout::{Cx, Panel, PanelId, PanelSpec, RegionId, Relevance, Scope, Variant, VariantId, Weight};
use sf_theme::Styles;

pub static PANEL: Footer = Footer;

pub struct Footer;

/// The real key set, spelled out once. `run.rs`'s `match` on `KeyCode` is the
/// other half of this contract -- if a key is added there and not here, or vice
/// versa, this is the one place both a human and the help overlay's registry sweep
/// would look first. `pub` so the help overlay (`sf-app`) renders the identical
/// list rather than keeping its own copy that could drift from this one.
pub const KEYS: &[(&str, &str)] = &[
    ("q", "quit"),
    ("^C", "quit"),
    ("r", "redraw"),
    ("Tab", "cycle bot"),
    ("1-9", "select bot"),
    ("?", "help"),
];

static SPEC: PanelSpec = PanelSpec {
    id: PanelId("footer"),
    title: "keys",
    keybind: None,
    help: "the real key set, always on screen so ? never has to be guessed at",
    region: RegionId::Footer,
    scope: Scope::Global,
    // Pinned, like the header: without it there is nothing on screen that says
    // `?` exists, which is the entire problem this panel exists to fix.
    weight: Weight::Pinned,
    variants: &[
        // Every binding, spelled out. 59 columns of actual content; 60 leaves a
        // one-column margin so "eligible" also means "will not truncate".
        Variant::fixed(VariantId::Full, 60, 1),
        // `^C` dropped -- `q` already covers quit, and every terminal user knows
        // Ctrl-C regardless of whether this bar says so. 44 columns of content.
        Variant::fixed(VariantId::Compact, 45, 1),
        // The one binding that matters most once everything else has been
        // dropped: how to find out about the rest.
        Variant::fixed(VariantId::Glyph, 8, 1),
    ],
};

impl Panel for Footer {
    fn spec(&self) -> &'static PanelSpec {
        &SPEC
    }

    fn relevance(&self, _cx: &Cx<'_>) -> Relevance {
        Relevance::Normal
    }

    fn render(&self, variant: VariantId, area: Rect, buf: &mut Buffer, _cx: &Cx<'_>) {
        let pairs: &[(&str, &str)] = match variant {
            VariantId::Full => KEYS,
            VariantId::Compact => &KEYS[2..], // drop the redundant q/^C second line
            _ => &KEYS[5..],                  // just "? help"
        };
        let mut spans: Vec<Span<'static>> = Vec::new();
        for (i, (key, what)) in pairs.iter().enumerate() {
            if i > 0 {
                spans.push(Span::styled(" ", Styles::faint()));
            }
            spans.push(Span::styled(*key, Styles::accent()));
            spans.push(Span::styled(format!(" {what}"), Styles::dim()));
        }
        // `fit`, not a plain `put`: this is a list of bindings, and a binding
        // clipped mid-name (`Tab cycle b`) reads as though that is the whole
        // binding rather than a casualty of a narrow terminal.
        put(buf, area, area.y, fit(spans, area.width));
    }
}
