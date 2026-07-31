//! Head-to-head matches: a verdict on a code change, arriving one game at a time.
//!
//! Two rules from PHILOSOPHY are enforced by this panel rather than merely honoured
//! nearby. **A number without an interval is not a result**, so the interval is
//! never optional here. And **a run's provenance is checked**: `sum(game.seconds) <=
//! job.seconds` is physically impossible to violate legitimately, all four rungs of
//! the exchange-rate ladder violate it, and nothing checked until now.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use sf_layout::widgets::{chrome, put, truncate};
use sf_layout::{
    Cx, Panel, PanelId, PanelSpec, RegionId, Relevance, Scope, Size, Variant, VariantId, Weight,
};
use sf_model::state::Provenance;
use sf_theme::Styles;

pub static PANEL: Matches = Matches;
pub struct Matches;

static SPEC: PanelSpec = PanelSpec {
    id: PanelId("matches"),
    title: "matches",
    keybind: None,
    help: "head-to-head matches: a verdict on a code change, one game at a time",
    region: RegionId::RailB,
    scope: Scope::Global,
    weight: Weight::Normal,
    variants: &[
        Variant::new(VariantId::Full, Size::new(44, 5), Size::new(60, 14), 1),
        Variant::new(VariantId::Compact, Size::new(24, 3), Size::new(44, 8), 1),
        Variant::fixed(VariantId::Line, 12, 1),
    ],
};

impl Panel for Matches {
    fn spec(&self) -> &'static PanelSpec {
        &SPEC
    }

    fn relevance(&self, cx: &Cx<'_>) -> Relevance {
        match cx.state.matches.runs.get(cx.now) {
            Some((r, _)) if r.iter().any(|m| m.running) => Relevance::Urgent,
            // A suspect result is worth interrupting for.
            Some((r, _)) if r.iter().any(|m| m.provenance != Provenance::Ok) => Relevance::Urgent,
            Some((r, _)) if !r.is_empty() => Relevance::Normal,
            _ => Relevance::Hidden,
        }
    }

    fn render(&self, variant: VariantId, area: Rect, buf: &mut Buffer, cx: &Cx<'_>) {
        let Some((runs, _)) = cx.state.matches.runs.get(cx.now) else { return };

        if variant == VariantId::Line {
            let live = runs.iter().filter(|m| m.running).count();
            put(buf, area, area.y, Line::from(Span::styled(format!("{} matches, {live} live", runs.len()), Styles::dim())));
            return;
        }
        let inner = chrome(area, SPEC.title, buf);
        if inner.height == 0 {
            return;
        }
        let mut y = inner.y;
        for m in runs.iter().take(inner.height as usize) {
            let mut row = vec![Span::styled(
                format!("{} ", truncate(&m.name, 18)),
                if m.running {
                    Styles::accent().add_modifier(Modifier::BOLD)
                } else {
                    Styles::dim()
                },
            )];
            row.push(Span::styled(format!("{}g ", m.games), Styles::faint()));
            match (m.elo, m.elo_err) {
                // The interval is never optional. A point estimate without one is
                // the thing PHILOSOPHY says is not a result at all.
                (Some(e), Some(err)) => {
                    row.push(Span::styled(
                        format!("{e:+.0}\u{00b1}{err:.0} "),
                        if e.abs() > err { Styles::body() } else { Styles::dim() },
                    ));
                }
                (Some(_), None) => {
                    row.push(Span::styled("no interval ", Styles::ink(sf_theme::WARM)));
                }
                _ => row.push(Span::styled("\u{2014} ", Styles::faint())),
            }
            if let Some(los) = m.los {
                row.push(Span::styled(format!("LOS {:.0}% ", los * 100.0), Styles::faint()));
            }
            // The cheapest total check in the repo, and it currently fails on
            // every published rung.
            match m.provenance {
                Provenance::Impossible => row.push(Span::styled(
                    "REPLAY?",
                    Styles::ink(sf_theme::BAD).add_modifier(Modifier::BOLD),
                )),
                Provenance::Unfingerprinted => {
                    row.push(Span::styled("no code sha", Styles::ink(sf_theme::WARM)))
                }
                Provenance::Unknown => row.push(Span::styled("?", Styles::faint())),
                Provenance::Ok => {}
            }
            put(buf, inner, y, Line::from(row));
            y += 1;
        }
    }
}
