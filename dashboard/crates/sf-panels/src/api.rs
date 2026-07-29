//! The lichess budget, on screen.
//!
//! Here because the whole multi-bot story is that the effective poll interval
//! degrades as bots are added, and the user should be able to *see* that rather than
//! infer it from a bot that has gone quiet. The dashboard also shares an IP with the
//! bot, and a throttled bot stops playing.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use sf_layout::widgets::{bar, chrome, lpad, put};
use sf_layout::{
    Cx, Panel, PanelId, PanelSpec, RegionId, Relevance, Scope, Size, Variant, VariantId, Weight,
};
use sf_theme::Styles;

pub static PANEL: ApiPanel = ApiPanel;
pub struct ApiPanel;

static SPEC: PanelSpec = PanelSpec {
    id: PanelId("api"),
    title: "lichess budget",
    region: RegionId::RailB,
    scope: Scope::Global,
    weight: Weight::Low,
    variants: &[
        Variant::new(VariantId::Full, Size::new(38, 4), Size::new(52, 8), 0),
        Variant::new(VariantId::Compact, Size::new(20, 2), Size::new(38, 4), 0),
        Variant::fixed(VariantId::Line, 10, 1),
    ],
};

impl Panel for ApiPanel {
    fn spec(&self) -> &'static PanelSpec {
        &SPEC
    }

    fn relevance(&self, cx: &Cx<'_>) -> Relevance {
        if cx.state.api.endpoints.is_empty() {
            Relevance::Hidden
        } else if cx.state.api.throttled > 0 {
            Relevance::Urgent
        } else {
            Relevance::Idle
        }
    }

    fn render(&self, variant: VariantId, area: Rect, buf: &mut Buffer, cx: &Cx<'_>) {
        let api = &cx.state.api;
        if variant == VariantId::Line {
            put(buf, area, area.y, Line::from(Span::styled(format!("api {}/min", api.total_last_minute), Styles::dim())));
            return;
        }
        let inner = chrome(area, SPEC.title, buf);
        if inner.height == 0 {
            return;
        }
        let mut y = inner.y;
        for (name, e) in api.endpoints.iter().take(inner.height as usize) {
            let frac = e.calls_last_minute as f64 / e.rpm_budget.max(1) as f64;
            let ink = if frac > 0.9 { sf_theme::BAD } else { sf_theme::COOL };
            let mut row = vec![Span::styled(lpad(name, 16), Styles::dim())];
            row.extend(bar(frac, 10, ink, sf_theme::BG_SOFT).spans);
            row.push(Span::styled(
                format!(" {}/{}", e.calls_last_minute, e.rpm_budget),
                Styles::faint(),
            ));
            // The honest answer to "add more bots as it grows": the interval each
            // source ASKED for, against the one it is actually getting.
            if let (Some(want), Some(eff)) = (e.wanted_interval, e.effective_interval) {
                if eff > want {
                    row.push(Span::styled(
                        format!("  {:.0}s (wanted {:.0}s)", eff.as_secs_f64(), want.as_secs_f64()),
                        Styles::ink(sf_theme::WARM),
                    ));
                }
            }
            if e.penalty_until.is_some() {
                row.push(Span::styled("  BACKING OFF", Styles::ink(sf_theme::BAD)));
            }
            put(buf, inner, y, Line::from(row));
            y += 1;
        }
        if api.throttled > 0 && y < inner.y + inner.height {
            put(
                buf,
                inner,
                y,
                Line::from(Span::styled(
                    format!("{} throttled \u{2014} the bot shares this IP", api.throttled),
                    Styles::ink(sf_theme::BAD),
                )),
            );
        }
    }
}
