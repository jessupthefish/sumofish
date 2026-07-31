//! The evaluation, as a number, a gauge and a chart that cannot lie about scale.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use sf_layout::widgets::{advantage, chrome, curve_chart, evalbar, put};
use sf_layout::{
    Cx, Panel, PanelId, PanelSpec, RegionId, Relevance, Scope, Size, Variant, VariantId, Weight,
};
use sf_theme::Styles;

pub static PANEL: Curve = Curve;
pub struct Curve;

/// Three columns, not two: at two the gauge is twelve times taller than wide and
/// reads as a rule rather than a bar.
const GAUGE_W: u16 = 3;

static SPEC: PanelSpec = PanelSpec {
    id: PanelId("curve"),
    title: "evaluation",
    keybind: None,
    help: "the evaluation, as a number, a gauge, and a chart that cannot lie about scale",
    region: RegionId::Rail,
    scope: Scope::FocusedBot,
    weight: Weight::Normal,
    variants: &[
        Variant::new(VariantId::Full, Size::new(30, 8), Size::new(60, 24), 3),
        Variant::new(VariantId::Compact, Size::new(14, 4), Size::new(30, 10), 2),
        Variant::fixed(VariantId::Line, 10, 1),
    ],
};

impl Panel for Curve {
    fn spec(&self) -> &'static PanelSpec {
        &SPEC
    }

    fn relevance(&self, cx: &Cx<'_>) -> Relevance {
        match cx.game() {
            Some(g) if !g.curve.is_empty() || g.search.is_some() => Relevance::Normal,
            _ => Relevance::Hidden,
        }
    }

    fn render(&self, variant: VariantId, area: Rect, buf: &mut Buffer, cx: &Cx<'_>) {
        let Some(game) = cx.game() else { return };
        // The engine's OWN curve here, not Stockfish's -- unlike the moves panel.
        // Two instruments disagreeing is information; averaging them is not.
        // White's frame, always, matching every other eval display -- see the
        // doc comment on `GameState::curve` for why this stopped converting to
        // "ours" (2026-07-30).
        let series: Vec<f64> = game.curve.values().map(|v| *v as f64).collect();
        let now = game.search.as_ref().map(|s| s.wp_white as f64).or_else(|| series.last().copied());

        if variant == VariantId::Line {
            let text = now.map(|v| advantage(v)).unwrap_or_else(|| "--".into());
            put(buf, area, area.y, Line::from(Span::styled(format!("eval {text}"), Styles::dim())));
            return;
        }

        let inner = chrome(area, SPEC.title, buf);
        if inner.height == 0 {
            return;
        }

        // The head row shows the UN-EASED number: a figure must never show a value
        // the engine did not produce. The gauge below is eased, so the boundary
        // slides while the number steps.
        let head = match now {
            None => Line::from(Span::styled("     --", Styles::dim())),
            Some(v) => Line::from(vec![
                Span::styled(format!("{:>7}", advantage(v)), Styles::ink(sf_theme::EVAL_FILL)),
                Span::styled(format!("   wp {v:.2}"), Styles::faint()),
            ]),
        };
        put(buf, inner, inner.y, head);

        let body_h = inner.height.saturating_sub(1);
        if body_h == 0 {
            return;
        }
        let gauge = evalbar(now, body_h, GAUGE_W);
        let chart = if series.len() >= 2 {
            curve_chart(
                &series,
                inner.width.saturating_sub(GAUGE_W + 6).max(8),
                body_h,
                0.5,
            )
        } else {
            vec![Line::from(""); body_h as usize]
        };

        for r in 0..body_h as usize {
            let mut spans = Vec::new();
            // The axis is only labelled where it means something.
            spans.push(Span::styled(
                if r == 0 {
                    "1.0 "
                } else if r == body_h as usize / 2 {
                    "0.5 "
                } else if r + 1 == body_h as usize {
                    "0.0 "
                } else {
                    "    "
                },
                Styles::faint(),
            ));
            spans.extend(gauge[r].spans.clone());
            spans.push(Span::raw("  "));
            spans.extend(chart[r].spans.clone());
            put(buf, inner, inner.y + 1 + r as u16, Line::from(spans));
        }
    }
}
