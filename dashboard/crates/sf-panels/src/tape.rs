//! The event log: things as they happened.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use sf_layout::widgets::{chrome, lpad, put, truncate};
use sf_layout::{
    Cx, Panel, PanelId, PanelSpec, RegionId, Relevance, Scope, Size, Variant, VariantId, Weight,
};
use sf_model::state::TapeKind;
use sf_theme::Styles;

pub static PANEL: Tape = Tape;
pub struct Tape;

static SPEC: PanelSpec = PanelSpec {
    id: PanelId("tape"),
    title: "event log",
    keybind: None,
    help: "things as they happened",
    // Under the board, which is where the Python put it and where the eye is
    // already looking.
    region: RegionId::Board,
    scope: Scope::Global,
    weight: Weight::Low,
    variants: &[
        Variant::new(VariantId::Full, Size::new(40, 5), Size::new(90, 12), 1),
        Variant::new(VariantId::Compact, Size::new(20, 3), Size::new(40, 6), 1),
        Variant::fixed(VariantId::Line, 12, 1),
    ],
};

impl Panel for Tape {
    fn spec(&self) -> &'static PanelSpec {
        &SPEC
    }

    fn relevance(&self, cx: &Cx<'_>) -> Relevance {
        if cx.state.tape.is_empty() { Relevance::Idle } else { Relevance::Normal }
    }

    fn render(&self, variant: VariantId, area: Rect, buf: &mut Buffer, cx: &Cx<'_>) {
        if variant == VariantId::Line {
            let last = cx.state.tape.tail(1).next().map(|e| e.text.clone()).unwrap_or_default();
            put(buf, area, area.y, Line::from(Span::styled(truncate(&last, area.width as usize), Styles::faint())));
            return;
        }
        let inner = chrome(area, SPEC.title, buf);
        if inner.height == 0 {
            return;
        }
        let rows = inner.height as usize;
        let entries: Vec<_> = cx.state.tape.tail(rows).collect();
        if entries.is_empty() {
            put(buf, inner, inner.y, Line::from(Span::styled("nothing yet", Styles::faint())));
            return;
        }
        for (i, e) in entries.iter().enumerate() {
            // A fixed wall clock in `Cx` is what makes this snapshot-testable; the
            // Python called `time.strftime` here, so no two frames could be compared.
            let hhmmss = e.at.strftime("%H:%M:%S").to_string();
            let (label, style) = kind_style(e.kind);
            put(
                buf,
                inner,
                inner.y + i as u16,
                Line::from(vec![
                    Span::styled(format!("{hhmmss} "), Styles::faint()),
                    Span::styled(lpad(label, 7), style),
                    Span::styled(
                        truncate(&e.text, inner.width.saturating_sub(17) as usize),
                        Styles::body(),
                    ),
                ]),
            );
        }
    }
}

fn kind_style(k: TapeKind) -> (&'static str, ratatui::style::Style) {
    match k {
        TapeKind::Move => ("move", Styles::accent()),
        TapeKind::Game => ("game", Styles::ink(sf_theme::INFO)),
        TapeKind::Engine => ("engine", Styles::ink(sf_theme::PLUM)),
        TapeKind::Train => ("train", Styles::ink(sf_theme::COOL)),
        TapeKind::Lab => ("lab", Styles::ink(sf_theme::COOL)),
        TapeKind::Warn => ("warn", Styles::ink(sf_theme::WARM)),
        TapeKind::Error => ("error", Styles::ink(sf_theme::BAD)),
    }
}
