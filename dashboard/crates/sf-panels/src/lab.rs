//! The lab: the largest thing happening on the machine, and invisible until now.
//!
//! Nine of fifteen jobs done, nine hours in, a forty-hour training run in flight and
//! decisions being taken unattended that gate what gets built next -- and no source
//! in `scripts/dash/` so much as opened `runs/lab/`.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use sf_layout::widgets::{chrome, put, track_tag, truncate};
use sf_layout::{
    Cx, Panel, PanelId, PanelSpec, RegionId, Relevance, Scope, Size, Variant, VariantId, Weight,
};
use sf_model::state::LabStatus;
use sf_theme::Styles;

pub static PANEL: Lab = Lab;
pub struct Lab;

static SPEC: PanelSpec = PanelSpec {
    id: PanelId("lab"),
    // The second rail, which is what the `Wide` tier exists for.
    region: RegionId::RailB,
    title: "lab",
    keybind: None,
    help: "the largest thing happening on the machine",
    scope: Scope::Global,
    weight: Weight::Normal,
    variants: &[
        Variant::new(VariantId::Full, Size::new(40, 6), Size::new(60, 20), 2),
        Variant::new(VariantId::Compact, Size::new(24, 3), Size::new(40, 8), 1),
        Variant::fixed(VariantId::Line, 12, 1),
    ],
};

impl Panel for Lab {
    fn spec(&self) -> &'static PanelSpec {
        &SPEC
    }

    fn relevance(&self, cx: &Cx<'_>) -> Relevance {
        match cx.state.lab.queue.get(cx.now) {
            Some((l, _)) if l.current.is_some() => Relevance::Urgent,
            Some((l, _)) if !l.jobs.is_empty() => Relevance::Normal,
            _ => Relevance::Hidden,
        }
    }

    fn render(&self, variant: VariantId, area: Rect, buf: &mut Buffer, cx: &Cx<'_>) {
        let Some((lab, track)) = cx.state.lab.queue.get(cx.now) else { return };
        let done = lab.jobs.iter().filter(|j| j.status == LabStatus::Done).count();

        if variant == VariantId::Line {
            put(buf, area, area.y, Line::from(Span::styled(format!("lab {done}/{}", lab.jobs.len()), Styles::dim())));
            return;
        }
        let inner = chrome(area, SPEC.title, buf);
        if inner.height == 0 {
            return;
        }
        let mut y = inner.y;
        put(
            buf,
            inner,
            y,
            Line::from(vec![
                Span::styled(format!("{done}/{} jobs  ", lab.jobs.len()), Styles::body()),
                Span::styled(
                    lab.current.clone().unwrap_or_else(|| "idle".into()),
                    Styles::ink(sf_theme::ACCENT),
                ),
                Span::raw("  "),
                track_tag(Cx::track(track), ""),
            ]),
        );
        y += 1;

        // The file `lab.py status` tells you to go and read is currently older than
        // the state it reports.
        if lab.report_stale && y < inner.y + inner.height {
            put(
                buf,
                inner,
                y,
                Line::from(Span::styled("report.md is stale", Styles::ink(sf_theme::WARM))),
            );
            y += 1;
        }

        for j in &lab.jobs {
            if y >= inner.y + inner.height {
                break;
            }
            let (glyph, style) = match j.status {
                LabStatus::Done => ("\u{2713}", Styles::ink(sf_theme::GOOD)),
                LabStatus::Running => ("\u{25b8}", Styles::accent()),
                LabStatus::Failed => ("\u{2717}", Styles::ink(sf_theme::BAD)),
                LabStatus::Pending => ("\u{00b7}", Styles::faint()),
            };
            let detail = j.detail.clone().unwrap_or_default();
            put(
                buf,
                inner,
                y,
                Line::from(vec![
                    Span::styled(format!("{glyph} "), style),
                    Span::styled(
                        truncate(&j.id, 18),
                        if j.status == LabStatus::Running {
                            Styles::body().add_modifier(Modifier::BOLD)
                        } else {
                            Styles::dim()
                        },
                    ),
                    Span::styled(
                        format!("  {}", truncate(&detail, inner.width.saturating_sub(22) as usize)),
                        Styles::faint(),
                    ),
                ]),
            );
            y += 1;
        }

        // The derived constants that gate later jobs. `scale_bar: 265.0` decided
        // whether forty GPU-hours were worth spending.
        if variant == VariantId::Full && !lab.facts.is_empty() && y + 1 < inner.y + inner.height {
            y += 1;
            for (k, v) in lab.facts.iter().take((inner.y + inner.height - y) as usize) {
                put(
                    buf,
                    inner,
                    y,
                    Line::from(vec![
                        Span::styled(format!("{k} "), Styles::dim()),
                        Span::styled(truncate(v, 20), Styles::ink(sf_theme::COOL)),
                    ]),
                );
                y += 1;
            }
        }
    }
}
