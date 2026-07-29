//! The search, as it happens. The most watchable panel there is.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use sf_layout::widgets::{Rung, bar, chrome_styled, ladder, put, track_tag, truncate};
use sf_layout::{
    Cx, Panel, PanelId, PanelSpec, RegionId, Relevance, Scope, Size, Variant, VariantId, Weight,
};
use sf_theme::{Styles, Track};

pub static PANEL: Mind = Mind;
pub struct Mind;

static SPEC: PanelSpec = PanelSpec {
    id: PanelId("mind"),
    title: "engine search",
    region: RegionId::Rail,
    scope: Scope::FocusedBot,
    weight: Weight::High,
    variants: &[
        Variant::new(VariantId::Full, Size::new(46, 12), Size::new(66, 18), 2),
        Variant::new(VariantId::Compact, Size::new(30, 6), Size::new(52, 8), 1),
        Variant::fixed(VariantId::Line, 16, 1),
    ],
};

impl Panel for Mind {
    fn spec(&self) -> &'static PanelSpec {
        &SPEC
    }

    fn relevance(&self, cx: &Cx<'_>) -> Relevance {
        match cx.game().and_then(|g| g.search.as_ref()) {
            Some(s) if !s.done => Relevance::Urgent,
            Some(_) => Relevance::Normal,
            None => Relevance::Idle,
        }
    }

    fn render(&self, variant: VariantId, area: Rect, buf: &mut Buffer, cx: &Cx<'_>) {
        let search = cx.game().and_then(|g| g.search.as_ref());
        let thinking = search.is_some_and(|s| !s.done);

        if variant == VariantId::Line {
            let text = match search {
                Some(s) => format!(
                    "{} {} {}",
                    if thinking { "\u{25b8}" } else { "\u{25aa}" },
                    s.best.as_deref().unwrap_or("-"),
                    s.sims
                ),
                None => "engine idle".into(),
            };
            put(buf, area, area.y, Line::from(Span::styled(truncate(&text, area.width as usize), Styles::dim())));
            return;
        }

        // The border carries the state too, so the panel reads as active from the
        // corner of the eye without parsing any text.
        let border = if thinking { Styles::accent() } else { Styles::border() };
        let inner = chrome_styled(area, SPEC.title, border, buf);
        if inner.height == 0 {
            return;
        }
        let Some(s) = search else {
            put(buf, inner, inner.y, Line::from(Span::styled("waiting for the engine to move", Styles::dim())));
            if inner.height > 1 {
                put(buf, inner, inner.y + 1, Line::from(Span::styled("logs/engine.jsonl", Styles::faint())));
            }
            return;
        };
        let track = cx
            .game()
            .map(|_| if thinking { Track::Live } else { Track::Coast })
            .unwrap_or(Track::Lost);

        // Row 1: the annunciator. State is carried by position and colour, not by
        // text you have to read.
        let ours = cx.game().map(|g| g.ours(s.wp_white)).unwrap_or(0.5);
        let mut row = vec![
            Span::styled(
                if thinking { " SEARCHING " } else { "   IDLE    " },
                Style::new()
                    .fg(sf_theme::BG.color())
                    .bg(if thinking { sf_theme::ACCENT.color() } else { sf_theme::FAINT.color() })
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(
                format!("{:5.1}%", ours * 100.0),
                Styles::body().add_modifier(Modifier::BOLD),
            ),
            // Explicitly OUR frame, and said so. The Python showed the raw
            // side-to-move number here and labelled it "to move"; converting once
            // through `ours` and saying "us" cannot be misread.
            Span::styled(" us", Styles::dim()),
        ];
        if s.mate {
            row.push(Span::styled(
                "   MATE IN LINE ",
                Style::new()
                    .fg(sf_theme::BG.color())
                    .bg(sf_theme::BAD.color())
                    .add_modifier(Modifier::BOLD),
            ));
        }
        row.push(Span::raw("   "));
        row.push(track_tag(track, ""));
        put(buf, inner, inner.y, Line::from(row));

        // Row 2: the time budget, DRAINING. An overrun is the one unrecoverable
        // failure on a clock, so the bar going red before it empties is the point.
        if inner.height > 1 {
            let used = s.elapsed.as_secs_f64();
            let budget = s.budget.as_secs_f64().max(1e-9);
            let frac = (used / budget).min(1.0);
            let w = gauge_w(inner.width);
            let ink = if frac > 0.9 { sf_theme::BAD } else { sf_theme::COOL };
            let mut spans = vec![Span::styled("time  ", Styles::dim())];
            spans.extend(bar(1.0 - frac, w, ink, sf_theme::BG_SOFT).spans);
            spans.push(Span::styled(format!(" {used:5.2}s of {budget:.2}s"), Styles::dim()));
            if used > budget {
                // Flagging is the whole reason to show this: 9 of 324 moves in one
                // day overran, worst by 0.81s.
                spans.push(Span::styled("  OVER", Styles::ink(sf_theme::BAD)));
            }
            put(buf, inner, inner.y + 1, Line::from(spans));
        }

        // Row 3: counters. `unique/s`, never `nps` -- PHILOSOPHY states that as an
        // imperative, and the Python printed `nps` right beside `sims`, whose value
        // is within 15% of it, which is exactly the confusion the rule prevents.
        if inner.height > 2 {
            let dedup = if s.sims > 0 { s.nodes as f64 / s.sims as f64 } else { 0.0 };
            put(
                buf,
                inner,
                inner.y + 2,
                Line::from(vec![
                    Span::styled("nodes ", Styles::dim()),
                    Span::styled(thousands(s.nodes), Styles::body()),
                    Span::styled("   unique/s ", Styles::dim()),
                    Span::styled(thousands(s.unique_per_s), Styles::body()),
                    Span::styled("   sims ", Styles::dim()),
                    Span::styled(thousands(s.sims), Styles::body()),
                    Span::styled("   dedup ", Styles::dim()),
                    Span::styled(format!("{:.0}%", (1.0 - dedup) * 100.0), Styles::body()),
                ]),
            );
        }

        // The engine changing its mind is the most watchable thing the 6Hz feed
        // carries, and the Python threw every frame of it away.
        if inner.height > 3 && s.best_changes > 0 {
            put(
                buf,
                inner,
                inner.y + 3,
                Line::from(vec![
                    Span::styled("changed its mind ", Styles::dim()),
                    Span::styled(format!("{}\u{00d7}", s.best_changes), Styles::ink(sf_theme::WARM)),
                    match s.reused {
                        // Visits inherited by re-rooting: a headline optimisation
                        // with zero observability until the engine emits it.
                        Some(r) if r > 0 => Span::styled(
                            format!("    reused {}", thousands(r)),
                            Styles::ink(sf_theme::COOL),
                        ),
                        _ => Span::raw(""),
                    },
                ]),
            );
        }

        if variant == VariantId::Compact {
            return;
        }

        // The ladder, ordered by visits.
        let head = 5u16;
        let room = inner.height.saturating_sub(head + 2);
        if room > 0 && !s.top.is_empty() {
            let rungs: Vec<Rung> = s
                .top
                .iter()
                .take(room as usize)
                .map(|c| Rung { san: &c.san, visits: c.visits, q: c.q, prior: c.prior })
                .collect();
            for (i, line) in ladder(&rungs, gauge_w(inner.width), sf_theme::ACCENT).into_iter().enumerate() {
                put(buf, inner, inner.y + head + i as u16, line);
            }
        }

        // The principal variation, last.
        let pv_y = inner.y + inner.height.saturating_sub(1);
        if !s.pv.is_empty() && pv_y > inner.y + head {
            let mut spans = vec![Span::styled("pv  ", Styles::dim())];
            for (i, san) in s.pv.iter().enumerate() {
                spans.push(Span::styled(
                    format!("{san} "),
                    if i == 0 { Styles::ink(sf_theme::PLUM) } else { Styles::body() },
                ));
            }
            put(buf, inner, pv_y, Line::from(spans));
        }
    }
}

/// A gauge is a comparison, not a decoration: past `GAUGE_MAX` the label is so far
/// from the fill that the eye cannot associate the two.
fn gauge_w(inner_width: u16) -> u16 {
    sf_theme::GAUGE_MAX.min(inner_width.saturating_sub(34).max(8))
}

/// Thousands separators, because a seven-digit node count is unreadable without.
fn thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn thousands_groups_from_the_right() {
        assert_eq!(super::thousands(0), "0");
        assert_eq!(super::thousands(999), "999");
        assert_eq!(super::thousands(1000), "1,000");
        assert_eq!(super::thousands(38708), "38,708");
        assert_eq!(super::thousands(1234567), "1,234,567");
    }
}
