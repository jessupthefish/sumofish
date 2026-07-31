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
    keybind: None,
    help: "the search, as it happens",
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
        // A blank row under the title border, always -- the annunciator sat
        // flush against the rule with nothing to breathe against, reported
        // directly 2026-07-30.
        let pad = 1u16;
        let Some(s) = search else {
            if inner.height > pad {
                put(buf, inner, inner.y + pad, Line::from(Span::styled("waiting for the engine to move", Styles::dim())));
            }
            if inner.height > pad + 1 {
                put(buf, inner, inner.y + pad + 1, Line::from(Span::styled("logs/engine.jsonl", Styles::faint())));
            }
            return;
        };
        let track = cx
            .game()
            .map(|_| if thinking { Track::Live } else { Track::Coast })
            .unwrap_or(Track::Lost);

        // Row 1: the annunciator. State is carried by position and colour, not by
        // text you have to read. Coloured text on the panel's own background,
        // not an inverted pill -- a solid chip in a different colour from
        // every panel around it was the "grey backing" reported 2026-07-30
        // (the ladder bars were the other half of that same report).
        let white_wp = s.wp_white;
        let mut row = vec![
            Span::styled(
                if thinking { " SEARCHING " } else { "   IDLE    " },
                Style::new()
                    .fg(if thinking { sf_theme::ACCENT.color() } else { sf_theme::FAINT.color() })
                    .bg(sf_theme::BG.color())
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(
                format!("{:5.1}%", white_wp * 100.0),
                Styles::body().add_modifier(Modifier::BOLD),
            ),
            // White's frame, always, matching every other eval display in the
            // dashboard (2026-07-30) -- see `GameState::curve`'s doc comment for
            // why this stopped converting to "ours". The Python's predecessor
            // showed the raw side-to-move number and labelled it "to move";
            // this says which side the percentage is FOR, which stays true
            // across the whole game rather than flipping with whoever's turn
            // it is.
            Span::styled(" white", Styles::dim()),
        ];
        if s.mate {
            row.push(Span::styled(
                "   MATE IN LINE ",
                Style::new()
                    .fg(sf_theme::BAD.color())
                    .bg(sf_theme::BG.color())
                    .add_modifier(Modifier::BOLD),
            ));
        }
        row.push(Span::raw("   "));
        row.push(track_tag(track, ""));
        put(buf, inner, inner.y + pad, Line::from(row));

        // Row 2: the time budget, DRAINING. An overrun is the one unrecoverable
        // failure on a clock, so the bar going red before it empties is the point.
        if inner.height > pad + 1 {
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
            put(buf, inner, inner.y + pad + 1, Line::from(spans));
        }

        // Row 3: counters. `unique/s`, never `nps` -- PHILOSOPHY states that as an
        // imperative, and the Python printed `nps` right beside `sims`, whose value
        // is within 15% of it, which is exactly the confusion the rule prevents.
        if inner.height > pad + 2 {
            let dedup = if s.sims > 0 { s.nodes as f64 / s.sims as f64 } else { 0.0 };
            put(
                buf,
                inner,
                inner.y + pad + 2,
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
        let mind_row = inner.height > pad + 3 && s.best_changes > 0;
        if mind_row {
            put(
                buf,
                inner,
                inner.y + pad + 3,
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

        // The ladder, ordered by visits. One blank row of separation from
        // whatever's above it, always -- not a fixed row 5 regardless of
        // whether the "changed its mind" line actually printed, which left a
        // double-blank gap on the common frame (no mind-change yet) and a
        // single one on the rare frame, so the widget's own spacing changed
        // shape from tick to tick for no reason a viewer could see.
        let head = pad + if mind_row { 5u16 } else { 4u16 };
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
    use super::*;
    use ratatui::buffer::Buffer;
    use sf_model::state::{BotConfig, BotState, Candidate, GameState, SearchSnapshot};
    use sf_model::{AppState, BotId, GameId};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    #[test]
    fn thousands_groups_from_the_right() {
        assert_eq!(super::thousands(0), "0");
        assert_eq!(super::thousands(999), "999");
        assert_eq!(super::thousands(1000), "1,000");
        assert_eq!(super::thousands(38708), "38,708");
        assert_eq!(super::thousands(1234567), "1,234,567");
    }

    /// The regression this whole conversion-removal was for: playing Black,
    /// dominating (White at 3% per the engine's own search), must show "3.0%",
    /// not "97.0%" -- the exact real-game confusion that prompted reverting
    /// `ours()`. White's frame, always, every panel, no per-game flip.
    #[test]
    fn white_frame_survives_a_black_dominated_position() {
        let cfg = Arc::new(BotConfig {
            id: BotId("sumofish".into()),
            user: "SumoFish".into(),
            label: String::new(),
            unit: None,
            telemetry: "/dev/null".into(),
            pgn_dir: None,
            versions: None,
            rating_log: None,
            watch: true,
        });
        let mut bot = BotState::new(cfg);
        let mut game = GameState::new(GameId("g1".into()), shakmaty::Color::Black);
        game.search = Some(SearchSnapshot {
            ply: 131,
            nodes: 61_388,
            unique_per_s: 4_783,
            sims: 67_070,
            elapsed: Duration::from_secs(13),
            budget: Duration::from_secs(13),
            wp_white: 0.03,
            best: Some("h3h4".into()),
            pv: vec!["h3h4".into()],
            top: vec![Candidate { san: "h4".into(), visits: 100, q: 0.03, prior: 0.5 }],
            mate: false,
            done: true,
            reused: None,
            best_changes: 0,
        });
        bot.games.insert(game.id.clone(), game);
        bot.focus_game = Some(GameId("g1".into()));
        let mut st = AppState::default();
        let id = bot.cfg.id.clone();
        st.bots.insert(id.clone(), bot);
        st.focus = Some(id);

        let area = Rect { x: 0, y: 0, width: 66, height: 12 };
        let mut buf = Buffer::empty(area);
        let now = Instant::now();
        let cx = Cx { state: &st, now, wall: jiff::Timestamp::UNIX_EPOCH, bot: None, frame: 0, picture: None };
        PANEL.render(VariantId::Full, area, &mut buf, &cx);
        let text = (0..area.height)
            .map(|y| (0..area.width).map(|x| buf.cell((x, y)).map(|c| c.symbol()).unwrap_or(" ")).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(text.contains("3.0% white"), "expected White's real 3% shown as 3%, not flipped to 97%:\n{text}");
        assert!(!text.contains("97.0%"), "the old `ours()` flip must not be reintroduced:\n{text}");

        // The spacing regression this test doubles as coverage for: with no
        // "changed its mind" line (the common frame), the ladder must sit
        // exactly one blank row below the counters, not two.
        let lines: Vec<&str> = text.lines().collect();
        let nodes_row = lines.iter().position(|l| l.contains("nodes")).expect("nodes row");
        let ladder_row = lines.iter().position(|l| l.contains("h4")).expect("ladder row");
        assert_eq!(
            ladder_row - nodes_row,
            2,
            "expected exactly one blank row between the counters and the ladder when there's no mind-change row:\n{text}"
        );
    }
}
