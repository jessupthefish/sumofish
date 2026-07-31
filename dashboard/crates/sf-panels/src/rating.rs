//! Rating history: the trend, not just the latest point `header` already shows.
//!
//! `header` prints the current rating per control, sourced from `/api/account`.
//! This reads `logs/rating.jsonl` directly through `BotState.rating_log` --
//! already proven safe for concurrent reads, and already polled by the same
//! `Tracked` mechanism `header` uses for everything else -- and plots where the
//! rating has been, plus which checkpoint was deployed at each sample. That
//! attribution is the file's own stated purpose (`rating_log.py` records it
//! "alongside what changed, so a jump can be attributed rather than guessed
//! at") and the one thing a single current number cannot show. This is the
//! panel that absorbed `sumofish rating`.
//!
//! One-directional pointer, not a merge: a footer hint at Chess DB's own
//! `/stats` page, which reads this *same* file directly (no API, no new
//! infrastructure -- see chess-db's `stats.sumofish_rating_history`) and
//! shows every sample across every time control, unbounded by this panel's
//! 500-sample file cap or its on-screen sparkline width. The live-process
//! watcher wanting historical context is a plausible workflow; nothing
//! points the other way. Cheap and reversible: pull it back out if it goes
//! unused.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use sf_layout::widgets::{chrome, fit, put, sparkline, track_tag, truncate};
use sf_layout::{
    Cx, Panel, PanelId, PanelSpec, RegionId, Relevance, Scope, Size, Variant, VariantId, Weight,
};
use sf_theme::Styles;

pub static PANEL: Rating = Rating;
pub struct Rating;

/// Chess DB's actual verified LAN bind (gunicorn `-b 0.0.0.0:8000`, confirmed
/// live 2026-07-30) -- not `spaceship.local`, which resolves to AAAA records
/// only over mDNS on this network and gunicorn does not listen on IPv6, so
/// that hostname silently fails to connect from any device but this one.
const CHESS_DB_STATS_URL: &str = "http://10.0.0.106:8000/stats";

static SPEC: PanelSpec = PanelSpec {
    id: PanelId("rating"),
    title: "rating",
    keybind: None,
    help: "rating history over time, with which checkpoint was deployed at each sample",
    region: RegionId::RailB,
    scope: Scope::FocusedBot,
    weight: Weight::Normal,
    variants: &[
        Variant::new(VariantId::Full, Size::new(40, 5), Size::new(60, 8), 0),
        Variant::new(VariantId::Compact, Size::new(24, 3), Size::new(40, 5), 0),
        Variant::fixed(VariantId::Line, 12, 1),
    ],
};

impl Panel for Rating {
    fn spec(&self) -> &'static PanelSpec {
        &SPEC
    }

    fn relevance(&self, cx: &Cx<'_>) -> Relevance {
        match cx.bot_state().and_then(|b| b.rating_log.get(cx.now)) {
            Some((samples, _)) if !samples.is_empty() => Relevance::Normal,
            _ => Relevance::Hidden,
        }
    }

    fn render(&self, variant: VariantId, area: Rect, buf: &mut Buffer, cx: &Cx<'_>) {
        let Some(bot) = cx.bot_state() else { return };
        let Some((samples, track)) = bot.rating_log.get(cx.now) else { return };
        let Some(latest) = samples.last() else { return };

        if variant == VariantId::Line {
            let text = latest
                .perfs
                .iter()
                .next()
                .map(|(k, p)| format!("{k} {}", p.rating))
                .unwrap_or_else(|| "rating --".into());
            put(
                buf,
                area,
                area.y,
                Line::from(Span::styled(truncate(&text, area.width as usize), Styles::dim())),
            );
            return;
        }

        let inner = chrome(area, SPEC.title, buf);
        if inner.height == 0 {
            return;
        }
        let mut y = inner.y;
        let w = sf_theme::GAUGE_MAX.min(inner.width.saturating_sub(28).max(6));

        for (control, perf) in latest.perfs.iter() {
            if y >= inner.y + inner.height {
                break;
            }
            let series: Vec<f64> = samples
                .iter()
                .filter_map(|s| s.perfs.get(control).map(|p| p.rating as f64))
                .collect();
            let mut row = vec![
                Span::styled(format!("{control:<9}"), Styles::dim()),
                Span::styled(format!("{:>5}", perf.rating), Styles::accent()),
            ];
            // A provisional rating and a settled one are different claims -- the
            // same distinction `header` already draws.
            if perf.provisional {
                row.push(Span::styled("?", Styles::ink(sf_theme::WARM)));
            } else if perf.rd > 0 {
                row.push(Span::styled(format!("\u{00b1}{}", perf.rd), Styles::faint()));
            }
            row.push(Span::styled(format!("  {}n", perf.games), Styles::faint()));
            if series.len() >= 2 && variant == VariantId::Full {
                let (spark, _lo, _hi) = sparkline(&series, w, sf_theme::INFO);
                row.push(Span::raw("  "));
                row.extend(spark.spans);
                let delta = series.last().unwrap() - series.first().unwrap();
                row.push(Span::styled(
                    format!(" {delta:+.0}"),
                    if delta >= 0.0 { Styles::ink(sf_theme::GOOD) } else { Styles::ink(sf_theme::BAD) },
                ));
            }
            put(buf, inner, y, Line::from(row));
            y += 1;
        }

        // The record, moved here from `header` 2026-07-30: `header` is pinned and
        // shows regardless of which bot is focused, which made it the wrong home
        // for a number scoped to one bot's lifetime. This panel already carries
        // that scoping (`Scope::FocusedBot`) and the per-control ratings the
        // record sits next to, so it belongs here, not duplicated in both places.
        if let Some((rec, _)) = bot.record.get(cx.now) {
            if y < inner.y + inner.height {
                put(
                    buf,
                    inner,
                    y,
                    Line::from(vec![
                        Span::styled(format!("{}W", rec.wins), Styles::ink(sf_theme::GOOD)),
                        Span::styled(format!(" {}D", rec.draws), Styles::dim()),
                        Span::styled(format!(" {}L", rec.losses), Styles::ink(sf_theme::BAD)),
                    ]),
                );
                y += 1;
            }
        }

        if y < inner.y + inner.height {
            put(
                buf,
                inner,
                y,
                Line::from(vec![
                    Span::styled(format!("{} samples  ", samples.len()), Styles::faint()),
                    track_tag(Cx::track(track), ""),
                ]),
            );
            y += 1;
        }

        // The file's own stated purpose: a jump can be attributed to what was
        // actually deployed at the time, rather than guessed at.
        if y < inner.y + inner.height {
            let mut row = vec![Span::styled("deployed  ", Styles::dim())];
            match &latest.deployed {
                Some(d) => {
                    let mut any = false;
                    for (label, sha) in
                        [("engine", &d.engine), ("policy", &d.policy), ("value", &d.value), ("current", &d.current)]
                    {
                        if let Some(sha) = sha {
                            any = true;
                            row.push(Span::styled(format!("{label} "), Styles::faint()));
                            row.push(Span::styled(format!("{} ", truncate(sha, 8)), Styles::body()));
                        }
                    }
                    if !any {
                        row.push(Span::styled("unknown", Styles::faint()));
                    }
                }
                None => row.push(Span::styled("unknown", Styles::faint())),
            }
            // `fit`, not a plain `put`: four label/sha pairs routinely overflow a
            // `RailB`-width panel, and a row silently clipped mid-word reads as
            // though `current` were never there rather than as a casualty of
            // width.
            put(buf, inner, y, fit(row, inner.width));
            y += 1;
        }

        // The pilot cross-link: unobtrusive, one line, dim -- same treatment
        // `header` gives its own secondary info (see `Styles::dim` there).
        // Only drawn if there's room left; a hint that clips at the tier
        // where someone most needs it is worse than no hint.
        if y < inner.y + inner.height {
            put(
                buf,
                inner,
                y,
                fit(
                    vec![Span::styled(
                        format!("full history: {CHESS_DB_STATS_URL}"),
                        Styles::dim(),
                    )],
                    inner.width,
                ),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture;
    use ratatui::buffer::Buffer;
    use std::time::Instant;

    fn render(variant: VariantId, area: Rect, now: Instant) -> String {
        let st = fixture::two_bots(now);
        let mut buf = Buffer::empty(area);
        let cx = Cx { state: &st, now, wall: fixture::wall(), bot: None, frame: 0, picture: None };
        PANEL.render(variant, area, &mut buf, &cx);
        (0..area.height)
            .map(|y| {
                (0..area.width)
                    .map(|x| buf.cell((x, y)).map(|c| c.symbol()).unwrap_or(" "))
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The pilot cross-link: a dim footer line pointing at Chess DB's own
    /// `/stats`, which reads this same `rating.jsonl` file directly (no new
    /// infrastructure, no merge -- see the module doc). Full variant has
    /// plenty of room in the fixture's 3-sample history, so it must appear.
    #[test]
    fn full_variant_points_at_chess_db() {
        let now = Instant::now();
        let area = Rect { x: 0, y: 0, width: 62, height: 16 };
        let text = render(VariantId::Full, area, now);
        assert!(
            text.contains("full history: http://10.0.0.106:8000/stats"),
            "expected the Chess DB cross-link on the Full variant:\n{text}"
        );
    }

    /// The Line variant is a single row already fully spent on the current
    /// rating -- the hint must never appear there, or it would silently
    /// replace the one thing that variant exists to show.
    #[test]
    fn line_variant_never_shows_the_hint() {
        let now = Instant::now();
        let area = Rect { x: 0, y: 0, width: 40, height: 1 };
        let text = render(VariantId::Line, area, now);
        assert!(
            !text.contains("full history"),
            "the Line variant has no room for the hint:\n{text}"
        );
    }

    /// A too-narrow Compact rect must drop the hint rather than clip it
    /// mid-word -- same room-check discipline as the `deployed` row above
    /// it, and the exact failure mode the plan's dashboard-A note warns
    /// about ("a hint... never checked at the narrowest will clip or
    /// vanish exactly where a confused user is most likely to need it").
    #[test]
    fn tiny_compact_rect_omits_rather_than_clips() {
        let now = Instant::now();
        // Just tall enough for chrome + two perf rows; no room for the
        // samples/deployed/hint rows underneath.
        let area = Rect { x: 0, y: 0, width: 24, height: 4 };
        let text = render(VariantId::Compact, area, now);
        for line in text.lines() {
            assert!(
                line.chars().count() <= area.width as usize,
                "line overflowed the rect: {line:?}"
            );
        }
        assert!(
            !text.contains("full history"),
            "no room for the hint at this height, it must be omitted, not clipped:\n{text}"
        );
    }
}
