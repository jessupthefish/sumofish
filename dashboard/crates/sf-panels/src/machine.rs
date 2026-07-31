//! The machine: is training starving the bot, and is anything actually up.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use sf_layout::widgets::{bar, chrome, fit, put, track_tag, truncate};
use sf_layout::{
    Cx, Panel, PanelId, PanelSpec, RegionId, Relevance, Scope, Size, Variant, VariantId, Weight,
};
use sf_theme::Styles;

pub static PANEL: Machine = Machine;
pub struct Machine;

static SPEC: PanelSpec = PanelSpec {
    id: PanelId("machine"),
    title: "machine",
    keybind: None,
    help: "is training starving the bot, and is anything actually up",
    region: RegionId::Rail,
    scope: Scope::Global,
    weight: Weight::High,
    variants: &[
        Variant::new(VariantId::Full, Size::new(44, 7), Size::new(64, 10), 1),
        Variant::new(VariantId::Compact, Size::new(24, 3), Size::new(44, 4), 0),
        Variant::fixed(VariantId::Line, 12, 1),
    ],
};

impl Panel for Machine {
    fn spec(&self) -> &'static PanelSpec {
        &SPEC
    }

    /// Urgent when something is actually wrong, so it resists being dropped
    /// exactly when it matters.
    fn relevance(&self, cx: &Cx<'_>) -> Relevance {
        let units = cx.state.machine.units.get(cx.now);
        let broken = units.is_some_and(|(us, _)| {
            us.iter().any(|u| !u.missing() && !u.healthy())
        });
        let restarted = cx
            .state
            .bots
            .values()
            .any(|b| b.watchdog.get(cx.now).is_some_and(|(w, _)| w.restarts_today > 0));
        if broken || restarted { Relevance::Urgent } else { Relevance::Normal }
    }

    fn render(&self, variant: VariantId, area: Rect, buf: &mut Buffer, cx: &Cx<'_>) {
        if variant == VariantId::Line {
            let util = cx
                .state
                .machine
                .gpus
                .get(cx.now)
                .and_then(|(g, _)| g.first().map(|g| g.util_gpu))
                .unwrap_or(0);
            put(buf, area, area.y, Line::from(Span::styled(format!("gpu {util}%"), Styles::dim())));
            return;
        }
        let inner = chrome(area, SPEC.title, buf);
        if inner.height == 0 {
            return;
        }
        let mut y = inner.y;
        let w = sf_theme::GAUGE_MAX.min(inner.width.saturating_sub(40).max(6));

        match cx.state.machine.gpus.get(cx.now) {
            Some((gpus, track)) if !gpus.is_empty() => {
                let g = &gpus[0];
                let mut row = vec![Span::styled("gpu  ", Styles::dim())];
                row.extend(bar(g.util_gpu as f64 / 100.0, w, sf_theme::COOL, sf_theme::BG_SOFT).spans);
                row.push(Span::styled(format!(" {:3}%", g.util_gpu), Styles::body()));
                row.push(Span::styled(
                    format!("  {}\u{00b0}C", g.temp_c),
                    if g.temp_c > 80 { Styles::ink(sf_theme::BAD) } else { Styles::dim() },
                ));
                row.push(Span::styled(
                    format!("  {:.0}/{:.0}W", g.power_w, g.power_limit_w),
                    Styles::dim(),
                ));
                put(buf, inner, y, Line::from(row));
                y += 1;

                if y < inner.y + inner.height {
                    let frac = g.mem_used as f64 / g.mem_total.max(1) as f64;
                    let mut row = vec![Span::styled("vram ", Styles::dim())];
                    row.extend(bar(frac, w, sf_theme::PLUM, sf_theme::BG_SOFT).spans);
                    row.push(Span::styled(
                        format!(
                            " {:.1}/{:.0}G  ",
                            g.mem_used as f64 / 1_073_741_824.0,
                            g.mem_total as f64 / 1_073_741_824.0
                        ),
                        Styles::dim(),
                    ));
                    row.push(track_tag(Cx::track(track), ""));
                    put(buf, inner, y, Line::from(row));
                    y += 1;
                }

                // A sudden drop in nodes per second has a reason and this is usually
                // it. Nothing in the Python could show it.
                if !g.throttle.is_empty() && y < inner.y + inner.height {
                    put(
                        buf,
                        inner,
                        y,
                        Line::from(Span::styled(
                            format!("throttled: {}", truncate(&g.throttle.join(" "), inner.width as usize - 12)),
                            Styles::ink(sf_theme::WARM),
                        )),
                    );
                    y += 1;
                }

                // THE per-process split. `machine_panel`'s own docstring asks "is one
                // starving the other"; this is that answer, rather than an inference
                // from one aggregate, which is the mistake that cost a session.
                if variant == VariantId::Full {
                    let mut procs: Vec<_> = g.procs.iter().collect();
                    procs.sort_by_key(|p| std::cmp::Reverse(p.mem.unwrap_or(0)));
                    for p in procs.iter().take(3) {
                        if y >= inner.y + inner.height {
                            break;
                        }
                        put(
                            buf,
                            inner,
                            y,
                            Line::from(vec![
                                Span::styled("  ", Styles::dim()),
                                Span::styled(
                                    match p.mem {
                                        Some(m) => format!("{:>6}M ", m / 1_048_576),
                                        // NVML says "unavailable" for processes it
                                        // cannot attribute; showing 0 would be a lie.
                                        None => "   n/a ".into(),
                                    },
                                    Styles::body(),
                                ),
                                Span::styled(truncate(&p.name, 24), Styles::dim()),
                            ]),
                        );
                        y += 1;
                    }
                }
            }
            _ => {
                put(buf, inner, y, Line::from(Span::styled("no GPU reading", Styles::dim())));
                y += 1;
            }
        }

        // Units. `not-found` is its own state, not a red dot: a permanently red dot
        // for a unit that was deleted trains you to ignore the row.
        if let Some((units, _)) = cx.state.machine.units.get(cx.now) {
            if y < inner.y + inner.height {
                let mut row = Vec::new();
                for u in units {
                    let (glyph, style) = if u.missing() {
                        ("\u{25cb}", Styles::faint())
                    } else if u.healthy() {
                        ("\u{25cf}", Styles::ink(sf_theme::GOOD))
                    } else {
                        ("\u{25cf}", Styles::ink(sf_theme::BAD))
                    };
                    row.push(Span::styled(glyph, style));
                    let short = u.name.strip_prefix("sumofish-").unwrap_or(&u.name);
                    let short = short.strip_suffix(".service").unwrap_or(short);
                    row.push(Span::styled(
                        format!(" {short} "),
                        if u.missing() { Styles::faint() } else { Styles::dim() },
                    ));
                }
                // Fitted, not clipped: a unit row cut mid-name reads as though that
                // unit is the last one.
                put(buf, inner, y, fit(row, inner.width));
                y += 1;
            }
        }

        // The restart count comes from the WATCHDOG, never from systemd's
        // NRestarts: the watchdog uses `systemctl kill` + `restart`, so NRestarts
        // stays 0 through a SIGKILL and the Python showed a green dot through two
        // of them in half an hour.
        for b in cx.state.bots.values() {
            if y >= inner.y + inner.height {
                break;
            }
            if let Some((wd, _)) = b.watchdog.get(cx.now) {
                if wd.restarts_today > 0 || wd.stuck_games > 0 {
                    put(
                        buf,
                        inner,
                        y,
                        Line::from(Span::styled(
                            format!(
                                "{}: {} watchdog restart(s) today, {} stuck",
                                b.cfg.id, wd.restarts_today, wd.stuck_games
                            ),
                            Styles::ink(sf_theme::BAD),
                        )),
                    );
                    y += 1;
                }
            }
        }
    }
}
