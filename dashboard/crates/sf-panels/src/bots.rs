//! One row per bot. Only appears when there is more than one.
//!
//! This is the panel that makes multi-bot legible: which bot the rest of the screen
//! is about, what each one is doing, and -- because the lichess budget is shared per
//! IP rather than per token -- what watching another one costs.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use sf_layout::widgets::{chrome, fit, lpad, put, track_tag, truncate};
use sf_layout::{
    Cx, Panel, PanelId, PanelSpec, RegionId, Relevance, Scope, Size, Variant, VariantId, Weight,
};
use sf_theme::Styles;

pub static PANEL: Bots = Bots;
pub struct Bots;

static SPEC: PanelSpec = PanelSpec {
    id: PanelId("bots"),
    title: "bots",
    region: RegionId::Rail,
    // The one panel that is genuinely about every bot at once. Every other panel
    // takes the focused one; this is how you choose it.
    scope: Scope::AllBots,
    weight: Weight::High,
    variants: &[
        Variant::new(VariantId::Full, Size::new(40, 3), Size::new(60, 10), 0),
        Variant::new(VariantId::Compact, Size::new(22, 2), Size::new(40, 6), 0),
        Variant::fixed(VariantId::Line, 10, 1),
    ],
};

impl Panel for Bots {
    fn spec(&self) -> &'static PanelSpec {
        &SPEC
    }

    /// Hidden with one bot. A list of one is a heading, not a list, and the header
    /// already says which bot this is.
    fn relevance(&self, cx: &Cx<'_>) -> Relevance {
        if cx.state.bots.len() < 2 { Relevance::Hidden } else { Relevance::Normal }
    }

    /// One row per bot. Without this the panel is a fixed three rows, which clips
    /// the list at two bots and wastes a row at one.
    fn rows_needed(&self, variant: VariantId, cx: &Cx<'_>) -> Option<u16> {
        (variant != VariantId::Line).then(|| cx.state.bots.len().min(9) as u16)
    }

    fn render(&self, variant: VariantId, area: Rect, buf: &mut Buffer, cx: &Cx<'_>) {
        if variant == VariantId::Line {
            let names: Vec<&str> = cx.state.bots.values().map(|b| b.cfg.label.as_str()).collect();
            put(buf, area, area.y, Line::from(Span::styled(truncate(&names.join(" "), area.width as usize), Styles::dim())));
            return;
        }
        let inner = chrome(area, SPEC.title, buf);
        if inner.height == 0 {
            return;
        }
        for (i, (id, bot)) in cx.state.bots.iter().enumerate() {
            if i as u16 >= inner.height {
                break;
            }
            let focused = cx.state.focus.as_ref() == Some(id);
            let mut row = vec![
                // 1-9 selects directly; the number is on screen so you do not have
                // to count.
                Span::styled(
                    format!("{} ", i + 1),
                    if focused { Styles::accent() } else { Styles::faint() },
                ),
                Span::styled(
                    lpad(&truncate(&bot.cfg.user, 16), 16),
                    if focused {
                        Styles::body().add_modifier(Modifier::BOLD)
                    } else {
                        Styles::dim()
                    },
                ),
            ];
            // Not watched is a state worth showing: it is the knob that makes more
            // bots affordable, and a bot with no data because you turned it off must
            // not look like a bot that has died.
            if !bot.cfg.watch {
                row.push(Span::styled("not watched", Styles::faint()));
                put(buf, inner, inner.y + i as u16, fit(row, inner.width));
                continue;
            }
            match bot.account.get(cx.now) {
                Some((acct, track)) => {
                    let best = acct.perfs.values().max_by_key(|p| p.games);
                    if let Some(p) = best {
                        row.push(Span::styled(format!("{} ", p.rating), Styles::body()));
                    }
                    row.push(track_tag(Cx::track(track), ""));
                }
                None => row.push(Span::styled("no account", Styles::faint())),
            }
            let playing = bot
                .playing
                .get(cx.now)
                .map(|(g, _)| g.len())
                .unwrap_or(0);
            row.push(Span::styled(
                format!("  {playing} playing"),
                if playing > 0 { Styles::accent() } else { Styles::faint() },
            ));
            put(buf, inner, inner.y + i as u16, fit(row, inner.width));
        }
    }
}
