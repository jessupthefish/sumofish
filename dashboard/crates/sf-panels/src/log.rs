//! The bot's own journal, raw: `journalctl -u chess-gpu-bot.service -f` without
//! leaving the dashboard.
//!
//! Distinct from `tape`: the tape is a curated cross-source narrative (moves,
//! games, watchdog restarts, tablebase attributions) built by `journal::Parser`
//! recognizing a handful of specific patterns in lichess-bot's own log shape.
//! Most of what the process actually logs -- HTTP retries, one-off warnings, a
//! traceback -- matches none of those patterns and would otherwise never appear
//! anywhere in the dashboard. This is the panel that absorbed `sumofish log`.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use sf_layout::widgets::{chrome, lpad, put, truncate};
use sf_layout::{
    Cx, Panel, PanelId, PanelSpec, RegionId, Relevance, Scope, Size, Variant, VariantId, Weight,
};
use sf_model::state::LogLevel;
use sf_theme::Styles;

pub static PANEL: Log = Log;
pub struct Log;

static SPEC: PanelSpec = PanelSpec {
    id: PanelId("log"),
    title: "bot log",
    keybind: None,
    help: "the bot's raw journal, unfiltered -- what `journalctl -f` would show",
    region: RegionId::Board,
    scope: Scope::FocusedBot,
    weight: Weight::Low,
    variants: &[
        Variant::new(VariantId::Full, Size::new(40, 5), Size::new(90, 12), 1),
        Variant::new(VariantId::Compact, Size::new(20, 3), Size::new(40, 6), 1),
        Variant::fixed(VariantId::Line, 12, 1),
    ],
};

impl Panel for Log {
    fn spec(&self) -> &'static PanelSpec {
        &SPEC
    }

    /// Urgent on a recent error or warning, the same way `machine` resists being
    /// dropped when something is actually wrong -- a raw log's whole reason to
    /// exist is surfacing exactly this.
    fn relevance(&self, cx: &Cx<'_>) -> Relevance {
        let Some(bot) = cx.bot_state() else { return Relevance::Hidden };
        if bot.log.is_empty() {
            return Relevance::Idle;
        }
        if bot.log.iter_rev().take(5).any(|e| e.level != LogLevel::Info) {
            Relevance::Urgent
        } else {
            Relevance::Normal
        }
    }

    fn render(&self, variant: VariantId, area: Rect, buf: &mut Buffer, cx: &Cx<'_>) {
        let Some(bot) = cx.bot_state() else { return };

        if variant == VariantId::Line {
            let last = bot.log.tail(1).next().map(|e| e.text.clone()).unwrap_or_default();
            put(
                buf,
                area,
                area.y,
                Line::from(Span::styled(truncate(&last, area.width as usize), Styles::faint())),
            );
            return;
        }

        let inner = chrome(area, SPEC.title, buf);
        if inner.height == 0 {
            return;
        }
        let rows = inner.height as usize;
        let entries: Vec<_> = bot.log.tail(rows).collect();
        if entries.is_empty() {
            put(buf, inner, inner.y, Line::from(Span::styled("nothing yet", Styles::faint())));
            return;
        }
        for (i, e) in entries.iter().enumerate() {
            // A fixed wall clock in `Cx` is what makes this snapshot-testable.
            let hhmmss = e.at.strftime("%H:%M:%S").to_string();
            let (label, style) = level_style(e.level);
            put(
                buf,
                inner,
                inner.y + i as u16,
                Line::from(vec![
                    Span::styled(format!("{hhmmss} "), Styles::faint()),
                    Span::styled(lpad(label, 6), style),
                    Span::styled(
                        truncate(&e.text, inner.width.saturating_sub(16) as usize),
                        Styles::body(),
                    ),
                ]),
            );
        }
    }
}

fn level_style(l: LogLevel) -> (&'static str, ratatui::style::Style) {
    match l {
        LogLevel::Info => ("info", Styles::dim()),
        LogLevel::Warn => ("warn", Styles::ink(sf_theme::WARM)),
        LogLevel::Error => ("error", Styles::ink(sf_theme::BAD)),
    }
}
