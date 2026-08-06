//! One row per game the focused bot is currently playing. Only appears when
//! there is more than one -- `concurrency: 2` in lichess-bot's own config
//! means this is routinely two, and until now there was no way to see the
//! second one at all: the board only ever draws `focus_game`, and nothing
//! set that to anything but "whichever game showed up first".
//!
//! Distinct from `bots`: that panel switches which BOT the rest of the
//! screen is about; this one switches which of the FOCUSED bot's own
//! simultaneous games it is about. Different axis, same shape.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use sf_layout::widgets::{chrome, clockstr, fit, lpad, put, truncate};
use sf_layout::{
    Cx, Panel, PanelId, PanelSpec, RegionId, Relevance, Scope, Size, Variant, VariantId, Weight,
};
use sf_theme::Styles;

pub static PANEL: Games = Games;
pub struct Games;

static SPEC: PanelSpec = PanelSpec {
    id: PanelId("games"),
    title: "games",
    keybind: Some("g"),
    help: "one row per game the focused bot is playing at once; g cycles",
    region: RegionId::Rail,
    scope: Scope::FocusedBot,
    weight: Weight::High,
    variants: &[
        Variant::new(VariantId::Full, Size::new(40, 3), Size::new(60, 10), 0),
        Variant::new(VariantId::Compact, Size::new(22, 2), Size::new(40, 6), 0),
        Variant::fixed(VariantId::Line, 10, 1),
    ],
};

impl Panel for Games {
    fn spec(&self) -> &'static PanelSpec {
        &SPEC
    }

    /// Hidden below two: a game of one is what the board already shows, same
    /// reasoning `bots` uses for a single bot.
    fn relevance(&self, cx: &Cx<'_>) -> Relevance {
        match cx.bot_state().and_then(|b| b.playing.get(cx.now)) {
            Some((games, _)) if games.len() > 1 => Relevance::Normal,
            _ => Relevance::Hidden,
        }
    }

    fn rows_needed(&self, variant: VariantId, cx: &Cx<'_>) -> Option<u16> {
        if variant == VariantId::Line {
            return None;
        }
        let n = cx.bot_state().and_then(|b| b.playing.get(cx.now)).map(|(g, _)| g.len()).unwrap_or(0);
        Some(n.min(9) as u16)
    }

    fn render(&self, variant: VariantId, area: Rect, buf: &mut Buffer, cx: &Cx<'_>) {
        let Some(bot) = cx.bot_state() else { return };
        let Some((games, _)) = bot.playing.get(cx.now) else { return };

        if variant == VariantId::Line {
            let n = games.len();
            let turn = games.iter().filter(|g| g.is_my_turn).count();
            put(
                buf,
                area,
                area.y,
                Line::from(Span::styled(format!("{n} games, {turn} to move"), Styles::dim())),
            );
            return;
        }

        let inner = chrome(area, SPEC.title, buf);
        if inner.height == 0 {
            return;
        }
        for (i, g) in games.iter().enumerate() {
            if i as u16 >= inner.height {
                break;
            }
            let focused = bot.focus_game.as_ref() == Some(&g.id);
            let mut row = vec![
                Span::styled(
                    format!("{} ", i + 1),
                    if focused { Styles::accent() } else { Styles::faint() },
                ),
                Span::styled(
                    lpad(&truncate(&g.opponent.username, 16), 16),
                    if focused {
                        Styles::body().add_modifier(Modifier::BOLD)
                    } else {
                        Styles::dim()
                    },
                ),
            ];
            if let Some(r) = g.opponent.rating {
                row.push(Span::styled(format!(" {r}"), Styles::dim()));
            }
            // Whose move it is, is the one fact worth a glance across every
            // game at once -- it is the difference between "thinking" and
            // "waiting on them".
            row.push(Span::styled(
                if g.is_my_turn { "  to move" } else { "  waiting" },
                if g.is_my_turn { Styles::ink(sf_theme::WARM) } else { Styles::faint() },
            ));
            if let Some(secs) = g.seconds_left {
                row.push(Span::styled(
                    format!("  {}", clockstr(std::time::Duration::from_secs(secs as u64))),
                    Styles::dim(),
                ));
            }
            put(buf, inner, inner.y + i as u16, fit(row, inner.width));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sf_model::state::{BotConfig, BotState, Opponent, PlayingGame};
    use sf_model::{AppState, BotId, GameId};
    use std::sync::Arc;
    use std::time::Instant;

    fn cfg() -> Arc<BotConfig> {
        Arc::new(BotConfig {
            id: BotId("sumofish".into()),
            user: "SumoFish".into(),
            label: String::new(),
            unit: None,
            telemetry: "/dev/null".into(),
            pgn_dir: None,
            versions: None,
            rating_log: None,
            watch: true,
        })
    }

    fn game(id: &str, opponent: &str, is_my_turn: bool) -> PlayingGame {
        PlayingGame {
            id: GameId(id.into()),
            our_colour: shakmaty::Color::White,
            fen: String::new(),
            last_move: None,
            is_my_turn,
            seconds_left: Some(300),
            opponent: Opponent { username: opponent.into(), title: None, rating: Some(2000), provisional: false, ai_level: None },
            speed: "rapid".into(),
            rated: true,
        }
    }

    /// The core reason this panel exists: a single game is what the board
    /// already shows, so a list of one is a heading, not a list.
    #[test]
    fn hidden_with_one_game_or_fewer() {
        let mut bot = BotState::new(cfg());
        let now = Instant::now();
        let mut st = AppState::default();

        bot.playing.set(vec![], now);
        st.bots.insert(bot.cfg.id.clone(), bot.clone());
        st.focus = Some(bot.cfg.id.clone());
        let cx = Cx { state: &st, now, wall: jiff::Timestamp::UNIX_EPOCH, bot: None, frame: 0, picture: None };
        assert_eq!(PANEL.relevance(&cx), Relevance::Hidden, "no games");

        bot.playing.set(vec![game("g1", "Foe", true)], now);
        st.bots.insert(bot.cfg.id.clone(), bot.clone());
        let cx = Cx { state: &st, now, wall: jiff::Timestamp::UNIX_EPOCH, bot: None, frame: 0, picture: None };
        assert_eq!(PANEL.relevance(&cx), Relevance::Hidden, "one game");

        bot.playing.set(vec![game("g1", "Foe", true), game("g2", "Bar", false)], now);
        st.bots.insert(bot.cfg.id.clone(), bot.clone());
        let cx = Cx { state: &st, now, wall: jiff::Timestamp::UNIX_EPOCH, bot: None, frame: 0, picture: None };
        assert_eq!(PANEL.relevance(&cx), Relevance::Normal, "two games");
    }

    /// The focused game is highlighted and numbered; whoever's actually to
    /// move is the one fact this panel exists to surface at a glance.
    #[test]
    fn rows_show_who_is_focused_and_who_is_to_move() {
        let mut bot = BotState::new(cfg());
        let now = Instant::now();
        bot.playing.set(vec![game("g1", "Foe", true), game("g2", "Bar", false)], now);
        bot.focus_game = Some(GameId("g2".into()));
        let mut st = AppState::default();
        st.bots.insert(bot.cfg.id.clone(), bot);
        st.focus = Some(BotId("sumofish".into()));
        let cx = Cx { state: &st, now, wall: jiff::Timestamp::UNIX_EPOCH, bot: None, frame: 0, picture: None };

        let area = Rect { x: 0, y: 0, width: 40, height: 4 };
        let mut buf = Buffer::empty(area);
        PANEL.render(VariantId::Full, area, &mut buf, &cx);
        let text = (0..area.height)
            .map(|y| (0..area.width).map(|x| buf.cell((x, y)).map(|c| c.symbol()).unwrap_or(" ")).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(text.contains("Foe") && text.contains("to move"), "g1 is to move:\n{text}");
        assert!(text.contains("Bar") && text.contains("waiting"), "g2 is waiting on the opponent:\n{text}");
        // Row 0 is the chrome border; row 1 is g1, row 2 is g2.
        let accent = sf_theme::ACCENT.color();
        assert_eq!(buf.cell((1, 2)).unwrap().fg, accent, "row 2 (g2) is the focused one and must be highlighted");
        assert_eq!(buf.cell((1, 1)).unwrap().fg, sf_theme::FAINT.color(), "row 1 (g1) is not focused");
    }
}
