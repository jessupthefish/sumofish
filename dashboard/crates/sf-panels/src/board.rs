//! The board: the picture, and a player block above and below it.
//!
//! The column is sized to the picture and the picture is centred in it. Two rows
//! per player, not one -- name, rating and clock on the first, captured material on
//! the second -- because those are the rows the old layout reserved and left blank
//! *under* the board, which is what made the picture look pinned to the top of a
//! column it was not filling.
//!
//! This panel **declares** the picture's cells and never draws them. The emitter
//! writes the image bytes and the solver owns the geometry, so there is exactly one
//! place that decides how big the board is. In the Python that arithmetic existed in
//! three places -- `Plan.image_*`, `board_panel`'s eight positional parameters, and
//! `verify_layout.py` recomputing it -- and the test's own docstring admitted
//! "nothing in either one can notice a disagreement".

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use sf_layout::widgets::{clockstr, put, truncate};
use sf_layout::{
    Cx, Panel, PanelId, PanelSpec, PictureRequest, RegionId, Relevance, Scope, Variant, VariantId,
    Weight,
};
use sf_theme::Styles;

pub static PANEL: BoardPanel = BoardPanel;

pub struct BoardPanel;

/// Rows reserved for each player block, above and below the picture.
const PLAYER_ROWS: u16 = 2;
/// Cells of clearance either side of the picture.
const GUTTER: u16 = 1;

static SPEC: PanelSpec = PanelSpec {
    id: PanelId("board"),
    title: "board",
    keybind: None,
    help: "the live position",
    region: RegionId::Board,
    scope: Scope::FocusedBot,
    // Pinned: a chess dashboard without the board is not the thing.
    weight: Weight::Pinned,
    variants: &[
        // A picture plus both player blocks.
        Variant::new(
            VariantId::Full,
            sf_layout::Size::new(24, 4 + 2 * PLAYER_ROWS),
            sf_layout::Size::new(80, 44),
            4,
        ),
        // No room for player blocks; just the board.
        Variant::new(
            VariantId::Compact,
            sf_layout::Size::new(18, 6),
            sf_layout::Size::new(40, 24),
            3,
        ),
        // A single line saying what is happening, for the narrowest tier. Every
        // panel needs one of these: without a variant that fits the narrowest
        // region it can land in, the panel would VANISH when the window is made
        // wider and the rail splits off. `check_registry` refuses that.
        Variant::fixed(VariantId::Line, 12, 1),
    ],
};

impl Panel for BoardPanel {
    fn spec(&self) -> &'static PanelSpec {
        &SPEC
    }

    fn relevance(&self, cx: &Cx<'_>) -> Relevance {
        match cx.game() {
            Some(_) => Relevance::Urgent,
            // No game: still shown, because "waiting for a game" is information and
            // an empty board column is where the eye goes first.
            None => Relevance::Normal,
        }
    }

    /// Declare where the picture goes. Square, centred, and inset by the gutters
    /// and the player rows.
    fn picture(&self, variant: VariantId, area: Rect, cx: &Cx<'_>) -> Option<PictureRequest> {
        if variant == VariantId::Line {
            return None;
        }
        let game = cx.game()?;
        let resolved = game.resolved()?;

        let players = if variant == VariantId::Full { PLAYER_ROWS } else { 0 };
        let avail_w = area.width.saturating_sub(2 * GUTTER);
        let avail_h = area.height.saturating_sub(2 * players);
        if avail_w == 0 || avail_h == 0 {
            return None;
        }
        // A cell is about twice as tall as it is wide, so a square picture needs
        // roughly half as many rows as columns.
        let cell_w = cx.picture.map(|p| p.cell_w).unwrap_or(8).max(1);
        let cell_h = cx.picture.map(|p| p.cell_h).unwrap_or(15).max(1);
        let by_w = avail_w as u32 * cell_w as u32;
        let by_h = avail_h as u32 * cell_h as u32;
        let px = by_w.min(by_h);
        let cols = ((px / cell_w as u32) as u16).min(avail_w).max(1);
        let rows = ((px / cell_h as u32) as u16).min(avail_h).max(1);

        let x = area.x + GUTTER + (avail_w.saturating_sub(cols)) / 2;
        let y = area.y + players;
        Some(PictureRequest {
            cells: Rect { x, y, width: cols, height: rows },
            fen: shakmaty_fen(&resolved.pos),
            flip: game.our_colour == shakmaty::Color::Black,
            last: resolved.last.map(|m| m.to_uci(shakmaty::CastlingMode::Standard).to_string()),
            check: None,
        })
    }

    fn render(&self, variant: VariantId, area: Rect, buf: &mut Buffer, cx: &Cx<'_>) {
        let Some(game) = cx.game() else {
            put(
                buf,
                area,
                area.y,
                Line::from(Span::styled("waiting for a game\u{2026}", Styles::faint())),
            );
            return;
        };

        if variant == VariantId::Line {
            let ply = game.resolved().map(|r| r.ply).unwrap_or(0);
            put(
                buf,
                area,
                area.y,
                Line::from(vec![
                    Span::styled(truncate(&game.opponent.username, 14), Styles::body()),
                    Span::styled(format!(" ply {ply}"), Styles::dim()),
                ]),
            );
            return;
        }

        if variant != VariantId::Full {
            return;
        }

        // The opponent above, us below, oriented so "below" is always our side --
        // which is what the flip is for.
        let pic_h = cx.picture.map(|p| p.cells.height).unwrap_or(0);
        let bottom = area.y + PLAYER_ROWS + pic_h;
        player_block(buf, area, area.y, &game.opponent.username, opponent_rating(game), game, false, cx);
        if bottom < area.y + area.height {
            let us = cx.bot_state().map(|b| b.cfg.user.clone()).unwrap_or_default();
            player_block(buf, area, bottom, &us, our_rating(game, cx), game, true, cx);
        }

        // Nothing is drawn over the picture's cells. With kitty graphics that is not
        // strictly required -- the M0 probe wrote ten lines of text across the
        // image's own rows and it survived -- but the sixel fallback needs it, and
        // `CellDiffOption::Skip` is applied by the app so ratatui does not clear
        // them either.
    }
}

fn opponent_rating(game: &sf_model::GameState) -> Option<i32> {
    game.opponent.rating
}

/// Our own rating for this game's time control, so the name we're playing
/// under carries a number the same way the opponent's already does -- the
/// opponent block has shown a rating since the panel's first version, ours
/// showed nothing at all.
fn our_rating(game: &sf_model::GameState, cx: &Cx<'_>) -> Option<i32> {
    let (account, _) = cx.bot_state()?.account.get(cx.now)?;
    account.perfs.get(&game.speed).map(|p| p.rating)
}

/// One player's two rows: name, rating and clock, then captured material.
fn player_block(
    buf: &mut Buffer,
    area: Rect,
    y: u16,
    name: &str,
    rating: Option<i32>,
    game: &sf_model::GameState,
    is_us: bool,
    _cx: &Cx<'_>,
) {
    if y >= area.y + area.height {
        return;
    }
    // Whose row this is, in chess terms, and whether it's their move right
    // now -- the one thing that actually distinguishes the two rows. Every
    // span below uses this same `running` check for both "us" and the
    // opponent, rather than `is_us` picking a permanently different style
    // for our own row: the two rows must read identically apart from
    // whichever one is actually to move, reported directly 2026-07-31 (the
    // opponent's row was always plain and ours was always accented,
    // regardless of the clock).
    let colour = if is_us { game.our_colour } else { !game.our_colour };
    let running = game.clocks.ticking == Some(colour);

    let mut spans = vec![Span::styled(
        truncate(if name.is_empty() { "?" } else { name }, 20),
        if running { Styles::accent() } else { Styles::body() },
    )];
    if let Some(r) = rating {
        spans.push(Span::styled(format!(" {r}"), Styles::dim()));
    }
    // The clock for this side. `ticking` says whose it is, so a stopped clock is
    // visibly stopped rather than merely not changing.
    let clock = match colour {
        shakmaty::Color::White => game.clocks.white,
        shakmaty::Color::Black => game.clocks.black,
    };
    if let Some(d) = clock {
        spans.push(Span::styled(
            format!("   {}", clockstr(d)),
            if running { Styles::accent() } else { Styles::dim() },
        ));
    }
    put(buf, area, y, Line::from(spans));
}

/// A full FEN string for a position, for the picture request.
fn shakmaty_fen(pos: &shakmaty::Chess) -> String {
    shakmaty::fen::Fen::from_position(pos, shakmaty::EnPassantMode::Always).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sf_model::state::{Account, BotConfig, BotState, Perf};
    use sf_model::{AppState, BotId, GameId, GameState};
    use std::sync::Arc;
    use std::time::Instant;

    /// The regression this was added for: the opponent block has carried a
    /// rating since the panel's first version, ours showed nothing at all
    /// because `player_block` was always called with a hardcoded `None`.
    #[test]
    fn our_rating_reads_the_perf_matching_the_games_speed() {
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
        let now = Instant::now();
        let mut account = Account::default();
        account.perfs.insert("blitz".into(), Perf { rating: 2201, games: 10, rd: 40, provisional: false, prog: 0 });
        account.perfs.insert("bullet".into(), Perf { rating: 1950, games: 30, rd: 60, provisional: false, prog: 0 });
        bot.account.set(account, now);

        let mut game = GameState::new(GameId("g1".into()), shakmaty::Color::White);
        game.speed = "blitz".into();

        let mut st = AppState::default();
        let id = bot.cfg.id.clone();
        st.bots.insert(id.clone(), bot);
        st.focus = Some(id);
        let cx = Cx { state: &st, now, wall: jiff::Timestamp::UNIX_EPOCH, bot: None, frame: 0, picture: None };

        assert_eq!(our_rating(&game, &cx), Some(2201), "must pick the perf matching the game's own speed, not just any perf");
    }

    /// The regression this was added for: the opponent's row was always
    /// plain and our own row was always accented, regardless of whose move
    /// it actually was -- reported directly 2026-07-31. Highlighting must
    /// follow `ticking`, the same rule the clock already used, not `is_us`.
    #[test]
    fn the_active_players_row_is_highlighted_whichever_side_it_is() {
        let mut game = GameState::new(GameId("g1".into()), shakmaty::Color::White);
        game.clocks = sf_model::state::Clocks {
            white: Some(std::time::Duration::from_secs(300)),
            black: Some(std::time::Duration::from_secs(280)),
            as_of: None,
            source: None,
            ticking: Some(shakmaty::Color::Black), // the opponent's move, not ours
        };

        let st = AppState::default();
        let cx = Cx { state: &st, now: Instant::now(), wall: jiff::Timestamp::UNIX_EPOCH, bot: None, frame: 0, picture: None };

        let area = Rect { x: 0, y: 0, width: 40, height: 4 };
        let mut buf = Buffer::empty(area);
        player_block(&mut buf, area, 0, "Foe", None, &game, false, &cx);
        player_block(&mut buf, area, 1, "SumoFish", None, &game, true, &cx);

        let accent = sf_theme::ACCENT.color();
        let body = sf_theme::FG.color();
        assert_eq!(buf.cell((0, 0)).unwrap().fg, accent, "the opponent is to move and must be highlighted");
        assert_eq!(buf.cell((0, 1)).unwrap().fg, body, "we are NOT to move and must be plain, same as any inactive row");

        // Flip whose move it is and confirm the highlight follows, not `is_us`.
        game.clocks.ticking = Some(shakmaty::Color::White);
        let mut buf = Buffer::empty(area);
        player_block(&mut buf, area, 0, "Foe", None, &game, false, &cx);
        player_block(&mut buf, area, 1, "SumoFish", None, &game, true, &cx);
        assert_eq!(buf.cell((0, 0)).unwrap().fg, body, "the opponent is no longer to move");
        assert_eq!(buf.cell((0, 1)).unwrap().fg, accent, "now it's our move, and OUR row is the one that highlights");
    }
}
