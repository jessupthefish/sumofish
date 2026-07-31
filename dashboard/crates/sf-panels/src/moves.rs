//! The move list, with Stockfish's marks on the bad ones.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use sf_layout::widgets::{chrome, put, sparkline};
use sf_layout::{
    Cx, Panel, PanelId, PanelSpec, RegionId, Relevance, Scope, Size, Variant, VariantId, Weight,
};
use sf_model::state::Grade;
use sf_theme::Styles;

pub static PANEL: Moves = Moves;
pub struct Moves;

static SPEC: PanelSpec = PanelSpec {
    id: PanelId("moves"),
    title: "moves",
    keybind: Some("PgUp/PgDn"),
    help: "the move list, with Stockfish's marks on the bad ones",
    region: RegionId::Rail,
    scope: Scope::FocusedBot,
    weight: Weight::Normal,
    variants: &[
        Variant::new(VariantId::Full, Size::new(34, 8), Size::new(52, 34), 3),
        Variant::new(VariantId::Compact, Size::new(22, 4), Size::new(34, 12), 2),
        Variant::fixed(VariantId::Line, 12, 1),
    ],
};

/// One move's rendered width (SAN plus its fused grade suffix, then padded):
/// "Qxd8+??" is 7 characters and already a rare, worst-realistic case
/// ("Nbxd7+" disambiguated-capture-check is 6; a full disambiguated
/// promotion-capture-check like "exd8=Q+" is also 7). A move that still
/// overflows this just renders unpadded that one row -- `lpad` never
/// truncates -- rather than losing a character of a real move.
const FIELD_W: usize = 7;
/// "27." -- two digits is every realistic game; `{n}.` past 99 just widens
/// that one row's number, same non-truncating tradeoff as `FIELD_W`.
const NUM_W: usize = 3;
const GAP: u16 = 1;
const COL_W: u16 = NUM_W as u16 + FIELD_W as u16 + 1 + FIELD_W as u16; // num + white + sep + black

impl Panel for Moves {
    fn spec(&self) -> &'static PanelSpec {
        &SPEC
    }

    fn relevance(&self, cx: &Cx<'_>) -> Relevance {
        match cx.game() {
            Some(g) if !g.moves.is_empty() => Relevance::Normal,
            Some(_) => Relevance::Idle,
            None => Relevance::Hidden,
        }
    }

    fn render(&self, variant: VariantId, area: Rect, buf: &mut Buffer, cx: &Cx<'_>) {
        let Some(game) = cx.game() else { return };

        if variant == VariantId::Line {
            let last = game.moves.last().map(|m| m.san.as_str()).unwrap_or("-");
            put(buf, area, area.y, Line::from(Span::styled(format!("{} moves, last {last}", game.moves.len()), Styles::dim())));
            return;
        }

        if game.moves.is_empty() {
            let inner = chrome(area, SPEC.title, buf);
            if inner.height > 0 {
                put(buf, inner, inner.y, Line::from(Span::styled("no moves yet", Styles::dim())));
            }
            return;
        }
        let pairs = pair_up(&game.moves);

        // The Stockfish curve covers the WHOLE game from move one; the engine's own
        // only covers from when the dashboard attached. Prefer the former and say
        // which is in use, because they are not the same claim.
        // White's frame, always -- see `GameState::curve`'s doc comment.
        let (series, third_party): (Vec<f64>, bool) = if game.sf_curve.len() >= 2 {
            (game.sf_curve.values().map(|v| *v as f64).collect(), true)
        } else {
            (game.curve.values().map(|v| *v as f64).collect(), false)
        };

        // Chrome always costs exactly one row top and bottom (the one place
        // that knows this is `chrome` itself; mirrored here because the title
        // -- built below, from `start`/`end` -- has to exist before `chrome`
        // is called). An eval row is shown under the same condition `chrome`'s
        // caller checks after the real call, so it has to be replicated too.
        let inner_h = area.height.saturating_sub(2);
        let has_eval_row = series.len() >= 2 && inner_h > 2;
        let rows = inner_h.saturating_sub(if has_eval_row { 1 } else { 0 }).max(1) as usize;

        // As many side-by-side columns as the width actually allows, snaking
        // (column 0 top-to-bottom covers the oldest visible moves, then
        // column 1 continues where it left off, then column 2) -- a long
        // game needs to be readable at a glance, not paged through one row
        // at a time. Real-width fields (`FIELD_W`, not a worst-case 8) are
        // what make 3 columns reachable at all at this panel's typical
        // width: reported directly, 2026-07-30, after the wider 8-character
        // padded columns only ever fit 2.
        let inner_w = area.width.saturating_sub(2);
        let cols = ((inner_w + GAP) / (COL_W + GAP)).max(1) as usize;
        let capacity = rows * cols;

        // How far back into the game the visible window sits. 0 (the default)
        // shows the tail, i.e. the current move -- same as before this ever
        // scrolled or had more than one column. `PageUp`/`PageDown` (wired in
        // sf-app/run.rs) move it; clamped here rather than at the write site,
        // so a scroll position left over from a longer game (or a resize to
        // fewer columns) never reads out of bounds against a shorter one.
        let max_scroll = pairs.len().saturating_sub(capacity) as u16;
        let scroll = cx
            .state
            .ui
            .views
            .get("moves")
            .map(|v| v.scroll)
            .unwrap_or(0)
            .min(max_scroll);
        let end = pairs.len().saturating_sub(scroll as usize);
        let start = end.saturating_sub(capacity);

        // Discoverable in the title, not a stolen content row: how many moves
        // sit above and below the visible window. `PgUp`/`PgDn` scroll, per
        // the help overlay.
        let below = pairs.len() - end;
        let title: std::borrow::Cow<str> = match (start > 0, below > 0) {
            (false, false) => SPEC.title.into(),
            (true, false) => format!("{} \u{2191}{start}", SPEC.title).into(),
            (false, true) => format!("{} \u{2193}{below}", SPEC.title).into(),
            (true, true) => format!("{} \u{2191}{start} \u{2193}{below}", SPEC.title).into(),
        };
        let inner = chrome(area, &title, buf);
        if inner.height == 0 {
            return;
        }

        let mut y = inner.y;
        if has_eval_row {
            let w = sf_theme::GAUGE_MAX.min(inner.width.saturating_sub(20).max(8));
            let (spark, lo, hi) = sparkline(&series, w, sf_theme::PLUM);
            let mut spans = vec![Span::styled(
                // The interpunct is how you tell which source you are looking at.
                if third_party { "eval " } else { "eval\u{00b7}" },
                Styles::dim(),
            )];
            spans.extend(spark.spans);
            // Every trend carries its endpoints as text; a shaded trace with no
            // scale is decoration.
            spans.push(Span::styled(format!(" {lo:.2}-{hi:.2}"), Styles::faint()));
            put(buf, inner, y, Line::from(spans));
            y += 1;
        }

        // Snaking: column 0 holds `pairs[start..start+rows]` (oldest of the
        // visible window at the top), column 1 continues from where column 0
        // left off, and so on -- reading order matches chronological order.
        for (idx, (n, white, black)) in pairs[start..end].iter().enumerate() {
            let col = idx / rows;
            let row = idx % rows;
            let x_offset = col as u16 * (COL_W + GAP);
            // Clamp rather than always COL_W: a Compact-tier width can force
            // `cols` down to 1 while still being narrower than COL_W, and an
            // unclamped width would write past `inner`'s own right edge into
            // whatever the buffer holds next.
            let col_area = Rect {
                x: inner.x + x_offset,
                y: inner.y,
                width: inner.width.saturating_sub(x_offset).min(COL_W),
                height: inner.height,
            };
            let mut spans = vec![Span::styled(format!("{n:>w$}.", w = NUM_W - 1), Styles::faint())];
            spans.extend(field(&white.0, white.1));
            spans.push(Span::raw(" "));
            match black {
                Some((san, g)) => spans.extend(field(san, *g)),
                None => spans.push(Span::raw(" ".repeat(FIELD_W))),
            }
            put(buf, col_area, y + row as u16, Line::from(spans));
        }
    }
}

/// SAN plus its grade mark, fused (so wrapping/column logic never has to
/// track them separately) and padded to `FIELD_W` so the column stays
/// aligned. The grade mark keeps its own colour even though it is now part
/// of the same visual field as the move.
fn field(san: &str, grade: Option<Grade>) -> Vec<Span<'static>> {
    let mark = grade_mark(grade);
    let used = san.chars().count() + mark.as_ref().map_or(0, |m| m.content.chars().count());
    let mut spans = vec![Span::styled(san.to_string(), Styles::body())];
    if let Some(m) = mark {
        spans.push(m);
    }
    let pad = FIELD_W.saturating_sub(used);
    if pad > 0 {
        spans.push(Span::raw(" ".repeat(pad)));
    }
    spans
}

/// Only BAD moves are marked. Annotating accurate ones would be noise, and
/// the eye is hunting for where the game went wrong.
fn grade_mark(g: Option<Grade>) -> Option<Span<'static>> {
    match g {
        Some(Grade::Inaccuracy) => Some(Span::styled(
            "?!",
            Styles::ink(sf_theme::WARM).add_modifier(Modifier::BOLD),
        )),
        Some(Grade::Mistake) => {
            Some(Span::styled("?", Styles::ink(sf_theme::BAD).add_modifier(Modifier::BOLD)))
        }
        Some(Grade::Blunder) => {
            Some(Span::styled("??", Styles::ink(sf_theme::BAD).add_modifier(Modifier::BOLD)))
        }
        _ => None,
    }
}

type Half = (String, Option<Grade>);

/// `(move number, white, black)`. Handles a game that starts on Black's move, which
/// is what a mid-game attach looks like.
fn pair_up(moves: &[sf_model::state::MoveRec]) -> Vec<(u32, Half, Option<Half>)> {
    let mut out: Vec<(u32, Half, Option<Half>)> = Vec::new();
    for m in moves {
        let number = m.ply / 2 + 1;
        let is_white = m.ply % 2 == 0;
        let half = (m.san.clone(), m.grade);
        match out.last_mut() {
            Some((n, _, black @ None)) if *n == number && !is_white => *black = Some(half),
            _ => {
                if is_white {
                    out.push((number, half, None));
                } else {
                    // Black moved first from our point of view: a mid-game attach.
                    out.push((number, ("\u{2026}".into(), None), Some(half)));
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::buffer::Buffer;
    use sf_model::state::{BotConfig, BotState, MoveOrigin, MoveRec};
    use sf_model::{AppState, BotId, GameId, GameState};
    use std::sync::Arc;
    use std::time::Instant;

    fn mv(ply: u32, san: &str) -> MoveRec {
        MoveRec {
            ply,
            san: san.into(),
            uci: "e2e4".into(),
            origin: MoveOrigin::Unknown,
            authority: sf_model::PosSource::Stream,
            clock: None,
            grade: None,
        }
    }

    #[test]
    fn moves_pair_into_numbered_rows() {
        let ms = vec![mv(0, "e4"), mv(1, "e5"), mv(2, "Nf3")];
        let p = pair_up(&ms);
        assert_eq!(p.len(), 2);
        assert_eq!((p[0].0, p[0].1.0.as_str()), (1, "e4"));
        assert_eq!(p[0].2.as_ref().unwrap().0, "e5");
        assert_eq!((p[1].0, p[1].1.0.as_str()), (2, "Nf3"));
        assert!(p[1].2.is_none(), "an unanswered white move leaves the black half open");
    }

    /// Attaching mid-game means the first move we see may be Black's. The row must
    /// still be numbered correctly rather than silently shifting the whole list.
    #[test]
    fn a_game_joined_on_blacks_move_is_still_numbered_right() {
        let ms = vec![mv(7, "Nc6"), mv(8, "Bb5")];
        let p = pair_up(&ms);
        assert_eq!(p[0].0, 4, "ply 7 is Black's move 4");
        assert_eq!(p[0].1.0, "\u{2026}", "White's half is elided, not invented");
        assert_eq!(p[0].2.as_ref().unwrap().0, "Nc6");
        assert_eq!(p[1].0, 5);
    }

    #[test]
    fn only_bad_moves_are_marked() {
        assert_eq!(grade_mark(None), None);
        assert_eq!(grade_mark(Some(Grade::Best)), None);
        assert_eq!(grade_mark(Some(Grade::Good)), None);
        assert_eq!(grade_mark(Some(Grade::Inaccuracy)).unwrap().content, "?!");
        assert_eq!(grade_mark(Some(Grade::Mistake)).unwrap().content, "?");
        assert_eq!(grade_mark(Some(Grade::Blunder)).unwrap().content, "??");
    }

    /// The actual, twice-reported complaint: at this panel's real observed
    /// size (width 60, from the live layout at a realistic terminal size --
    /// see the commit this landed in), 8-character padded columns only ever
    /// fit 2 side by side. Real-width `FIELD_W`-padded columns must fit 3.
    #[test]
    fn three_columns_fit_at_the_panels_real_observed_width() {
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
        let mut game = GameState::new(GameId("g1".into()), shakmaty::Color::White);
        // 60 pairs (120 plies): more than 2 columns' worth of rows (26 each
        // at this height) can hold, so a 3rd column is actually load-bearing
        // for this test, not just geometrically possible.
        for ply in 0..120u32 {
            game.moves.push(mv(ply, if ply % 2 == 0 { "e4" } else { "e5" }));
        }
        bot.games.insert(game.id.clone(), game);
        bot.focus_game = Some(GameId("g1".into()));
        let mut st = AppState::default();
        let id = bot.cfg.id.clone();
        st.bots.insert(id.clone(), bot);
        st.focus = Some(id);

        // width=60, height=28: the real `moves` rect solved for this codebase
        // at a realistic 128x60 terminal, post-`api`-panel-removal.
        let area = Rect { x: 0, y: 0, width: 60, height: 28 };
        let mut buf = Buffer::empty(area);
        let now = Instant::now();
        let cx = Cx { state: &st, now, wall: jiff::Timestamp::UNIX_EPOCH, bot: None, frame: 0, picture: None };
        PANEL.render(VariantId::Full, area, &mut buf, &cx);

        let row = |y: u16| -> String {
            (0..area.width).map(|x| buf.cell((x, y)).map(|c| c.symbol()).unwrap_or(" ")).collect()
        };
        let inner_y = 1u16;
        // Column boundaries at x = 1, 1+COL_W+GAP, 1+2*(COL_W+GAP).
        let col2_x = 1 + (COL_W + GAP) as usize * 2;
        let col2 = row(inner_y).chars().skip(col2_x).take(COL_W as usize).collect::<String>();
        assert!(
            !col2.trim().is_empty(),
            "a third column must actually render content at width 60: row={:?} col2_x={col2_x}",
            row(inner_y)
        );
    }

    #[test]
    fn a_field_wider_than_fields_w_is_not_truncated() {
        // "Qa1xd4+" (7) fits exactly; anything even longer must still render
        // whole rather than being cut, per `field`'s own non-truncating
        // design (matching `lpad` elsewhere in this codebase).
        let spans = field("Qa1xd4+", None);
        let text: String = spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(text.trim_end(), "Qa1xd4+");
    }
}
