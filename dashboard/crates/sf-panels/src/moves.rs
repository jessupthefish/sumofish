//! The move list, with Stockfish's marks on the bad ones.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use sf_layout::widgets::{chrome, lpad, put, sparkline};
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
    region: RegionId::Rail,
    scope: Scope::FocusedBot,
    weight: Weight::Normal,
    variants: &[
        Variant::new(VariantId::Full, Size::new(34, 8), Size::new(52, 34), 3),
        Variant::new(VariantId::Compact, Size::new(22, 4), Size::new(34, 12), 2),
        Variant::fixed(VariantId::Line, 12, 1),
    ],
};

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

        let inner = chrome(area, SPEC.title, buf);
        if inner.height == 0 {
            return;
        }
        if game.moves.is_empty() {
            put(buf, inner, inner.y, Line::from(Span::styled("no moves yet", Styles::dim())));
            return;
        }

        // The Stockfish curve covers the WHOLE game from move one; the engine's own
        // only covers from when the dashboard attached. Prefer the former and say
        // which is in use, because they are not the same claim.
        let (series, third_party): (Vec<f64>, bool) = if game.sf_curve.len() >= 2 {
            (game.sf_curve.values().map(|v| game.ours(*v) as f64).collect(), true)
        } else {
            (game.curve.values().map(|v| game.ours(*v) as f64).collect(), false)
        };

        let mut y = inner.y;
        if series.len() >= 2 && inner.height > 2 {
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

        // Pairs, tail-first, so the current move is always visible.
        let rows = inner.height.saturating_sub(y - inner.y) as usize;
        let pairs = pair_up(&game.moves);
        let start = pairs.len().saturating_sub(rows.max(1));
        for (i, (n, white, black)) in pairs[start..].iter().enumerate() {
            let mut spans = vec![
                Span::styled(format!("{n:>3}. "), Styles::faint()),
                Span::styled(lpad(&white.0, 8), Styles::body()),
                grade_mark(white.1),
            ];
            match black {
                Some((san, g)) => {
                    spans.push(Span::styled(format!("  {}", lpad(san, 8)), Styles::body()));
                    spans.push(grade_mark(*g));
                }
                None => spans.push(Span::raw("          ")),
            }
            put(buf, inner, y + i as u16, Line::from(spans));
        }
    }
}

/// Only BAD moves are marked. Annotating accurate ones would be a column of noise,
/// and the eye is hunting for where the game went wrong.
fn grade_mark(g: Option<Grade>) -> Span<'static> {
    match g {
        Some(Grade::Inaccuracy) => Span::styled(
            "?!",
            Styles::ink(sf_theme::WARM).add_modifier(Modifier::BOLD),
        ),
        Some(Grade::Mistake) => {
            Span::styled("? ", Styles::ink(sf_theme::BAD).add_modifier(Modifier::BOLD))
        }
        Some(Grade::Blunder) => {
            Span::styled("??", Styles::ink(sf_theme::BAD).add_modifier(Modifier::BOLD))
        }
        _ => Span::styled("  ", Styles::faint()),
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
    use sf_model::state::{MoveOrigin, MoveRec};

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
        assert_eq!(grade_mark(None).content, "  ");
        assert_eq!(grade_mark(Some(Grade::Best)).content, "  ");
        assert_eq!(grade_mark(Some(Grade::Good)).content, "  ");
        assert_eq!(grade_mark(Some(Grade::Inaccuracy)).content, "?!");
        assert_eq!(grade_mark(Some(Grade::Blunder)).content, "??");
    }

    #[test]
    fn every_mark_is_exactly_two_cells_so_the_columns_line_up() {
        for g in [
            None,
            Some(Grade::Best),
            Some(Grade::Good),
            Some(Grade::Inaccuracy),
            Some(Grade::Mistake),
            Some(Grade::Blunder),
        ] {
            assert_eq!(grade_mark(g).content.chars().count(), 2, "{g:?}");
        }
    }
}
