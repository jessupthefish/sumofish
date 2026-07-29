//! Recent games, and what each one cost.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use sf_layout::widgets::{chrome, lpad, put, rpad, truncate};
use sf_layout::{
    Cx, Panel, PanelId, PanelSpec, RegionId, Relevance, Scope, Size, Variant, VariantId, Weight,
};
use sf_theme::Styles;

pub static PANEL: Results = Results;
pub struct Results;

/// "time forfeit" is the longest termination lichess sends.
const HOW_W: usize = 12;

static SPEC: PanelSpec = PanelSpec {
    id: PanelId("results"),
    title: "recent games",
    region: RegionId::Rail,
    scope: Scope::FocusedBot,
    weight: Weight::Low,
    variants: &[
        Variant::new(VariantId::Full, Size::new(46, 5), Size::new(66, 12), 1),
        Variant::new(VariantId::Compact, Size::new(24, 3), Size::new(46, 8), 1),
        Variant::fixed(VariantId::Line, 12, 1),
    ],
};

impl Panel for Results {
    fn spec(&self) -> &'static PanelSpec {
        &SPEC
    }

    fn relevance(&self, cx: &Cx<'_>) -> Relevance {
        match cx.bot_state().and_then(|b| b.results.get(cx.now)) {
            Some((r, _)) if !r.is_empty() => Relevance::Normal,
            _ => Relevance::Hidden,
        }
    }

    fn render(&self, variant: VariantId, area: Rect, buf: &mut Buffer, cx: &Cx<'_>) {
        let Some((games, _)) = cx.bot_state().and_then(|b| b.results.get(cx.now)) else { return };

        let (w, d, l) = tally(games);
        if variant == VariantId::Line {
            put(buf, area, area.y, Line::from(Span::styled(format!("last {}: {w}W {d}D {l}L", w + d + l), Styles::dim())));
            return;
        }

        let inner = chrome(area, SPEC.title, buf);
        if inner.height == 0 {
            return;
        }
        if games.is_empty() {
            put(buf, inner, inner.y, Line::from(Span::styled("no finished games yet", Styles::dim())));
            return;
        }

        // Every column including its separators is counted. The Python sliced by
        // hand here and shipped the same off-by-one twice, both times turning
        // "time forfeit" into "time forfei" -- a different, plausible-looking word.
        let fixed = 6 + 2 + 8 + 6 + HOW_W;
        let who_w = (inner.width as usize).saturating_sub(fixed).max(8);

        let mut y = inner.y;
        put(
            buf,
            inner,
            y,
            Line::from(vec![
                Span::styled(format!("last {}: ", w + d + l), Styles::faint()),
                Span::styled(format!("{w}W"), Styles::ink(sf_theme::GOOD)),
                Span::styled(format!(" {d}D"), Styles::dim()),
                Span::styled(format!(" {l}L"), Styles::ink(sf_theme::BAD)),
            ]),
        );
        y += 1;

        for g in games.iter().take(inner.height.saturating_sub(1) as usize) {
            let (mark, style) = match g.result.as_str() {
                "win" => ("W", Styles::ink(sf_theme::GOOD)),
                "loss" => ("L", Styles::ink(sf_theme::BAD)),
                "draw" => ("D", Styles::dim()),
                // PGN result `*`: aborted or abandoned, which is not a draw.
                _ => ("\u{00b7}", Styles::faint()),
            };
            let when = g.at.map(|t| t.strftime("%H:%M").to_string()).unwrap_or_else(|| "  ?  ".into());
            let mut row = vec![
                Span::styled(format!("{when} "), Styles::faint()),
                Span::styled(format!("{mark} "), style.add_modifier(Modifier::BOLD)),
                Span::styled(lpad(&truncate(&g.opponent, who_w), who_w), Styles::dim()),
            ];
            // What the game actually cost. On disk in every PGN as
            // WhiteRatingDiff/BlackRatingDiff and read by nothing before now.
            row.push(match g.rating_delta {
                Some(d) if d > 0 => Span::styled(rpad(&format!("+{d}"), 5), Styles::ink(sf_theme::GOOD)),
                Some(d) => Span::styled(rpad(&d.to_string(), 5), Styles::ink(sf_theme::BAD)),
                None => Span::styled("     ", Styles::faint()),
            });
            row.push(Span::styled(
                format!(" {}", truncate(&g.reason, HOW_W)),
                Styles::faint(),
            ));
            put(buf, inner, y, Line::from(row));
            y += 1;
        }
    }
}

/// Aborted games count in neither column: they are not a result.
fn tally(games: &[sf_model::state::FinishedGame]) -> (usize, usize, usize) {
    let count = |k: &str| games.iter().filter(|g| g.result == k).count();
    (count("win"), count("draw"), count("loss"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sf_model::state::FinishedGame;

    fn g(result: &str) -> FinishedGame {
        FinishedGame {
            id: None,
            opponent: "Foe".into(),
            opponent_rating: None,
            result: result.into(),
            reason: "normal".into(),
            our_colour: None,
            rating_delta: None,
            eco: None,
            opening: None,
            plies: None,
            at: None,
        }
    }

    #[test]
    fn aborted_games_are_in_no_column() {
        let games = vec![g("win"), g("loss"), g("draw"), g("none"), g("win")];
        assert_eq!(tally(&games), (2, 1, 1), "the aborted game counts nowhere");
    }

    /// The Python turned "time forfeit" into "time forfei" twice. The shared
    /// truncate marks the cut, so a clipped value can never be misread as a
    /// complete one.
    #[test]
    fn a_long_termination_is_marked_when_clipped_not_silently_cut() {
        assert_eq!(truncate("time forfeit", HOW_W), "time forfeit");
        assert_eq!(truncate("time forfeit", HOW_W - 1), "time forfe\u{2026}");
    }
}
