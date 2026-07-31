//! Recent games, and what each one cost.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use sf_layout::widgets::{chrome, lpad, put, rpad, truncate};
use sf_layout::{
    Cx, Panel, PanelId, PanelSpec, RegionId, Relevance, Scope, Size, Variant, VariantId, Weight,
};
use sf_model::state::FinishedGame;
use sf_theme::Styles;

pub static PANEL: Results = Results;
pub struct Results;

/// "time forfeit" is the longest termination lichess sends.
const HOW_W: usize = 12;

static SPEC: PanelSpec = PanelSpec {
    id: PanelId("results"),
    title: "recent games",
    keybind: None,
    help: "recent games, and what each one cost",
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
        let Some(bot) = cx.bot_state() else { return };
        let Some((games, _)) = bot.results.get(cx.now) else { return };

        // Scoped to the currently deployed version when that's known, same
        // reasoning `header` already applies to the rating: a lifetime
        // record averages engines that no longer exist, and "1W 3L" reads as
        // "this version is struggling" when three of those losses were two
        // versions ago. Falls back to the full window when there's no
        // deploy timestamp to scope against (offline, fixtures, a bot with
        // no `versions` file configured), so this never produces less
        // information than before, only more precise information when it can.
        let deployed = bot.version.get(cx.now).and_then(|(v, _)| v.at.map(|at| (v.version.clone(), at)));
        let scoped: Vec<FinishedGame> = match &deployed {
            Some((_, at)) => games.iter().filter(|g| g.at.is_some_and(|a| a >= *at)).cloned().collect(),
            None => games.clone(),
        };
        if variant == VariantId::Line {
            // No list under this one, just the summary -- the whole scoped
            // record, always, since there are no rows below it to disagree
            // with.
            let (w, d, l) = tally(&scoped);
            let label = header_label(&deployed, scoped.len());
            put(
                buf,
                area,
                area.y,
                Line::from(Span::styled(format!("{label}{w}W {d}D {l}L"), Styles::dim())),
            );
            return;
        }

        let inner = chrome(area, SPEC.title, buf);
        if inner.height == 0 {
            return;
        }
        if scoped.is_empty() {
            let msg = match &deployed {
                Some((v, _)) => format!("no games yet on {v}"),
                None => "no finished games yet".to_string(),
            };
            put(buf, inner, inner.y, Line::from(Span::styled(msg, Styles::dim())));
            return;
        }

        // How many rows actually fit -- a physical constraint on the LIST
        // only. The record above it is a different claim ("your record on
        // this version") than "here are exactly N games below", so unlike
        // the list it is never truncated to fit: a version's record must
        // read the same whether the panel is 4 rows tall or 40. Requested
        // directly 2026-07-31, after the previous fix (matching the count
        // to the listed rows) turned out to be the wrong invariant once the
        // header stopped being "the last N games" and became "your record
        // on vN" -- a record has no natural row-count to match at all.
        let shown = &scoped[..scoped.len().min(inner.height.saturating_sub(1) as usize)];

        // Version known: the record is the FULL scoped set, not just
        // whichever games happened to fit on screen -- "vN: 15W 2D 3L" must
        // be true regardless of how many of those 20 games this panel has
        // room to list underneath. No version known: falls back to the
        // old "last N" framing, where N has to match `shown` or the header
        // reads as having silently dropped games (the 2026-07-31 bug).
        let (w, d, l) = match &deployed {
            Some(_) => tally(&scoped),
            None => tally(shown),
        };
        let label = header_label(&deployed, shown.len());

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
                Span::styled(label, Styles::faint()),
                Span::styled(format!("{w}W"), Styles::ink(sf_theme::GOOD)),
                Span::styled(format!(" {d}D"), Styles::dim()),
                Span::styled(format!(" {l}L"), Styles::ink(sf_theme::BAD)),
            ]),
        );
        y += 1;

        for g in shown {
            let (mark, style) = match g.result.as_str() {
                "win" => ("W", Styles::ink(sf_theme::GOOD)),
                "loss" => ("L", Styles::ink(sf_theme::BAD)),
                "draw" => ("D", Styles::dim()),
                // PGN result `*`: aborted or abandoned, which is not a draw.
                _ => ("\u{00b7}", Styles::faint()),
            };
            let when = g.at.map(|t| t.strftime("%H:%M").to_string()).unwrap_or_else(|| "  ?  ".into());
            let who = describe_opponent(&g.opponent, g.opponent_rating, g.opponent_is_bot);
            let mut row = vec![
                Span::styled(format!("{when} "), Styles::faint()),
                Span::styled(format!("{mark} "), style.add_modifier(Modifier::BOLD)),
                Span::styled(lpad(&truncate(&who, who_w), who_w), Styles::dim()),
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

/// "vN: " when the version is known -- a record needs no count, it just is
/// what it is. "last N: " otherwise, where `shown` must be the number of
/// rows actually listed underneath, or the header claims games that were
/// silently dropped.
fn header_label(deployed: &Option<(String, jiff::Timestamp)>, shown: usize) -> String {
    match deployed {
        Some((v, _)) => format!("{v}: "),
        None => format!("last {shown}: "),
    }
}

/// Name, rating and bot-ness in one truncatable string rather than a separate
/// fixed column each -- "beat a 2546" reads very differently depending on
/// whether the 2546 was a bot or a person, and both fields were already
/// sitting unused on `FinishedGame` (`opponent_rating` since this panel's
/// first version, `opponent_is_bot` new tonight from the PGN's own
/// WhiteTitle/BlackTitle header).
fn describe_opponent(name: &str, rating: Option<i32>, is_bot: bool) -> String {
    match (rating, is_bot) {
        (Some(r), true) => format!("{name} ({r}, BOT)"),
        (Some(r), false) => format!("{name} ({r})"),
        (None, true) => format!("{name} (BOT)"),
        (None, false) => name.to_string(),
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

    fn g(result: &str) -> FinishedGame {
        FinishedGame {
            id: None,
            opponent: "Foe".into(),
            opponent_rating: None,
            opponent_is_bot: false,
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

    /// The bug reported live: three abandoned games plus one win rendered
    /// "last 1: 1W 0D 0L" above four listed rows, because the header counted
    /// only decisive results while the list below showed every game. The
    /// header must count what is actually listed.
    #[test]
    fn header_count_matches_the_listed_games_not_just_decisive_ones() {
        use sf_model::state::{BotConfig, BotState};
        use sf_model::{AppState, BotId};
        use std::sync::Arc;
        use std::time::Instant;

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
        bot.results.set(vec![g("win"), g("none"), g("none"), g("none")], now);
        let mut st = AppState::default();
        let id = bot.cfg.id.clone();
        st.bots.insert(id.clone(), bot);
        st.focus = Some(id);

        // Tall enough for the header plus all 4 rows (6 = 2 border + 4 data
        // rows would leave no room for the header row itself; 7 is the real
        // floor).
        let area = Rect { x: 0, y: 0, width: 66, height: 7 };
        let mut buf = Buffer::empty(area);
        let cx = Cx { state: &st, now, wall: jiff::Timestamp::UNIX_EPOCH, bot: None, frame: 0, picture: None };
        PANEL.render(VariantId::Full, area, &mut buf, &cx);
        let text = (0..area.height)
            .map(|y| (0..area.width).map(|x| buf.cell((x, y)).map(|c| c.symbol()).unwrap_or(" ")).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(text.contains("last 4: 1W 0D 0L"), "header must count all 4 listed games, not just the 1 decisive one:\n{text}");
    }

    /// The other half of the same class of bug, found live 2026-07-31: with
    /// MORE history than the panel has room to list, the header must shrink
    /// to match what's actually shown, not claim the full history while
    /// silently cutting rows -- "last 9" over four visible rows reads as the
    /// panel having dropped five games.
    #[test]
    fn header_count_shrinks_to_match_a_panel_too_short_to_list_everything() {
        use sf_model::state::{BotConfig, BotState};
        use sf_model::{AppState, BotId};
        use std::sync::Arc;
        use std::time::Instant;

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
        // 9 games in history, same shape the live report showed.
        bot.results.set(
            vec![g("win"), g("none"), g("none"), g("none"), g("none"), g("none"), g("none"), g("none"), g("none")],
            now,
        );
        let mut st = AppState::default();
        let id = bot.cfg.id.clone();
        st.bots.insert(id.clone(), bot);
        st.focus = Some(id);

        // Inner height 4 (6 - 2 border): room for the header plus 3 data rows.
        let area = Rect { x: 0, y: 0, width: 66, height: 6 };
        let mut buf = Buffer::empty(area);
        let cx = Cx { state: &st, now, wall: jiff::Timestamp::UNIX_EPOCH, bot: None, frame: 0, picture: None };
        PANEL.render(VariantId::Full, area, &mut buf, &cx);
        let text = (0..area.height)
            .map(|y| (0..area.width).map(|x| buf.cell((x, y)).map(|c| c.symbol()).unwrap_or(" ")).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n");

        let listed = text.matches(" Foe").count();
        assert_eq!(listed, 3, "sanity: this height only fits 3 rows:\n{text}");
        assert!(
            text.contains("last 3: 1W 0D 0L"),
            "header must match the 3 rows actually shown, not the 9 in history:\n{text}"
        );
        assert!(!text.contains("last 9"), "must never claim more games than are actually listed:\n{text}");
    }

    /// A lifetime W/L record averages engines that no longer exist -- the
    /// same reasoning `header` already applies to the rating. Requested
    /// directly 2026-07-31: scope the record to the currently deployed
    /// version, not an arbitrary "last N" window that can straddle a
    /// redeploy and blame the new engine for the old one's losses.
    #[test]
    fn record_is_scoped_to_the_deployed_version_not_a_raw_last_n_window() {
        use sf_model::state::{BotConfig, BotState, Version};
        use sf_model::{AppState, BotId};
        use std::sync::Arc;
        use std::time::Instant;

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
        let deployed_at = jiff::Timestamp::from_second(1_000_000).unwrap();

        let mut before = g("loss");
        before.at = Some(deployed_at - jiff::SignedDuration::from_secs(3600));
        let mut after1 = g("win");
        after1.at = Some(deployed_at + jiff::SignedDuration::from_secs(60));
        let mut after2 = g("win");
        after2.at = Some(deployed_at + jiff::SignedDuration::from_secs(120));

        bot.results.set(vec![after2.clone(), after1.clone(), before], now);
        bot.version.set(
            Version { version: "v9".into(), at: Some(deployed_at), ..Default::default() },
            now,
        );

        let mut st = AppState::default();
        let id = bot.cfg.id.clone();
        st.bots.insert(id.clone(), bot);
        st.focus = Some(id);

        let area = Rect { x: 0, y: 0, width: 66, height: 8 };
        let mut buf = Buffer::empty(area);
        let cx = Cx { state: &st, now, wall: jiff::Timestamp::UNIX_EPOCH, bot: None, frame: 0, picture: None };
        PANEL.render(VariantId::Full, area, &mut buf, &cx);
        let text = (0..area.height)
            .map(|y| (0..area.width).map(|x| buf.cell((x, y)).map(|c| c.symbol()).unwrap_or(" ")).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(
            text.contains("v9: 2W 0D 0L"),
            "must count only the 2 games played since v9 deployed, not the loss from before it, and must not show a bare count:\n{text}"
        );
    }

    /// The record must reflect the WHOLE version, not just whichever games
    /// happen to fit on screen -- requested directly 2026-07-31, after the
    /// prior fix (matching the header count to the listed rows) turned out
    /// to be the wrong invariant once the header became "your record on
    /// vN" rather than "the last N games". A version's record must read
    /// the same at any panel height.
    #[test]
    fn versioned_record_counts_every_game_of_that_version_even_past_what_the_panel_can_list() {
        use sf_model::state::{BotConfig, BotState, Version};
        use sf_model::{AppState, BotId};
        use std::sync::Arc;
        use std::time::Instant;

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
        let deployed_at = jiff::Timestamp::from_second(1_000_000).unwrap();

        // 9 games under this version, but the panel below will only have
        // room to list 3 of them.
        let mut games: Vec<FinishedGame> = (0..9)
            .map(|i| {
                let mut game = g("win");
                game.at = Some(deployed_at + jiff::SignedDuration::from_secs(i * 60));
                game
            })
            .collect();
        games.reverse(); // newest first, matching how `read_results` sorts

        bot.results.set(games, now);
        bot.version.set(
            Version { version: "v5".into(), at: Some(deployed_at), ..Default::default() },
            now,
        );

        let mut st = AppState::default();
        let id = bot.cfg.id.clone();
        st.bots.insert(id.clone(), bot);
        st.focus = Some(id);

        // Inner height 4: room for the header plus 3 data rows.
        let area = Rect { x: 0, y: 0, width: 66, height: 6 };
        let mut buf = Buffer::empty(area);
        let cx = Cx { state: &st, now, wall: jiff::Timestamp::UNIX_EPOCH, bot: None, frame: 0, picture: None };
        PANEL.render(VariantId::Full, area, &mut buf, &cx);
        let text = (0..area.height)
            .map(|y| (0..area.width).map(|x| buf.cell((x, y)).map(|c| c.symbol()).unwrap_or(" ")).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(
            text.contains("v5: 9W 0D 0L"),
            "the record must count all 9 games on v5, not just the 3 the panel has room to list:\n{text}"
        );
        assert_eq!(text.matches(" Foe").count(), 3, "sanity: only 3 rows actually fit:\n{text}");
    }

    #[test]
    fn opponent_description_covers_all_four_combinations() {
        assert_eq!(describe_opponent("Foe", Some(2546), true), "Foe (2546, BOT)");
        assert_eq!(describe_opponent("Foe", Some(2546), false), "Foe (2546)");
        assert_eq!(describe_opponent("Foe", None, true), "Foe (BOT)");
        assert_eq!(describe_opponent("Foe", None, false), "Foe");
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
