//! The header: who is playing, what they are rated, and whether anything is
//! actually connected.
//!
//! Three rows, full width, pinned. Pinned because without it there is nothing on
//! screen that says which bot you are looking at, which matters the moment there
//! is more than one.
//!
//! Two things the Python version did that this one must not:
//!
//! - It read and YAML-parsed `config/lichess-bot.yml` from inside the render, at
//!   twelve times a second, to find which time controls were enabled. Config
//!   belongs in `AppState`, put there once by a source.
//! - It hardcoded `" SUMOFISH "` as the badge text and `"SumoFish"` as the
//!   username in five other places, one of which the `--user` flag could not
//!   reach. The name comes from config, always.
//!
//! Also carries the one aggregate "something, somewhere, is stale" badge
//! (`sf_model::state::health`), so a degraded field on a panel that is not
//! currently on screen is not invisible until someone happens to look at it.
//! It composes only, no new I/O: every `Tracked` field already knows its own
//! freshness.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use sf_layout::widgets::{put, truncate};
use sf_layout::{
    Cx, Panel, PanelId, PanelSpec, RegionId, Relevance, Scope, Size, Variant, VariantId, Weight,
};
use sf_theme::{Styles, Track};

pub static PANEL: Header = Header;

pub struct Header;

static SPEC: PanelSpec = PanelSpec {
    id: PanelId("header"),
    title: "header",
    keybind: None,
    help: "who's playing, ratings, and whether lichess is actually connected",
    region: RegionId::Band,
    scope: Scope::Global,
    weight: Weight::Pinned,
    variants: &[
        // Two rows: the badge line and a rule. Per-control ratings and the W/D/L
        // record used to live on a second row here; both are the `rating` panel's
        // job now (current number there per control, plus the trend and history
        // this row never had room for), not duplicated in the one thing that's
        // pinned and always on screen regardless of which bot is focused.
        Variant::fixed(VariantId::Full, 40, 2),
        // One row on a short terminal -- the same badge row Line shows.
        Variant::fixed(VariantId::Compact, 24, 1),
        // One row when that is all there is.
        Variant::fixed(VariantId::Line, 12, 1),
    ],
};

impl Panel for Header {
    fn spec(&self) -> &'static PanelSpec {
        &SPEC
    }

    fn relevance(&self, _cx: &Cx<'_>) -> Relevance {
        // Always. See the module docs.
        Relevance::Urgent
    }

    fn render(&self, variant: VariantId, area: Rect, buf: &mut Buffer, cx: &Cx<'_>) {
        let bot = cx.bot_state();
        let name = bot.map(|b| b.cfg.user.as_str()).unwrap_or("no bot");
        let label = bot.map(|b| b.cfg.label.as_str()).unwrap_or("");

        // Row 0: the badge, the bot's own label, and the connection track.
        let mut left: Vec<Span> = vec![
            Span::styled(format!(" {} ", name.to_uppercase()), Styles::badge()),
            Span::raw(" "),
        ];
        if !label.is_empty() && label != name {
            left.push(Span::styled(truncate(label, 24), Styles::dim()));
            left.push(Span::raw(" "));
        }

        // Whether lichess is answering at all, and about which account. The
        // failure this reports is the one the whole track model exists for: a dead
        // connection must not render identically to a live one.
        let (acct_track, acct_note) = match bot.map(|b| (b.account.get(cx.now), b.account.error())) {
            Some((Some((acct, t)), _)) => (
                Cx::track(t),
                if acct.username.eq_ignore_ascii_case(name) {
                    String::new()
                } else {
                    // The token and the configured username disagree. In the
                    // Python these were asserted independently and never
                    // reconciled, so pointing `--user` at the wrong account made
                    // every colour test silently fail to match.
                    format!(" token is {}!", acct.username)
                },
            ),
            Some((None, Some(err))) => (Track::Lost, format!(" {}", truncate(err, 28))),
            _ => (Track::Lost, " no account".to_string()),
        };
        left.push(Span::styled(
            format!("{} lichess", acct_track.glyph()),
            Styles::track(acct_track),
        ));
        if !acct_note.is_empty() {
            left.push(Span::styled(acct_note, Styles::ink(sf_theme::BAD)));
        }

        // One aggregate glance at every `Tracked` field in `AppState`, not just
        // this bot's lichess connection. Pure composition of data every panel
        // already carries -- see `sf_model::state::health`. Nothing shown at
        // all when everything is live, so this can only ever ADD a reason to
        // look, never noise on a healthy screen.
        let health = sf_model::state::health(cx.state, cx.now);
        if health.degraded > 0 {
            let track = Cx::track(health.worst);
            left.push(Span::raw("  "));
            left.push(Span::styled(
                format!("{} {} degraded", track.glyph(), health.degraded),
                Styles::track(track),
            ));
        }

        // The version the record is scoped to, because a lifetime record averages
        // engines that no longer exist.
        if let Some((v, _)) = bot.and_then(|b| b.version.get(cx.now)) {
            left.push(Span::raw("  "));
            left.push(Span::styled(format!("in {}", v.version), Styles::accent()));
        }

        put(buf, area, area.y, Line::from(left));
        if variant != VariantId::Full || area.height < 2 {
            return;
        }

        // Row 1: a rule, and nothing else. Per-control ratings and the W/D/L
        // record moved to the `rating` panel -- see the variants doc comment.
        put(
            buf,
            area,
            area.y + 1,
            Line::from(Span::styled("\u{2500}".repeat(area.width as usize), Styles::border())),
        );
    }
}

/// Declaration-shaped helper so the spec above reads as data.
#[allow(dead_code)]
const fn sz(w: u16, h: u16) -> Size {
    Size::new(w, h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::buffer::Buffer;
    use sf_model::state::{Account, BotConfig, BotState};
    use sf_model::{AppState, BotId};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    fn render_at(st: &AppState, now: Instant) -> String {
        let area = Rect { x: 0, y: 0, width: 70, height: 3 };
        let mut buf = Buffer::empty(area);
        let cx = Cx { state: st, now, wall: jiff::Timestamp::UNIX_EPOCH, bot: None, frame: 0, picture: None };
        PANEL.render(VariantId::Full, area, &mut buf, &cx);
        (0..area.height)
            .map(|y| (0..area.width).map(|x| buf.cell((x, y)).map(|c| c.symbol()).unwrap_or(" ")).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn bot() -> (BotId, BotState) {
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
        (cfg.id.clone(), BotState::new(cfg))
    }

    /// A freshly-fed bot shows no aggregate badge at all: "all live" means
    /// nothing shown, not a green dot competing for the same row.
    #[test]
    fn no_badge_when_everything_is_live() {
        let base = Instant::now();
        let (id, mut b) = bot();
        b.account.set(Account::default(), base);
        let mut st = AppState::default();
        st.bots.insert(id.clone(), b);
        st.focus = Some(id);

        let text = render_at(&st, base);
        assert!(!text.contains("degraded"), "fresh state should not show the badge:\n{text}");
    }

    /// The core acceptance case from the task: a field fed once and then
    /// abandoned must surface as "N degraded" on the header, composed purely
    /// from the existing per-field Track, with no new I/O.
    #[test]
    fn stale_field_surfaces_as_a_degraded_badge() {
        let base = Instant::now();
        let (id, mut b) = bot();
        b.account.set(Account::default(), base);
        let mut st = AppState::default();
        st.bots.insert(id.clone(), b);
        st.focus = Some(id);

        // account's interval is 45s; well past 6x puts it at Lost.
        let text = render_at(&st, base + Duration::from_secs(400));
        assert!(text.contains("1 degraded"), "expected a degraded badge:\n{text}");
    }
}
