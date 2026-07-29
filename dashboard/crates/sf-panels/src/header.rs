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
    region: RegionId::Band,
    scope: Scope::Global,
    weight: Weight::Pinned,
    variants: &[
        // Three rows: the badge line, the ratings line, and a rule. No border,
        // because a full-width box around three rows spends two of them on chrome.
        Variant::fixed(VariantId::Full, 40, 3),
        // Two rows on a short terminal.
        Variant::fixed(VariantId::Compact, 24, 2),
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

        // The version the record is scoped to, because a lifetime record averages
        // engines that no longer exist.
        if let Some((v, _)) = bot.and_then(|b| b.version.get(cx.now)) {
            left.push(Span::raw("  "));
            left.push(Span::styled(format!("in {}", v.version), Styles::accent()));
        }

        put(buf, area, area.y, Line::from(left));
        if variant == VariantId::Line || area.height < 2 {
            return;
        }

        // Row 1: ratings, then the record.
        let mut mid: Vec<Span> = Vec::new();
        if let Some((acct, t)) = bot.and_then(|b| b.account.get(cx.now)) {
            let style = Styles::track(Cx::track(t));
            for (perf, p) in acct.perfs.iter() {
                if p.games == 0 {
                    continue;
                }
                mid.push(Span::styled(format!(" {perf} "), Styles::dim()));
                mid.push(Span::styled(p.rating.to_string(), style));
                // A provisional rating and a settled one are different claims.
                if p.provisional {
                    mid.push(Span::styled("?", Styles::ink(sf_theme::WARM)));
                } else if p.rd > 0 {
                    mid.push(Span::styled(format!("\u{00b1}{}", p.rd), Styles::faint()));
                }
            }
        }
        if let Some((rec, _)) = bot.and_then(|b| b.record.get(cx.now)) {
            mid.push(Span::styled("   ", Styles::body()));
            mid.push(Span::styled(format!("{}W", rec.wins), Styles::ink(sf_theme::GOOD)));
            mid.push(Span::styled(format!(" {}D", rec.draws), Styles::dim()));
            mid.push(Span::styled(format!(" {}L", rec.losses), Styles::ink(sf_theme::BAD)));
        }
        if mid.is_empty() {
            mid.push(Span::styled("waiting for lichess\u{2026}", Styles::faint()));
        }
        put(buf, area, area.y + 1, Line::from(mid));

        if variant == VariantId::Compact || area.height < 3 {
            return;
        }
        put(
            buf,
            area,
            area.y + 2,
            Line::from(Span::styled("\u{2500}".repeat(area.width as usize), Styles::border())),
        );
    }
}

/// Declaration-shaped helper so the spec above reads as data.
#[allow(dead_code)]
const fn sz(w: u16, h: u16) -> Size {
    Size::new(w, h)
}
