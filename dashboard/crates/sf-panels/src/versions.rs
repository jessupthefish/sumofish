//! What is deployed, and whether it is what this version shipped.
//!
//! `VERSIONS.jsonl` ties a rating to a code sha AND a checkpoint hash, and the
//! Python read exactly two of its fields -- the timestamp, as a cutoff, and the
//! label. `checkpoint_sha` is the only thing on disk that answers "is the bot
//! playing what v1 shipped?", and promotion swaps `runs/value.pt` under a running
//! bot with nothing on screen saying so.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use sf_layout::widgets::{chrome, put, truncate};
use sf_layout::{
    Cx, Panel, PanelId, PanelSpec, RegionId, Relevance, Scope, Size, Variant, VariantId, Weight,
};
use sf_theme::Styles;

pub static PANEL: Versions = Versions;
pub struct Versions;

static SPEC: PanelSpec = PanelSpec {
    id: PanelId("versions"),
    title: "deployed",
    keybind: None,
    help: "what's deployed, and whether it's what this version shipped",
    region: RegionId::RailB,
    scope: Scope::FocusedBot,
    weight: Weight::Low,
    variants: &[
        Variant::new(VariantId::Full, Size::new(38, 4), Size::new(56, 8), 0),
        Variant::new(VariantId::Compact, Size::new(20, 2), Size::new(38, 3), 0),
        Variant::fixed(VariantId::Line, 10, 1),
    ],
};

impl Panel for Versions {
    fn spec(&self) -> &'static PanelSpec {
        &SPEC
    }

    fn relevance(&self, cx: &Cx<'_>) -> Relevance {
        match cx.bot_state().and_then(|b| b.version.get(cx.now)) {
            Some(_) => Relevance::Idle,
            None => Relevance::Hidden,
        }
    }

    fn render(&self, variant: VariantId, area: Rect, buf: &mut Buffer, cx: &Cx<'_>) {
        let Some(bot) = cx.bot_state() else { return };
        let Some((v, _)) = bot.version.get(cx.now) else { return };

        if variant == VariantId::Line {
            put(buf, area, area.y, Line::from(Span::styled(format!("{} deployed", v.version), Styles::dim())));
            return;
        }
        let inner = chrome(area, SPEC.title, buf);
        if inner.height == 0 {
            return;
        }
        let mut y = inner.y;
        put(
            buf,
            inner,
            y,
            Line::from(vec![
                Span::styled(
                    format!("{} ", v.version),
                    Styles::accent().add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    truncate(&v.title, inner.width.saturating_sub(6) as usize),
                    Styles::body(),
                ),
            ]),
        );
        y += 1;

        // The checkpoint this version shipped, against what the running engine
        // actually booted with. A drift means a promotion landed mid-session.
        let running = cx
            .state
            .engines
            .values()
            .filter(|e| e.bot.as_ref() == Some(&bot.cfg.id))
            .filter_map(|e| e.boot.as_ref())
            .max_by_key(|b| b.value_step.unwrap_or(0));

        if y < inner.y + inner.height {
            let mut row = vec![Span::styled("checkpoint ", Styles::dim())];
            match (&v.checkpoint_sha, v.step) {
                (Some(sha), Some(step)) => {
                    row.push(Span::styled(format!("{} step {step}", truncate(sha, 12)), Styles::body()))
                }
                _ => row.push(Span::styled("unknown", Styles::faint())),
            }
            put(buf, inner, y, Line::from(row));
            y += 1;
        }

        if y < inner.y + inner.height {
            let mut row = vec![Span::styled("running    ", Styles::dim())];
            match running {
                Some(b) => {
                    let drift = v.step.is_some() && b.value_step.is_some() && v.step != b.value_step;
                    row.push(Span::styled(
                        format!(
                            "value step {}  {}M params",
                            b.value_step.map(|s| s.to_string()).unwrap_or_else(|| "?".into()),
                            b.params.map(|p| p / 1_000_000).unwrap_or(0)
                        ),
                        if drift { Styles::ink(sf_theme::WARM) } else { Styles::body() },
                    ));
                    // Which search core. The rollback is one environment variable
                    // and a restart, so this is a fact about the running bot rather
                    // than about the repo.
                    if let Some(core) = b.core.as_deref() {
                        row.push(Span::styled(
                            format!("  core {core}"),
                            if core == "rust" {
                                Styles::ink(sf_theme::COOL)
                            } else {
                                Styles::dim()
                            },
                        ));
                    }
                    if drift {
                        row.push(Span::styled("  DRIFT", Styles::ink(sf_theme::BAD)));
                    }
                }
                None => row.push(Span::styled("no engine has booted yet", Styles::faint())),
            }
            put(buf, inner, y, Line::from(row));
        }
    }
}
