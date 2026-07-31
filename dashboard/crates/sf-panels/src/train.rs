//! The training run: how far, how fast, and whether anything is watching it.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use sf_layout::widgets::{bar, chrome, put, sparkline, track_tag, truncate};
use sf_layout::{
    Cx, Panel, PanelId, PanelSpec, RegionId, Relevance, Scope, Size, Variant, VariantId, Weight,
};
use sf_model::state::{LogEntry, LogLevel};
use sf_theme::Styles;

pub static PANEL: Train = Train;
pub struct Train;

static SPEC: PanelSpec = PanelSpec {
    id: PanelId("train"),
    title: "training",
    keybind: None,
    help: "the training run: how far, how fast, and whether anything is watching it",
    region: RegionId::Rail,
    scope: Scope::Global,
    weight: Weight::Normal,
    variants: &[
        Variant::new(VariantId::Full, Size::new(40, 6), Size::new(60, 6), 0),
        Variant::new(VariantId::Compact, Size::new(24, 3), Size::new(40, 3), 0),
        Variant::fixed(VariantId::Line, 12, 1),
    ],
};

impl Panel for Train {
    fn spec(&self) -> &'static PanelSpec {
        &SPEC
    }

    /// Hidden when there is no run. The Python showed "no training run active" in a
    /// six-row box forever, which is six rows spent saying nothing.
    fn relevance(&self, cx: &Cx<'_>) -> Relevance {
        match cx.state.train.run.get(cx.now) {
            // A finished run is still worth a glance; a run in flight with nothing
            // watching it is worth interrupting for.
            Some((r, _)) if r.step > 0 && r.process_alive => {
                if supervised(cx) { Relevance::Normal } else { Relevance::Urgent }
            }
            Some((r, _)) if r.step > 0 => Relevance::Idle,
            _ => Relevance::Hidden,
        }
    }

    /// Both `Full` and `Compact` declare `grow: 0` -- a fixed height, never
    /// expanded by leftover space -- so this dynamic ask is the only lever this
    /// panel has for a row it does not always need. A recent warning or error
    /// earns one more row at the bottom: the earliest sign of trouble the
    /// structured telemetry stream does not carry at all (a CUDA OOM, a
    /// traceback, a data-pipeline exception), absorbed from what `sumofish
    /// train`'s raw `journalctl -f` used to show.
    ///
    /// The solver's formula is `final_min = max(min.h, rows + (min.h - 1))`.
    /// The declared `Full` minimum (6 rows / 4 content rows after the border)
    /// already sits one short of `grad`, the panel's own 5th content row --
    /// `render` gates it on `inner.height > 4`, which the fixed 6-row height
    /// never satisfies, so it has been silently unreachable in the live layout
    /// since it was added. `rows: 3` asks for `min.h + 2`: one row to actually
    /// reach `grad`, one for the new problem line right after it.
    fn rows_needed(&self, variant: VariantId, cx: &Cx<'_>) -> Option<u16> {
        if variant != VariantId::Full {
            return None;
        }
        last_problem(cx).map(|_| 3)
    }

    fn render(&self, variant: VariantId, area: Rect, buf: &mut Buffer, cx: &Cx<'_>) {
        let Some((run, track)) = cx.state.train.run.get(cx.now) else { return };
        let track = Cx::track(track);

        if variant == VariantId::Line {
            let pct = run.progress().map(|p| format!("{:.0}%", p * 100.0)).unwrap_or_else(|| "?".into());
            put(buf, area, area.y, Line::from(Span::styled(format!("train {} {pct}", truncate(&run.name, 12)), Styles::dim())));
            return;
        }

        let inner = chrome(area, SPEC.title, buf);
        if inner.height == 0 {
            return;
        }

        let mut row = vec![
            Span::styled(format!("{}  ", truncate(&run.name, 18)), Styles::ink(sf_theme::COOL)),
            Span::styled(run.step.to_string(), Styles::body().add_modifier(Modifier::BOLD)),
        ];
        // The total comes from the RUN's own config, never a constant. The Python
        // hardcoded 300k steps and a batch of 1024; the live run is 400k at 256, so
        // its bar read 13.5% instead of 10.1% and its ETA said 76.5h where the
        // arithmetic gives ~26.4h.
        match run.total_steps {
            Some(t) => row.push(Span::styled(format!("/{t}  "), Styles::dim())),
            None => row.push(Span::styled("/?  ", Styles::faint())),
        }
        match run.eta() {
            Some(d) => row.push(Span::styled(
                format!("eta {:.1}h  ", d.as_secs_f64() / 3600.0),
                Styles::dim(),
            )),
            None => row.push(Span::styled("eta unknown  ", Styles::faint())),
        }
        row.push(track_tag(track, ""));
        put(buf, inner, inner.y, Line::from(row));

        // A long run with nothing watching it is worth shouting about -- but only
        // when there IS a run. An earlier version keyed this on whether the training
        // process was alive, so it shouted UNSUPERVISED at every finished run, which
        // is both wrong and the kind of false alarm that teaches you to ignore the
        // panel. `train_watchdog.py` was separately fixed on 2026-07-29 to watch the
        // process rather than a unit that no longer exists, so the honest check is
        // whether its timer is armed.
        if run.process_alive && !supervised(cx) && inner.height > 1 {
            put(
                buf,
                inner,
                inner.y + 1,
                Line::from(Span::styled(
                    "UNSUPERVISED \u{2014} no stall detector armed for this run",
                    Styles::ink(sf_theme::BAD),
                )),
            );
        } else if inner.height > 1 {
            if !run.process_alive {
                put(
                    buf,
                    inner,
                    inner.y + 1,
                    Line::from(Span::styled("run is not active", Styles::faint())),
                );
            } else if let Some(f) = run.progress() {
                let w = sf_theme::GAUGE_MAX.min(inner.width.saturating_sub(8).max(8));
                let mut spans = bar(f, w, sf_theme::COOL, sf_theme::BG_SOFT).spans;
                spans.push(Span::styled(format!(" {:4.1}%", f * 100.0), Styles::dim()));
                put(buf, inner, inner.y + 1, Line::from(spans));
            }
        }

        if variant == VariantId::Compact || inner.height < 3 {
            return;
        }
        let w = sf_theme::GAUGE_MAX.min(inner.width.saturating_sub(30).max(8));
        if let Some(loss) = run.loss {
            let (spark, lo, hi) = sparkline(&run.loss_hist, w, sf_theme::INFO);
            let mut spans = vec![
                Span::styled("loss ", Styles::dim()),
                Span::styled(format!("{loss:.4} "), Styles::body()),
            ];
            spans.extend(spark.spans);
            spans.push(Span::styled(format!(" {lo:.3}-{hi:.3}"), Styles::faint()));
            put(buf, inner, inner.y + 2, Line::from(spans));
        }

        // Held-out loss, which is what `best.pt` is actually selected on and what
        // PHILOSOPHY calls the honest number. The Python collected it and plotted
        // the noisy puzzle accuracy instead.
        if inner.height > 3 {
            match run.val_loss {
                Some(v) => {
                    let vals: Vec<f64> = run.val_hist.iter().map(|(_, v)| *v).collect();
                    let (spark, lo, hi) = sparkline(&vals, w, sf_theme::ACCENT);
                    let mut spans = vec![
                        Span::styled("val  ", Styles::dim()),
                        Span::styled(format!("{v:.4} "), Styles::ink(sf_theme::ACCENT)),
                    ];
                    spans.extend(spark.spans);
                    spans.push(Span::styled(format!(" {lo:.3}-{hi:.3}"), Styles::faint()));
                    put(buf, inner, inner.y + 3, Line::from(spans));
                }
                None => put(
                    buf,
                    inner,
                    inner.y + 3,
                    Line::from(Span::styled("no held-out loss yet", Styles::faint())),
                ),
            }
        }

        // A grad-norm spike is the earliest visible sign a long run is going wrong,
        // hours before val_loss moves.
        if inner.height > 4 {
            if let Some(g) = run.grad_norm {
                put(
                    buf,
                    inner,
                    inner.y + 4,
                    Line::from(vec![
                        Span::styled("grad ", Styles::dim()),
                        Span::styled(
                            format!("{g:.2}"),
                            if g > 20.0 { Styles::ink(sf_theme::BAD) } else { Styles::body() },
                        ),
                        Span::styled(
                            run.lr.map(|l| format!("    lr {l:.2e}")).unwrap_or_default(),
                            Styles::dim(),
                        ),
                    ]),
                );
            }
        }

        // The raw journal's own most recent warning or error, beyond what the
        // structured telemetry stream reports at all. `inner.height > 5` is
        // exactly what `rows_needed`'s `Some(3)` guarantees when there is one.
        if inner.height > 5 {
            if let Some(p) = last_problem(cx) {
                let (label, ink) = if p.level == LogLevel::Error {
                    ("error ", sf_theme::BAD)
                } else {
                    ("warn  ", sf_theme::WARM)
                };
                put(
                    buf,
                    inner,
                    inner.y + 5,
                    Line::from(vec![
                        Span::styled(label, Styles::ink(ink)),
                        Span::styled(
                            truncate(&p.text, inner.width.saturating_sub(7) as usize),
                            Styles::dim(),
                        ),
                    ]),
                );
            }
        }
    }
}

/// The most recent warning or error line from the training unit's own journal.
fn last_problem<'a>(cx: &Cx<'a>) -> Option<&'a LogEntry> {
    cx.state.train.log.iter_rev().find(|e| e.level != LogLevel::Info)
}

/// Is a stall detector actually armed for the training run?
///
/// The timer must be both loaded and active. `not-found` is the state
/// `sumofish-train.service` sat in for weeks while its watchdog logged "nothing to
/// watch" every five minutes, which is exactly the failure this asks about.
fn supervised(cx: &Cx<'_>) -> bool {
    cx.state.machine.units.get(cx.now).is_some_and(|(units, _)| {
        units.iter().any(|u| u.name.contains("train-watchdog") && u.healthy())
    })
}
