//! Shared drawing primitives, so panels do not each re-derive their own chrome
//! arithmetic.
//!
//! The Python had nine separate open-coded interior expressions (`width - 4`,
//! `height - 8`, `height - 4 - (1 if curve else 0)`, `7 + 2 + 1 + 8 + 12`, ...)
//! and its own comment records the same off-by-one shipping twice from that
//! pattern: "`time forfeit` became `time forfei` twice: once from an 11-char
//! slice, and again from an off-by-one here". One `chrome()` and one `truncate()`
//! is the fix.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Widget};
use sf_theme::{BLOCKS, Ink, SPARK, Styles, Track};

/// Draw a panel border and title, and return the interior. **The one place that
/// knows what chrome costs.**
pub fn chrome(area: Rect, title: &str, buf: &mut Buffer) -> Rect {
    chrome_styled(area, title, Styles::border(), buf)
}

pub fn chrome_styled(area: Rect, title: &str, border: Style, buf: &mut Buffer) -> Rect {
    if area.width < 2 || area.height < 2 {
        return area;
    }
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border)
        .title(Line::from(Span::styled(format!(" {title} "), Styles::title())));
    let inner = block.inner(area);
    block.render(area, buf);
    inner
}

/// Clip to `width` display columns, marking the cut so a truncated value never
/// looks like a complete one. A silently clipped "time forfeit" reads as
/// "time forfei", which is a different and plausible-looking word.
pub fn truncate(s: &str, width: usize) -> String {
    let n = s.chars().count();
    if n <= width {
        return s.to_string();
    }
    if width == 0 {
        return String::new();
    }
    if width == 1 {
        return "\u{2026}".into();
    }
    let mut out: String = s.chars().take(width - 1).collect();
    out.push('\u{2026}');
    out
}

/// Write a line at `y`, clipped to the area. Nothing else may write text.
pub fn put(buf: &mut Buffer, area: Rect, y: u16, line: Line<'_>) {
    if y >= area.y + area.height {
        return;
    }
    buf.set_line(area.x, y, &line, area.width);
}

/// Fit a run of spans to `width`, marking the cut.
///
/// `put` clips silently, which is right for text that is allowed to run out but
/// wrong for a *list*: a row of unit dots clipped mid-name reads as though that
/// unit is the last one, and the reader has no way to know two more exist. Same
/// principle as `truncate`, one level up: a clipped thing must say it was clipped.
pub fn fit(spans: Vec<Span<'static>>, width: u16) -> Line<'static> {
    let total: usize = spans.iter().map(|s| s.content.chars().count()).sum();
    if total <= width as usize {
        return Line::from(spans);
    }
    let budget = (width as usize).saturating_sub(1);
    let mut out: Vec<Span<'static>> = Vec::new();
    let mut used = 0usize;
    for s in spans {
        let n = s.content.chars().count();
        if used + n <= budget {
            used += n;
            out.push(s);
        } else {
            let room = budget - used;
            if room > 0 {
                let head: String = s.content.chars().take(room).collect();
                out.push(Span::styled(head, s.style));
            }
            break;
        }
    }
    out.push(Span::styled("\u{2026}", Styles::faint()));
    Line::from(out)
}

/// A horizontal bar in eighth-cell resolution, with an explicit ground.
///
/// Both halves are always painted. A bar whose empty half is the panel colour is
/// unreadable: the ground was #32302f on a #282828 panel, 1.12:1, so a gauge at
/// 95% looked like a stripe floating in space rather than a bar filled nearly to
/// the top, and there was no way to tell a full gauge from an absent one.
pub fn bar(fraction: f64, width: u16, fill: Ink, ground: Ink) -> Line<'static> {
    let width = width.max(1);
    let f = fraction.clamp(0.0, 1.0);
    let eighths = (f * width as f64 * 8.0).round() as u32;
    let full = (eighths / 8) as u16;
    let rem = (eighths % 8) as usize;

    let mut spans = Vec::new();
    if full > 0 {
        spans.push(Span::styled(
            "\u{2588}".repeat(full as usize),
            Style::new().fg(fill.color()).bg(ground.color()),
        ));
    }
    let mut used = full;
    if rem > 0 && used < width {
        spans.push(Span::styled(
            BLOCKS[rem].to_string(),
            Style::new().fg(fill.color()).bg(ground.color()),
        ));
        used += 1;
    }
    if used < width {
        spans.push(Span::styled(
            " ".repeat((width - used) as usize),
            Style::new().bg(ground.color()),
        ));
    }
    Line::from(spans)
}

/// A sparkline over the last `width` samples, scaled to its own extent. Returns
/// the line plus the range it was scaled over, because a trend without its bounds
/// is decoration.
pub fn sparkline(values: &[f64], width: u16, ink: Ink) -> (Line<'static>, f64, f64) {
    if values.is_empty() || width == 0 {
        return (Line::from(""), 0.0, 0.0);
    }
    let take = values.len().min(width as usize);
    let slice = &values[values.len() - take..];
    let lo = slice.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = slice.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = (hi - lo).max(f64::EPSILON);
    // Every styled run names its background explicitly. `rich` redraws by diffing
    // frames, and a block painted with an implicit background leaves the previous
    // frame's colour behind when the new frame is narrower -- which is why the old
    // sparklines read as static rather than as a trend.
    let style = Style::new().fg(ink.color()).bg(sf_theme::BG.color());
    let text: String = slice
        .iter()
        .map(|v| {
            let idx = (((v - lo) / span) * (SPARK.len() - 1) as f64).round() as usize;
            SPARK[idx.min(SPARK.len() - 1)]
        })
        .collect();
    (Line::from(Span::styled(text, style)), lo, hi)
}

/// The freshness marker every panel prints beside a number.
pub fn track_tag(track: Track, label: &str) -> Span<'static> {
    let text = if label.is_empty() {
        track.glyph().to_string()
    } else {
        format!("{} {label}", track.glyph())
    };
    Span::styled(text, Styles::track(track))
}

/// Right-align `s` in `width` columns.
pub fn rpad(s: &str, width: usize) -> String {
    let n = s.chars().count();
    if n >= width { s.to_string() } else { format!("{}{s}", " ".repeat(width - n)) }
}

/// Left-align `s` in `width` columns.
pub fn lpad(s: &str, width: usize) -> String {
    let n = s.chars().count();
    if n >= width { s.to_string() } else { format!("{s}{}", " ".repeat(width - n)) }
}

/// A duration as a chess clock: `1:23:45`, `12:34`, or `9.8` under ten seconds,
/// because the tenths matter exactly then and nowhere else.
pub fn clockstr(d: std::time::Duration) -> String {
    let secs = d.as_secs_f64();
    if secs < 10.0 {
        return format!("{secs:.1}");
    }
    let s = secs as u64;
    let (h, m, sec) = (s / 3600, (s % 3600) / 60, s % 60);
    if h > 0 { format!("{h}:{m:02}:{sec:02}") } else { format!("{m}:{sec:02}") }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_marks_the_cut() {
        assert_eq!(truncate("time forfeit", 12), "time forfeit");
        // The Python's bug: an 11-char slice silently produced "time forfei".
        assert_eq!(truncate("time forfeit", 11), "time forfe\u{2026}");
        assert_eq!(truncate("abc", 1), "\u{2026}");
        assert_eq!(truncate("abc", 0), "");
    }

    #[test]
    fn truncate_counts_characters_not_bytes() {
        assert_eq!(truncate("♞♞♞♞", 4), "♞♞♞♞");
        assert_eq!(truncate("♞♞♞♞", 3), "♞♞\u{2026}");
    }

    /// A bar must always be exactly `width` cells: a short one leaves cells that
    /// nothing owns, and the previous frame survives in them.
    #[test]
    fn a_bar_is_always_exactly_its_width() {
        for width in 1..=60u16 {
            for pct in 0..=100 {
                let line = bar(pct as f64 / 100.0, width, sf_theme::EVAL_FILL, sf_theme::EVAL_GROUND);
                let cells: usize = line.spans.iter().map(|s| s.content.chars().count()).sum();
                assert_eq!(cells, width as usize, "width {width} at {pct}% gave {cells} cells");
            }
        }
    }

    #[test]
    fn a_bar_paints_its_ground_even_when_empty() {
        let line = bar(0.0, 10, sf_theme::EVAL_FILL, sf_theme::EVAL_GROUND);
        assert!(!line.spans.is_empty(), "an empty gauge must still show its extent");
        assert!(
            line.spans.iter().all(|s| s.style.bg == Some(sf_theme::EVAL_GROUND.color())),
            "every cell needs an explicit background"
        );
    }

    #[test]
    fn a_full_bar_is_distinguishable_from_an_absent_one() {
        let full = bar(1.0, 8, sf_theme::EVAL_FILL, sf_theme::EVAL_GROUND);
        let empty = bar(0.0, 8, sf_theme::EVAL_FILL, sf_theme::EVAL_GROUND);
        let text = |l: &Line| -> String { l.spans.iter().map(|s| s.content.as_ref()).collect() };
        assert_ne!(text(&full), text(&empty));
    }

    #[test]
    fn out_of_range_fractions_clamp_rather_than_overflow() {
        for f in [-5.0, -0.001, 1.001, 12.0, f64::NAN] {
            let line = bar(f, 10, sf_theme::EVAL_FILL, sf_theme::EVAL_GROUND);
            let cells: usize = line.spans.iter().map(|s| s.content.chars().count()).sum();
            assert_eq!(cells, 10, "fraction {f} broke the width");
        }
    }

    #[test]
    fn sparkline_reports_the_range_it_scaled_over() {
        let (line, lo, hi) = sparkline(&[1.0, 2.0, 3.0], 8, sf_theme::DIM);
        assert_eq!(lo, 1.0);
        assert_eq!(hi, 3.0);
        let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(text.chars().count(), 3);
    }

    #[test]
    fn sparkline_survives_a_flat_series() {
        let (line, lo, hi) = sparkline(&[5.0; 6], 10, sf_theme::DIM);
        assert_eq!((lo, hi), (5.0, 5.0));
        let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(text.chars().count(), 6, "a flat series must not divide by zero");
    }

    #[test]
    fn clocks_read_like_a_chess_clock() {
        use std::time::Duration;
        assert_eq!(clockstr(Duration::from_millis(9800)), "9.8");
        assert_eq!(clockstr(Duration::from_secs(75)), "1:15");
        assert_eq!(clockstr(Duration::from_secs(3725)), "1:02:05");
    }
}

// ---------------------------------------------------------------- charts

/// The conventional pawn scale, from a win probability.
///
/// `400 * log10(p/(1-p)) / 100` — the inverted Elo logistic, and the same mapping
/// `search_engine.centipawns` uses for UCI `score cp`. Reimplemented rather than
/// imported, because importing it would pull in torch.
///
/// **Signed from our own point of view**: `+` means SumoFish is better whichever
/// colour it is pushing, deliberately not the White-relative sign a chess GUI uses.
/// The caller has already converted through `GameState::ours`, so the frame is
/// decided in one place.
///
/// Monotone in `wp`, which is what makes it safe to print beside the gauge: the two
/// cannot tell different stories.
pub fn advantage(wp: f64) -> String {
    let p = wp.clamp(1e-4, 1.0 - 1e-4);
    format!("{:+.2}", 400.0 * (p / (1.0 - p)).log10() / 100.0)
}

/// A vertical gauge, eighths of a cell, filling from the bottom.
///
/// `None` returns completely **unstyled** rows -- nothing at all, rather than a dark
/// column beside the board saying something it does not know.
///
/// Eighths and not halves: most of a game is spent between 0.45 and 0.60, which on a
/// 70-row half-block bar is twenty positions, so small changes land on the same
/// cell and the gauge looks frozen. Eighths give 560.
pub fn evalbar(value: Option<f64>, height: u16, width: u16) -> Vec<Line<'static>> {
    let (height, width) = (height.max(1), width.max(1));
    let Some(v) = value else {
        return (0..height).map(|_| Line::from(" ".repeat(width as usize))).collect();
    };
    let filled = (v.clamp(0.0, 1.0) * height as f64 * 8.0).round() as i64;
    (0..height)
        .map(|r| {
            let base = (height - 1 - r) as i64 * 8;
            let k = (filled - base).clamp(0, 8) as usize;
            let (ch, style) = match k {
                0 => (' ', Style::new().bg(sf_theme::EVAL_GROUND.color())),
                8 => (' ', Style::new().bg(sf_theme::EVAL_FILL.color())),
                k => (
                    EIGHTHS_V[k],
                    Style::new()
                        .fg(sf_theme::EVAL_FILL.color())
                        .bg(sf_theme::EVAL_GROUND.color()),
                ),
            };
            Line::from(Span::styled(ch.to_string().repeat(width as usize), style))
        })
        .collect()
}

/// Eighth-height blocks, filling upward.
const EIGHTHS_V: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// A braille line chart on a **fixed 0..1 scale**.
///
/// Never auto-scaled, and that is the point. Renormalising to the game's own range
/// makes a dead-level draw look like a collapse, which is the standard way an
/// evaluation chart lies. Vertical distance from the midline has to mean the same
/// thing in every game.
///
/// Braille gives a 2x4 dot grid per cell: four times the vertical and twice the
/// horizontal resolution of half-blocks. The cost is one colour per cell, so the
/// midline is drawn only where the trace is not.
pub fn curve_chart(values: &[f64], width: u16, height: u16, mid: f64) -> Vec<Line<'static>> {
    let (width, height) = (width.max(1), height.max(1));
    if values.len() < 2 {
        return (0..height).map(|_| Line::from("")).collect();
    }
    let (px_w, px_h) = (width as usize * 2, height as usize * 4);
    let samples = resample(values, px_w);

    // dots[cell_y][cell_x] is a bitmask of the 8 braille dots.
    let mut dots = vec![vec![0u8; width as usize]; height as usize];
    let mut colour = vec![vec![None::<Ink>; width as usize]; height as usize];

    let y_of = |v: f64| -> usize {
        (((1.0 - v.clamp(0.0, 1.0)) * (px_h - 1) as f64).round() as usize).min(px_h - 1)
    };
    let mut prev_y = y_of(samples[0]);
    for (x, v) in samples.iter().enumerate() {
        let y = y_of(*v);
        // Join consecutive samples, so the trace is continuous rather than a dotted
        // scatter that the eye has to interpolate for itself.
        let (lo, hi) = if y < prev_y { (y, prev_y) } else { (prev_y, y) };
        for yy in lo..=hi {
            let (cx, cy) = (x / 2, yy / 4);
            if cy < dots.len() && cx < dots[cy].len() {
                dots[cy][cx] |= braille_dot(x % 2, yy % 4);
                colour[cy][cx] =
                    Some(if *v >= mid { sf_theme::GOOD } else { sf_theme::BAD });
            }
        }
        prev_y = y;
    }

    // The midline, only in cells the trace does not occupy: a braille cell carries
    // one foreground colour, so the two cannot share.
    let mid_y = y_of(mid);
    let (mcx, mcy) = (mid_y % 4, mid_y / 4);
    if mcy < dots.len() {
        for x in 0..width as usize {
            if dots[mcy][x] == 0 {
                dots[mcy][x] = braille_dot(0, mcx) | braille_dot(1, mcx);
                colour[mcy][x] = Some(sf_theme::EVAL_MID);
            }
        }
    }

    dots.into_iter()
        .zip(colour)
        .map(|(row, colours)| {
            Line::from(
                row.into_iter()
                    .zip(colours)
                    .map(|(bits, ink)| {
                        let ch = if bits == 0 {
                            ' '
                        } else {
                            char::from_u32(0x2800 + bits as u32).unwrap_or(' ')
                        };
                        Span::styled(
                            ch.to_string(),
                            Style::new()
                                .fg(ink.unwrap_or(sf_theme::FAINT).color())
                                .bg(sf_theme::BG.color()),
                        )
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .collect()
}

/// Braille dot bit for a (column, row) within a cell. The encoding is historical
/// and not row-major, which is why this is a table rather than arithmetic.
fn braille_dot(col: usize, row: usize) -> u8 {
    match (col, row) {
        (0, 0) => 0x01,
        (0, 1) => 0x02,
        (0, 2) => 0x04,
        (0, 3) => 0x40,
        (1, 0) => 0x08,
        (1, 1) => 0x10,
        (1, 2) => 0x20,
        (_, _) => 0x80,
    }
}

/// Stretch a series to `n` samples through a Catmull-Rom spline.
///
/// With twenty moves across two hundred columns, linear interpolation is visibly
/// straight segments meeting at corners, and those corners read as evaluation
/// features that are not there. Catmull-Rom **passes through every real point** and
/// invents shape only between them, which is the honest kind of smoothing.
fn resample(points: &[f64], n: usize) -> Vec<f64> {
    if points.len() < 2 || n < 2 {
        return points.to_vec();
    }
    let at = |i: isize| -> f64 {
        points[(i.max(0) as usize).min(points.len() - 1)]
    };
    let segments = points.len() - 1;
    (0..n)
        .map(|s| {
            let t = s as f64 / (n - 1) as f64 * segments as f64;
            let i = (t.floor() as isize).min(segments as isize - 1);
            let f = t - i as f64;
            let (p0, p1, p2, p3) = (at(i - 1), at(i), at(i + 1), at(i + 2));
            let (f2, f3) = (f * f, f * f * f);
            0.5 * ((2.0 * p1)
                + (-p0 + p2) * f
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * f2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * f3)
        })
        .collect()
}

/// One candidate move in the search's ladder.
pub struct Rung<'a> {
    pub san: &'a str,
    pub visits: u64,
    pub q: f32,
    pub prior: f32,
}

/// The depth ladder.
///
/// Sorted by **visits, not by score**, because visits are the resource the search
/// actually committed and the move it plays is the most-visited one. One long bar
/// means a forced move; five stubs of equal length mean the engine has no idea, and
/// that is worth being able to see at a glance.
pub fn ladder(rows: &[Rung<'_>], width: u16, fg: Ink) -> Vec<Line<'static>> {
    let top = rows.iter().map(|r| r.visits).max().unwrap_or(1).max(1);
    rows.iter()
        .enumerate()
        .map(|(i, r)| {
            let ink = if i == 0 { fg } else { sf_theme::DIM };
            let mut spans = vec![Span::styled(
                lpad(r.san, 7),
                if i == 0 {
                    Style::new()
                        .fg(fg.color())
                        .bg(sf_theme::BG.color())
                        .add_modifier(ratatui::style::Modifier::BOLD)
                } else {
                    Styles::dim()
                },
            )];
            // Ground is the panel background, not `BG_SOFT`: unlike the eval
            // gauge (one full-width bar, where the unfilled portion needs its
            // own visible colour or "95%" reads as a stripe with no context),
            // five-plus of these sit stacked with their own numbers already
            // to the right. A `BG_SOFT` ground behind every one of them
            // painted a solid grey block the full gauge width regardless of
            // how few visits a candidate actually got, which is what read as
            // "messy" -- reported directly, 2026-07-30. Blending the empty
            // part into the background leaves only the proportional streak,
            // which is the number the ladder exists to show.
            spans.extend(bar(r.visits as f64 / top as f64, width, ink, sf_theme::BG).spans);
            spans.push(Span::styled(format!(" {:>6}", r.visits), Styles::dim()));
            spans.push(Span::styled(format!("  q {:.3}", r.q), Styles::dim()));
            spans.push(Span::styled(format!("  p {:.3}", r.prior), Styles::dim()));
            Line::from(spans)
        })
        .collect()
}

#[cfg(test)]
mod chart_tests {
    use super::*;

    #[test]
    fn advantage_is_signed_from_our_point_of_view_and_monotone() {
        assert_eq!(advantage(0.5), "+0.00");
        assert!(advantage(0.9).starts_with('+'));
        assert!(advantage(0.1).starts_with('-'));
        // Monotone, which is what makes it safe beside the gauge.
        let mut prev = f64::NEG_INFINITY;
        for i in 0..=100 {
            let v: f64 = advantage(i as f64 / 100.0).parse().unwrap();
            assert!(v >= prev, "advantage is not monotone at {i}");
            prev = v;
        }
    }

    #[test]
    fn advantage_does_not_blow_up_at_the_extremes() {
        for wp in [0.0, 1.0, -5.0, 12.0] {
            let s = advantage(wp);
            assert!(!s.contains("inf") && !s.contains("NaN"), "advantage({wp}) = {s}");
        }
    }

    /// A missing evaluation draws NOTHING, not a dark column claiming zero.
    #[test]
    fn an_absent_value_draws_nothing_at_all() {
        let rows = evalbar(None, 6, 3);
        assert_eq!(rows.len(), 6);
        for r in &rows {
            assert!(r.spans.iter().all(|s| s.style.bg.is_none()), "an unknown eval must be blank");
        }
    }

    #[test]
    fn the_eval_gauge_fills_from_the_bottom() {
        let rows = evalbar(Some(0.25), 4, 2);
        assert_eq!(rows.len(), 4);
        let bg = |l: &Line| l.spans[0].style.bg;
        assert_eq!(bg(&rows[3]), Some(sf_theme::EVAL_FILL.color()), "bottom row filled");
        assert_eq!(bg(&rows[0]), Some(sf_theme::EVAL_GROUND.color()), "top row empty");
    }

    #[test]
    fn the_eval_gauge_is_always_exactly_its_size() {
        for h in 1..=20u16 {
            for w in 1..=4u16 {
                for v in [0.0, 0.37, 0.5, 1.0] {
                    let rows = evalbar(Some(v), h, w);
                    assert_eq!(rows.len(), h as usize);
                    for r in &rows {
                        let cells: usize =
                            r.spans.iter().map(|s| s.content.chars().count()).sum();
                        assert_eq!(cells, w as usize);
                    }
                }
            }
        }
    }

    /// The chart's scale is fixed 0..1. A dead-level draw must not look like a
    /// collapse, which is the standard way an evaluation chart lies.
    #[test]
    fn the_chart_never_auto_scales() {
        let flat: Vec<f64> = vec![0.50, 0.505, 0.495, 0.50, 0.502];
        let wild: Vec<f64> = vec![0.05, 0.95, 0.10, 0.90, 0.50];
        let a = curve_chart(&flat, 20, 6, 0.5);
        let b = curve_chart(&wild, 20, 6, 0.5);
        let rows_used = |ls: &[Line]| {
            ls.iter()
                .filter(|l| l.spans.iter().any(|s| s.content.trim() != ""))
                .count()
        };
        assert!(
            rows_used(&a) < rows_used(&b),
            "a flat game must occupy fewer rows than a wild one; auto-scaling would \
             make them identical"
        );
    }

    #[test]
    fn the_chart_is_always_exactly_its_size() {
        for h in 1..=10u16 {
            for w in 1..=40u16 {
                let rows = curve_chart(&[0.1, 0.9, 0.4], w, h, 0.5);
                assert_eq!(rows.len(), h as usize);
            }
        }
    }

    #[test]
    fn a_chart_of_fewer_than_two_points_is_blank_not_a_panic() {
        assert_eq!(curve_chart(&[], 10, 4, 0.5).len(), 4);
        assert_eq!(curve_chart(&[0.5], 10, 4, 0.5).len(), 4);
    }

    /// Catmull-Rom must pass THROUGH the real points; inventing shape at them would
    /// be inventing evaluation features.
    #[test]
    fn resampling_preserves_the_real_points() {
        let pts = vec![0.2, 0.8, 0.3, 0.7];
        let out = resample(&pts, 100);
        assert!((out[0] - 0.2).abs() < 1e-9, "first point moved");
        assert!((out[99] - 0.7).abs() < 1e-9, "last point moved");
        assert_eq!(out.len(), 100);
    }

    #[test]
    fn the_ladder_is_ordered_by_what_the_search_committed() {
        let rows = vec![
            Rung { san: "b4", visits: 13926, q: 0.396, prior: 0.213 },
            Rung { san: "Nf3", visits: 900, q: 0.52, prior: 0.30 },
        ];
        let lines = ladder(&rows, 12, sf_theme::ACCENT);
        assert_eq!(lines.len(), 2);
        let first: String = lines[0].spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(first.contains("13926"), "visits are printed: {first}");
        assert!(first.contains("q 0.396") && first.contains("p 0.213"));
        // The leader is emphasised; the rest are dim.
        assert!(lines[0].spans[0].style.add_modifier.contains(ratatui::style::Modifier::BOLD));
    }

    #[test]
    fn an_empty_ladder_is_empty() {
        assert!(ladder(&[], 10, sf_theme::ACCENT).is_empty());
    }
}

#[cfg(test)]
mod fit_tests {
    use super::*;

    fn spans(parts: &[&str]) -> Vec<Span<'static>> {
        parts.iter().map(|p| Span::raw(p.to_string())).collect()
    }
    fn text(l: &Line) -> String {
        l.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    #[test]
    fn a_run_that_fits_is_untouched() {
        let l = fit(spans(&["ab", "cd"]), 10);
        assert_eq!(text(&l), "abcd");
    }

    /// The bug this exists for: a row of unit dots clipped mid-name reads as though
    /// that unit is the last one in the list.
    #[test]
    fn an_overlong_run_is_marked_where_it_was_cut() {
        let l = fit(spans(&["bot ", "lab ", "train ", "rating "]), 12);
        let t = text(&l);
        assert!(t.ends_with('\u{2026}'), "the cut must be visible: {t:?}");
        assert!(t.chars().count() <= 12, "and it must still fit: {t:?}");
    }

    #[test]
    fn fitting_never_exceeds_the_width() {
        for w in 0..=40u16 {
            let l = fit(spans(&["aaaa", "bbbb", "cccc", "dddd"]), w);
            assert!(
                text(&l).chars().count() <= w.max(1) as usize,
                "width {w} produced {:?}",
                text(&l)
            );
        }
    }
}
