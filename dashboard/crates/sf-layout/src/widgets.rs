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
