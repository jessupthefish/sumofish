//! Gruvbox, semantically named, plus the six colours the *picture* uses.
//!
//! Two rules carried over from `scripts/dash/theme.py`, both load-bearing:
//!
//! **Board squares and highlights come in light/dark pairs.** Tinting a
//! highlighted square with one flat colour destroys the checker pattern exactly
//! where the eye is being asked to look. Highlighting keeps the pattern and
//! shifts the hue.
//!
//! **The picture's colours are lichess's, not gruvbox's.** They live here rather
//! than in the board renderer so that the palette-conformance test can *exempt*
//! them by name. The point of drawing a real image is that it is the board from
//! the website; the first person to enforce "keep to the theme" without this
//! distinction would gruvbox the board and destroy that.
//!
//! Names are semantic (`eval_fill`, `board_light`) rather than gruvbox
//! (`yellow`, `aqua`), so a light theme is a drop-in. The Python had 27 names
//! over 24 distinct hexes; the aliases are gone and the duplication with them.

use ratatui::style::{Color, Modifier, Style};

/// One RGB colour, with the hex kept so the palette test can name it in a
/// failure message.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Ink {
    pub name: &'static str,
    pub rgb: (u8, u8, u8),
}

impl Ink {
    const fn new(name: &'static str, hex: u32) -> Self {
        Ink {
            name,
            rgb: (
                ((hex >> 16) & 0xff) as u8,
                ((hex >> 8) & 0xff) as u8,
                (hex & 0xff) as u8,
            ),
        }
    }
    pub const fn color(self) -> Color {
        Color::Rgb(self.rgb.0, self.rgb.1, self.rgb.2)
    }
    /// Relative luminance, WCAG 2.x. Used by the contrast test that encodes the
    /// "a gauge whose empty half matched the panel at 1.12:1" Lab Note.
    pub fn luminance(self) -> f64 {
        fn ch(v: u8) -> f64 {
            let s = v as f64 / 255.0;
            if s <= 0.03928 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
        }
        0.2126 * ch(self.rgb.0) + 0.7152 * ch(self.rgb.1) + 0.0722 * ch(self.rgb.2)
    }
}

/// WCAG contrast ratio between two inks, 1.0..=21.0.
pub fn contrast(a: Ink, b: Ink) -> f64 {
    let (hi, lo) = {
        let (x, y) = (a.luminance(), b.luminance());
        if x > y { (x, y) } else { (y, x) }
    };
    (hi + 0.05) / (lo + 0.05)
}

macro_rules! inks {
    ($( $konst:ident = $name:literal, $hex:literal; )*) => {
        $( pub const $konst: Ink = Ink::new($name, $hex); )*
        /// Every ink in the theme. The palette-conformance test set-differences
        /// the colours actually rendered against this.
        pub const ALL: &[Ink] = &[ $($konst),* ];
    };
}

inks! {
    // --- ground and text ---
    // BG must match the real terminal background or every explicitly-styled
    // cell (nearly all of them -- every `Styles::*()` function sets `.bg()`)
    // reads as a lighter patch against whatever's NOT explicitly styled
    // (inter-panel gaps, the outer margin), which is the terminal's own
    // default. Checked against the actual profile in use
    // (`~/.local/share/konsole/GruvboxHard.colorscheme`, `[Background]
    // Color=29,32,33`): that is `0x1d2021`, not the `0x282828` this used to
    // hold -- reported directly as "all the text has a softer grey
    // background", 2026-07-30. Swapped with `BG_HARD`, which held the
    // correct value already but was never actually applied anywhere.
    BG        = "bg",         0x1d2021;
    BG_SOFT   = "bg-soft",    0x32302f;
    BG_HARD   = "bg-hard",    0x282828;
    FG        = "fg",         0xebdbb2;
    DIM       = "dim",        0x928374;
    FAINT     = "faint",      0x665c54;

    // --- meaning ---
    ACCENT    = "accent",     0xfabd2f;   // ours, the thing to look at
    GOOD      = "good",       0xb8bb26;
    BAD       = "bad",        0xfb4934;
    INFO      = "info",       0x83a598;
    COOL      = "cool",       0x8ec07c;
    WARM      = "warm",       0xfe8019;
    PLUM      = "plum",       0xd3869b;

    // --- the evaluation gauge ---
    // White for our share whichever colour SumoFish is pushing, because white is
    // the ink every evaluation bar uses for the side that is winning. What it is
    // NOT is white-against-black: this bar measures OUR chance, not White's, and
    // an opposing black half would carry chess meaning contradicting the number.
    EVAL_FILL = "eval-fill",  0xffffff;
    // The empty half is a ground, a colour that means only "not filled". It was
    // #32302f, which is 1.12:1 against the #282828 panel -- indistinguishable --
    // so a gauge reading 95% looked like a stripe floating in space rather than a
    // bar filled nearly to the top. #504945 is 1.67:1 against the panel and
    // 8.9:1 against the fill, so both extent and level are legible. There is a
    // test.
    EVAL_GROUND = "eval-ground", 0x504945;
    EVAL_MID  = "eval-mid",   0x7c6f64;   // the chart's 0.50 line

    // --- the board, when drawn as text (the no-graphics fallback) ---
    // Warm neutral pair, far enough apart to read at a glance and neither close
    // in luminance to a piece fill.
    BOARD_LIGHT  = "board-light",  0xbdae93;
    BOARD_DARK   = "board-dark",   0x7c6f64;
    LAST_LIGHT   = "last-light",   0xa9a03f;   // last move: the pair shifted green
    LAST_DARK    = "last-dark",    0x79762f;
    CHECK_LIGHT  = "check-light",  0xcc5b47;   // king in check: shifted red
    CHECK_DARK   = "check-dark",   0x9d4436;
    PIECE_W      = "piece-w",      0xf9f5d7;
    PIECE_W_EDGE = "piece-w-edge", 0x20211f;
    PIECE_B      = "piece-b",      0x16181a;
    PIECE_B_EDGE = "piece-b-edge", 0xebdbb2;
}

/// The picture's own palette: lichess's board, deliberately not gruvbox.
/// Exempt from the palette-conformance test -- see the module docs.
pub mod picture {
    use super::Ink;
    pub const SQ_LIGHT: Ink = Ink::new("pic-square-light", 0xf0d9b5);
    pub const SQ_DARK: Ink = Ink::new("pic-square-dark", 0xb58863);
    pub const SQ_LIGHT_LAST: Ink = Ink::new("pic-square-light-last", 0xcdd26a);
    pub const SQ_DARK_LAST: Ink = Ink::new("pic-square-dark-last", 0xaaa23a);
    pub const MARGIN: Ink = Ink::new("pic-margin", 0x2b2724);
    pub const COORD: Ink = Ink::new("pic-coord", 0xe5e0d5);
    /// The check ring, drawn on the square's OUTER pixels over everything: the
    /// king covers almost all of its own square, so a background tint hides the
    /// warning behind the piece it is about.
    pub const CHECK_RING: Ink = Ink::new("pic-check-ring", 0xfb4934);

    pub const ALL: &[Ink] = &[
        SQ_LIGHT, SQ_DARK, SQ_LIGHT_LAST, SQ_DARK_LAST, MARGIN, COORD, CHECK_RING,
    ];
}

/// Freshness of a value, in the sense a plot display means it. Every panel that
/// draws a number draws this beside it. The failure it exists to prevent is the
/// old `profile = api(...) or profile`, where a dead connection rendered
/// pixel-identical to a live one.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Track {
    /// Being fed on schedule.
    #[default]
    Live,
    /// Coasting on its last value.
    Coast,
    /// Gone.
    Lost,
}

impl Track {
    pub const fn ink(self) -> Ink {
        match self {
            Track::Live => GOOD,
            Track::Coast => ACCENT,
            Track::Lost => BAD,
        }
    }
    pub const fn glyph(self) -> &'static str {
        match self {
            Track::Live => "\u{25cf}",  // ●
            Track::Coast => "\u{25d1}", // ◑
            Track::Lost => "\u{25cb}",  // ○
        }
    }
}

/// Styles, so panels never construct a `Style` from a raw colour and the
/// palette test has a small surface to check.
pub struct Styles;

impl Styles {
    pub fn body() -> Style {
        Style::new().fg(FG.color()).bg(BG.color())
    }
    pub fn dim() -> Style {
        Style::new().fg(DIM.color()).bg(BG.color())
    }
    pub fn faint() -> Style {
        Style::new().fg(FAINT.color()).bg(BG.color())
    }
    pub fn title() -> Style {
        Style::new().fg(FG.color()).bg(BG.color()).add_modifier(Modifier::BOLD)
    }
    pub fn border() -> Style {
        Style::new().fg(FAINT.color()).bg(BG.color())
    }
    pub fn accent() -> Style {
        Style::new().fg(ACCENT.color()).bg(BG.color())
    }
    pub fn ink(i: Ink) -> Style {
        Style::new().fg(i.color()).bg(BG.color())
    }
    pub fn track(t: Track) -> Style {
        Style::new().fg(t.ink().color()).bg(BG.color())
    }
    /// The badge in the header. Was ground-and-figure swapped (dark text on a
    /// solid accent chip); changed 2026-07-30 to coloured text on the same
    /// background every other panel uses, per direct report that the
    /// swapped chips read as a mismatched grey/coloured block sitting on top
    /// of the panel rather than belonging to it.
    pub fn badge() -> Style {
        Style::new().fg(ACCENT.color()).bg(BG.color()).add_modifier(Modifier::BOLD)
    }
}

/// Eighth-width blocks, for horizontal gauges.
pub const BLOCKS: [char; 9] = [' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█'];
/// Eighth-height blocks, for sparklines.
pub const SPARK: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// Gauges get a cap. On a 2560px-wide terminal an "as wide as the panel" bar is
/// 140 characters, which puts its own label so far from its fill that the eye
/// cannot associate the two. A bar is a comparison, not a decoration.
pub const GAUGE_MAX: u16 = 44;

#[cfg(test)]
mod tests {
    use super::*;

    /// The Lab Note, as a test: a gauge's ground must be distinguishable from
    /// the panel behind it, or a full gauge and an absent one look the same.
    #[test]
    fn eval_ground_is_visible_against_the_panel() {
        let c = contrast(EVAL_GROUND, BG);
        assert!(c > 1.5, "eval ground vs panel is {c:.2}:1, need >1.5 (the old #32302f was 1.12)");
        let f = contrast(EVAL_FILL, EVAL_GROUND);
        assert!(f > 4.5, "eval fill vs ground is {f:.2}:1, need >4.5");
    }

    #[test]
    fn body_text_is_readable() {
        assert!(contrast(FG, BG) > 7.0, "body text must clear AAA");
        // DIM is deliberately low-contrast, but not invisible.
        assert!(contrast(DIM, BG) > 3.0, "dim text must still clear AA-large");
    }

    /// Board square pairs must be far enough apart to read the checker at a
    /// glance, and neither may sit close to a piece fill.
    #[test]
    fn board_pairs_are_distinguishable() {
        for (l, d) in [
            (BOARD_LIGHT, BOARD_DARK),
            (LAST_LIGHT, LAST_DARK),
            (CHECK_LIGHT, CHECK_DARK),
            (picture::SQ_LIGHT, picture::SQ_DARK),
            (picture::SQ_LIGHT_LAST, picture::SQ_DARK_LAST),
        ] {
            let c = contrast(l, d);
            assert!(c > 1.3, "{} vs {} is only {c:.2}:1", l.name, d.name);
        }
    }

    /// Two inks in the *same* visual context must never share a hex, or one of
    /// them is invisible where it matters. Across contexts, sharing is fine and
    /// deliberate: `eval-mid` and `board-dark` are both #7c6f64 and never appear
    /// in the same place, and keeping them separately named is what lets either
    /// move without dragging the other.
    ///
    /// (The Python's problem was not shared hexes, it was 27 names for 24 hexes
    /// with no way to tell an alias from a coincidence -- `LIVE` *was* `GOOD`,
    /// so changing one changed the other silently. Here `Track::ink()` maps to
    /// the meaning colours explicitly instead.)
    #[test]
    fn inks_within_a_context_are_distinct() {
        let contexts: &[(&str, &[Ink])] = &[
            ("chrome", &[BG, BG_SOFT, BG_HARD, FG, DIM, FAINT]),
            ("meaning", &[ACCENT, GOOD, BAD, INFO, COOL, WARM, PLUM]),
            ("gauge", &[EVAL_FILL, EVAL_GROUND, EVAL_MID]),
            (
                "text-board",
                &[
                    BOARD_LIGHT, BOARD_DARK, LAST_LIGHT, LAST_DARK, CHECK_LIGHT, CHECK_DARK,
                    PIECE_W, PIECE_W_EDGE, PIECE_B, PIECE_B_EDGE,
                ],
            ),
            ("picture", picture::ALL),
        ];
        for (ctx, inks) in contexts {
            for (i, a) in inks.iter().enumerate() {
                for b in &inks[i + 1..] {
                    assert_ne!(
                        a.rgb, b.rgb,
                        "in context {ctx}: {} and {} are the same colour",
                        a.name, b.name
                    );
                }
            }
        }
    }

    #[test]
    fn names_are_unique_across_theme_and_picture() {
        let mut names: Vec<&str> = ALL.iter().chain(picture::ALL).map(|i| i.name).collect();
        names.sort_unstable();
        let n = names.len();
        names.dedup();
        assert_eq!(n, names.len(), "duplicate ink name");
    }
}
