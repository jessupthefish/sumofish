//! `chess.svg.board()`, ported.
//!
//! This is a deliberate transcription of python-chess's own assembler rather than
//! a fresh design, because the output has to be *identical*: the artwork is the
//! same cburnett set lichess draws, and the whole value of drawing a real picture
//! is that it is the board from the website and not an impression of it.
//!
//! Fidelity is not asserted by reading. `tests/fidelity.rs` execs the original,
//! renders both its output and ours through the same rasteriser, and requires the
//! PIXELS to match exactly. That is CLAUDE.md's rule for porting a reference
//! implementation, and it earned its keep three times over: the first draft guessed
//! the coordinate placement, drew 16 glyphs where python-chess draws 32, and
//! flipped file labels as well as their positions. None of the three was visible to
//! inspection.
//!
//! # The geometry, and the one number that matters
//!
//! A 45-unit square, eight of them, plus a 15-unit margin either side when
//! coordinates are drawn: `2*15 + 8*45 = 390`. A square edge therefore lands on an
//! exact pixel only when the raster size is a multiple of `390/15 = 26`. Anywhere
//! else the rasteriser has to blend the two square colours across the boundary and
//! the board grows a hairline grid it does not have -- which then moves around,
//! because which edges get a blended pixel depends on where each one rounds.
//! Measured through both rsvg and resvg: eight stray pixels per scanline at 1152,
//! zero at 1144.

use crate::cburnett;
use shakmaty::{Board, Color, File, Rank, Square};
use std::fmt::Write;

/// The picture's colours. lichess's, deliberately not gruvbox: see `sf_theme::picture`.
#[derive(Clone, Copy, Debug)]
pub struct Colors {
    pub square_light: &'static str,
    pub square_dark: &'static str,
    pub square_light_lastmove: &'static str,
    pub square_dark_lastmove: &'static str,
    pub margin: &'static str,
    pub coord: &'static str,
}

impl Default for Colors {
    fn default() -> Self {
        // Mirrors `scripts/dash/sixel.py::COLOURS` exactly.
        Colors {
            square_light: "#f0d9b5",
            square_dark: "#b58863",
            square_light_lastmove: "#cdd26a",
            square_dark_lastmove: "#aaa23a",
            margin: "#2b2724",
            coord: "#e5e0d5",
        }
    }
}

/// Everything that changes what is drawn. Also the cache key: two requests with
/// equal `Request`s produce byte-identical pixels, which is what lets the encoder
/// skip work and what makes a golden-hash test meaningful.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Request {
    /// Placement-only FEN. Nothing else affects the picture.
    pub board_fen: String,
    /// Draw from Black's side.
    pub flip: bool,
    /// Last move, as two squares, for the highlight.
    pub last: Option<(Square, Square)>,
    /// The square of a king in check.
    pub check: Option<Square>,
    /// Edge length in pixels. Must be a multiple of 26; `snap26` guarantees it.
    pub px: u32,
}

pub const VIEWBOX: u32 = 2 * cburnett::COORD_MARGIN + 8 * cburnett::SQUARE_SIZE; // 390

/// Assemble the SVG.
///
/// Equivalent to `chess.svg.board(board, orientation, lastmove, check, size,
/// coordinates=True, colors=...)` in **geometry and paint**, which is the property
/// that matters and the one `tests/fidelity.rs` asserts: it renders python-chess's
/// own output and this function's output through the same rasteriser and requires
/// the pixels to match.
///
/// Deliberately not byte-identical. python-chess builds an `ElementTree` and lets
/// it serialise, so matching bytes would mean matching attribute insertion order
/// and its float formatting -- a lot of fiddly work to pin a *means* rather than
/// the end. It also emits a `<desc><pre>` containing the ASCII board, which is
/// dead weight for a rasteriser and is omitted here.
///
/// Element order follows the original anyway, because it is free: margin, then all
/// 32 coordinate glyphs, then all 64 squares, then the check mark, then the pieces.
/// Getting that wrong would be invisible today (nothing overlaps except a piece
/// with its own square) and would matter the moment anything translucent is added.
pub fn board_svg(board: &Board, req: &Request, colors: &Colors) -> String {
    let margin = cburnett::COORD_MARGIN;
    let sq = cburnett::SQUARE_SIZE;
    let full = VIEWBOX;
    let orientation = if req.flip { Color::Black } else { Color::White };

    let mut s = String::with_capacity(8 * 1024);
    // `width`/`height` as well as the viewBox, as python-chess does when `size` is
    // passed -- which means usvg reports the requested pixel size as the tree's
    // natural size, and the rasteriser must scale from THAT and not from 390.
    let _ = write!(
        s,
        r#"<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {full} {full}" width="{px}" height="{px}">"#,
        px = req.px
    );

    // <defs>: only the pieces actually on the board, in the order python-chess
    // emits them (its own iteration order over the piece map's distinct symbols).
    s.push_str("<defs>");
    let mut emitted: Vec<char> = Vec::new();
    for square in Square::ALL {
        if let Some(piece) = board.piece_at(square) {
            let c = piece_char(piece);
            if !emitted.contains(&c) {
                emitted.push(c);
                s.push_str(cburnett::piece(c).expect("cburnett covers every piece"));
            }
        }
    }
    if req.check.is_some() {
        s.push_str(cburnett::CHECK_GRADIENT);
    }
    s.push_str("</defs>");

    // The margin is drawn as a stroke on an inset rect, not as a fill. That is why
    // the pixmap must not be assumed opaque: use `take_demultiplied`.
    let _ = write!(
        s,
        r#"<rect x="{x}" y="{x}" width="{w}" height="{w}" fill="none" stroke="{c}" stroke-width="{m}" />"#,
        x = margin as f64 / 2.0,
        w = full - margin,
        c = colors.margin,
        m = margin,
    );

    // Coordinates, on ALL FOUR sides -- 32 glyphs, not 16. An earlier version of
    // this port guessed at one file row and one rank column, which put 87 stray
    // pixels on the seam-count scanline and is what caught the guess.
    //
    // `_coord`'s centring: the glyph is scaled by `margin / 20` = 0.75, and the
    // leftover is halved with a *truncating* int conversion before the division:
    // `int(45 - 0.75*45) // 2` is `int(11.25) // 2` = `11 // 2` = 5. Doing that
    // division in floating point would give 5.625 and shift every label.
    let scale = margin as f64 / cburnett::GLYPH_MARGIN as f64;
    let centre = |extent: u32| -> u32 { (((extent as f64) - scale * extent as f64) as u32) / 2 };
    for i in 0..8u32 {
        // Files: top and bottom. The top one sits at y=1 rather than y=0 to keep
        // the ascender off the border, which is python-chess's own comment.
        // The LABEL is always 'a'+i; only its POSITION flips. Flipping both is a
        // double flip that put 'h' where 'a' belongs on a board seen from Black's
        // side -- 1128 differing pixels, found by the diff and invisible to
        // inspection.
        let file_char = (b'a' + i as u8) as char;
        let fx = if orientation == Color::White { i } else { 7 - i };
        let x = fx * sq + margin + centre(sq);
        emit_coord(&mut s, file_char, x, 1, scale, colors.coord);
        emit_coord(&mut s, file_char, x, full - margin, scale, colors.coord);
    }
    for i in 0..8u32 {
        // Ranks: left and right.
        let rank_char = (b'1' + i as u8) as char;
        let ry = if orientation == Color::White { 7 - i } else { i };
        let y = ry * sq + margin + centre(sq);
        emit_coord(&mut s, rank_char, 0, y, scale, colors.coord);
        emit_coord(&mut s, rank_char, full - margin, y, scale, colors.coord);
    }

    // All 64 squares first.
    for square in Square::ALL {
        let (x, y) = square_xy(square, orientation, margin, sq);
        let dark = square.is_dark();
        let is_last = req.last.is_some_and(|(from, to)| square == from || square == to);
        // The colour key is built as `square light lastmove` / `square dark
        // lastmove`, which is why `Colors` names them that way round.
        let fill = match (dark, is_last) {
            (false, false) => colors.square_light,
            (true, false) => colors.square_dark,
            (false, true) => colors.square_light_lastmove,
            (true, true) => colors.square_dark_lastmove,
        };
        let cls = format!(
            "square {} {}{}",
            if dark { "dark" } else { "light" },
            if is_last { "lastmove " } else { "" },
            square
        );
        let _ = write!(
            s,
            r#"<rect x="{x}" y="{y}" width="{sq}" height="{sq}" class="{cls}" stroke="none" fill="{fill}" />"#
        );
    }

    // Then the check mark, once. Not per-square: it is one square's worth of
    // gradient and emitting it inside the square loop put it under every later
    // square's rect in an earlier draft.
    if let Some(check) = req.check {
        let (x, y) = square_xy(check, orientation, margin, sq);
        let _ = write!(
            s,
            r#"<rect x="{x}" y="{y}" width="{sq}" height="{sq}" class="check" fill="url(#check_gradient)" />"#
        );
    }

    // Then the pieces, on top of everything.
    for square in Square::ALL {
        if let Some(piece) = board.piece_at(square) {
            let (x, y) = square_xy(square, orientation, margin, sq);
            let _ = write!(
                s,
                // r##: the `"#` in the href would close a single-hash raw string.
                r##"<use xlink:href="#{name}" transform="translate({x}, {y})" />"##,
                name = piece_id(piece)
            );
        }
    }

    s.push_str("</svg>");
    s
}

/// `fill` *and* `stroke` are both set to the coordinate colour, as python-chess
/// does. The glyph paths carry no paint of their own, so setting only `fill` leaves
/// the outline black and the labels look bolder than they should.
fn emit_coord(s: &mut String, c: char, x: u32, y: u32, scale: f64, colour: &str) {
    let Some(path) = cburnett::coord(c) else { return };
    let _ = write!(
        s,
        r#"<g transform="translate({x}, {y}) scale({scale}, {scale})" fill="{colour}" stroke="{colour}">{path}</g>"#
    );
}

fn square_xy(square: Square, orientation: Color, margin: u32, sq: u32) -> (u32, u32) {
    let file = square.file() as u32;
    let rank = square.rank() as u32;
    let (fx, ry) = if orientation == Color::White {
        (file, 7 - rank)
    } else {
        (7 - file, rank)
    };
    (margin + fx * sq, margin + ry * sq)
}

fn piece_char(p: shakmaty::Piece) -> char {
    let c = match p.role {
        shakmaty::Role::Pawn => 'p',
        shakmaty::Role::Knight => 'n',
        shakmaty::Role::Bishop => 'b',
        shakmaty::Role::Rook => 'r',
        shakmaty::Role::Queen => 'q',
        shakmaty::Role::King => 'k',
    };
    if p.color == Color::White { c.to_ascii_uppercase() } else { c }
}

fn piece_id(p: shakmaty::Piece) -> String {
    let colour = if p.color == Color::White { "white" } else { "black" };
    let role = match p.role {
        shakmaty::Role::Pawn => "pawn",
        shakmaty::Role::Knight => "knight",
        shakmaty::Role::Bishop => "bishop",
        shakmaty::Role::Rook => "rook",
        shakmaty::Role::Queen => "queen",
        shakmaty::Role::King => "king",
    };
    format!("{colour}-{role}")
}

/// Parse a square name like "e4". Used at the seam where telemetry hands us a
/// string.
pub fn square_from_name(name: &str) -> Option<Square> {
    let b = name.as_bytes();
    if b.len() != 2 {
        return None;
    }
    let file = File::from_char(b[0] as char)?;
    let rank = Rank::from_char(b[1] as char)?;
    Some(Square::from_coords(file, rank))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_viewbox_is_390_and_divides_by_26() {
        assert_eq!(VIEWBOX, 390);
        assert_eq!(VIEWBOX % 26, 0);
        assert_eq!(390 / 15, 26, "the grid pitch is the viewbox over the margin");
    }

    #[test]
    fn cburnett_covers_every_piece_and_coordinate() {
        for c in cburnett::PIECE_SYMBOLS.chars() {
            assert!(cburnett::piece(c).is_some(), "no artwork for {c}");
        }
        for c in cburnett::COORD_CHARS.chars() {
            assert!(cburnett::coord(c).is_some(), "no glyph for {c}");
        }
        assert_eq!(cburnett::PIECE_SYMBOLS.chars().count(), 12);
        assert_eq!(cburnett::COORD_CHARS.chars().count(), 16);
    }

    #[test]
    fn squares_map_to_the_same_places_as_python_chess() {
        let m = cburnett::COORD_MARGIN;
        let sq = cburnett::SQUARE_SIZE;
        // a8 is top-left from White's side.
        assert_eq!(square_xy(Square::A8, Color::White, m, sq), (15, 15));
        // h1 is bottom-right.
        assert_eq!(square_xy(Square::H1, Color::White, m, sq), (15 + 7 * 45, 15 + 7 * 45));
        // Flipped, h1 is top-left.
        assert_eq!(square_xy(Square::H1, Color::Black, m, sq), (15, 15));
    }

    /// Labels do not change with orientation; their POSITIONS do. An earlier
    /// version of this file had helpers that flipped the character too, and the
    /// test asserted that wrong belief right alongside them -- which is why the
    /// authoritative check is the pixel diff against python-chess and not this.
    #[test]
    fn only_the_position_of_a_label_depends_on_orientation() {
        let m = cburnett::COORD_MARGIN;
        let sq = cburnett::SQUARE_SIZE;
        // The a-file is on the left from White's side and the right from Black's.
        assert_eq!(square_xy(Square::A1, Color::White, m, sq).0, m);
        assert_eq!(square_xy(Square::A1, Color::Black, m, sq).0, m + 7 * sq);
    }

    #[test]
    fn square_names_round_trip() {
        for s in Square::ALL {
            let name = s.to_string();
            assert_eq!(square_from_name(&name), Some(s), "{name}");
        }
        assert_eq!(square_from_name("z9"), None);
        assert_eq!(square_from_name("e"), None);
        assert_eq!(square_from_name(""), None);
    }
}
