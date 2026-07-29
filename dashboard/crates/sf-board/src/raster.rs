//! Position to pixels, in pure Rust.
//!
//! Replaces two subprocesses. The Python shelled out to `rsvg-convert` (54ms) and
//! then ImageMagick (100ms, or 286ms with dithering left on) for every board
//! change, which was fine at one move per nine seconds and quietly catastrophic
//! for bullet: the bot moved faster than the encode, so every move queued behind
//! the last and the board fell further behind for the whole game.
//!
//! Verified in the M0 probe against `rsvg-convert` on the same SVG string: the
//! differences are entirely antialiasing gamma on piece outlines, one pixel wide,
//! invisible at 8x amplification. And resvg reproduces the multiple-of-26 seam
//! rule exactly, which is the property `snap26` exists to protect.

use crate::svg::{Colors, Request, VIEWBOX, board_svg};
use anyhow::{Context, Result};
use resvg::{tiny_skia, usvg};
use shakmaty::Board;

/// Straight (not premultiplied) RGBA pixels, square.
#[derive(Clone, PartialEq, Eq)]
pub struct Raster {
    pub px: u32,
    pub rgba: Vec<u8>,
}

impl std::fmt::Debug for Raster {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Raster").field("px", &self.px).field("bytes", &self.rgba.len()).finish()
    }
}

impl Raster {
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * self.px + x) * 4) as usize;
        [self.rgba[i], self.rgba[i + 1], self.rgba[i + 2], self.rgba[i + 3]]
    }
    /// RGB triples, for the kitty `f=24` path. F2 established that Konsole accepts
    /// raw pixel formats, so the PNG encode is optional and we skip it: no
    /// deflate on the render path at all.
    pub fn rgb(&self) -> Vec<u8> {
        self.rgba.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect()
    }
}

/// A parsed SVG, kept so a re-render at a new size does not re-parse. The XML
/// parse is the expensive half of the pipeline and it does not depend on the
/// output size.
pub struct Renderer {
    opts: usvg::Options<'static>,
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new()
    }
}

impl Renderer {
    pub fn new() -> Self {
        // No font database: python-chess draws coordinate labels as <path>
        // outlines, so `resvg` is built with `default-features = false` and there
        // is no fontconfig anywhere in the binary.
        Renderer { opts: usvg::Options::default() }
    }

    /// Render a position. `req.px` must already be snapped to the 26-pixel grid;
    /// the layout solver guarantees it and this asserts it in debug.
    pub fn render(&self, board: &Board, req: &Request, colors: &Colors) -> Result<Raster> {
        debug_assert_eq!(req.px % 26, 0, "unsnapped size {} will grow a hairline grid", req.px);
        let svg = board_svg(board, req, colors);
        self.render_svg(&svg, req.px)
    }

    pub fn render_svg(&self, svg: &str, px: u32) -> Result<Raster> {
        let tree = usvg::Tree::from_str(svg, &self.opts).context("parse board svg")?;
        let mut pixmap =
            tiny_skia::Pixmap::new(px, px).context("allocate pixmap for the board")?;
        // Scale from the tree's OWN size, not from `VIEWBOX`. `board_svg` writes
        // `width`/`height` as well as the viewBox, exactly as python-chess does, so
        // usvg already reports the requested pixel size and the correct scale is
        // 1.0. Dividing by 390 instead rendered a 2.93x-zoomed crop of the
        // top-left corner -- which looked like a plausible board until the seam
        // count found 87 blended pixels on a scanline that should have had none.
        let natural = tree.size().width();
        let scale = if natural > 0.0 { px as f32 / natural } else { 1.0 };
        resvg::render(
            &tree,
            tiny_skia::Transform::from_scale(scale, scale),
            &mut pixmap.as_mut(),
        );
        // Demultiplied, because the margin is drawn as a stroke rather than a fill
        // so the pixmap is not guaranteed opaque, and a quantiser fed
        // premultiplied edge pixels produces garbage in the border.
        Ok(Raster { px, rgba: pixmap.take_demultiplied() })
    }
}

/// One terminal cell of the half-block fallback: two vertically stacked pixels
/// drawn as `▀`, foreground on top.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct HalfCell {
    pub top: [u8; 3],
    pub bottom: [u8; 3],
}

/// Downsample a raster into half-block cells.
///
/// Terminal cells are roughly 1:2, so one cell showing two stacked pixels is
/// square. Block elements are used rather than chess glyphs deliberately: `♞` and
/// the Nerd Font codepoints have ambiguous `wcwidth`, and that ambiguity drifts
/// table borders row by row.
pub fn halfblocks(r: &Raster, cols: u16, rows: u16) -> Vec<Vec<HalfCell>> {
    let (cols, rows) = (cols.max(1) as u32, rows.max(1) as u32);
    let (px_w, px_h) = (cols, rows * 2);
    let mut out = Vec::with_capacity(rows as usize);
    for row in 0..rows {
        let mut line = Vec::with_capacity(cols as usize);
        for col in 0..cols {
            let top = sample(r, col, row * 2, px_w, px_h);
            let bottom = sample(r, col, row * 2 + 1, px_w, px_h);
            line.push(HalfCell { top, bottom });
        }
        out.push(line);
    }
    out
}

/// Box-average the source region that maps to one output pixel. Averaging rather
/// than nearest-neighbour because a chess board downsampled by point sampling
/// loses whole piece outlines.
fn sample(r: &Raster, x: u32, y: u32, out_w: u32, out_h: u32) -> [u8; 3] {
    let x0 = x * r.px / out_w;
    let x1 = (((x + 1) * r.px).div_ceil(out_w)).min(r.px).max(x0 + 1);
    let y0 = y * r.px / out_h;
    let y1 = (((y + 1) * r.px).div_ceil(out_h)).min(r.px).max(y0 + 1);
    let mut acc = [0u32; 3];
    let mut n = 0u32;
    for sy in y0..y1 {
        for sx in x0..x1 {
            let p = r.pixel(sx.min(r.px - 1), sy.min(r.px - 1));
            acc[0] += p[0] as u32;
            acc[1] += p[1] as u32;
            acc[2] += p[2] as u32;
            n += 1;
        }
    }
    if n == 0 {
        return [0, 0, 0];
    }
    [(acc[0] / n) as u8, (acc[1] / n) as u8, (acc[2] / n) as u8]
}

/// Count pixels on the board's central scanline that are neither square colour.
/// A blended edge pixel is a seam, and this is the measurement that justifies
/// `snap26`. Exposed rather than test-local so the M2 gate and the xtask probe can
/// share one definition.
pub fn seam_count(r: &Raster, light: [u8; 3], dark: [u8; 3]) -> usize {
    let margin = (15.0 / VIEWBOX as f64 * r.px as f64).ceil() as u32;
    if r.px <= 2 * margin {
        return 0;
    }
    let y = r.px / 2;
    (margin..r.px - margin)
        .filter(|x| {
            let p = r.pixel(*x, y);
            [p[0], p[1], p[2]] != light && [p[0], p[1], p[2]] != dark
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::svg::Request;
    use shakmaty::Board;

    /// Parse the placement field only. Deliberately NOT via `Chess`: a position
    /// is validated (an empty board is `EMPTY_BOARD | MISSING_KING`) and the
    /// picture happily draws boards that are not legal positions.
    fn board_of(fen: &str) -> Board {
        fen.split(' ').next().unwrap().parse::<Board>().unwrap()
    }

    fn req(px: u32) -> Request {
        Request { board_fen: String::new(), flip: false, last: None, check: None, px }
    }

    const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    #[test]
    fn renders_at_the_requested_size() {
        let r = Renderer::new();
        let out = r.render(&board_of(START), &req(390), &Colors::default()).unwrap();
        assert_eq!(out.px, 390);
        assert_eq!(out.rgba.len(), (390 * 390 * 4) as usize);
    }

    /// The whole reason `snap26` exists, as an assertion rather than a comment.
    /// Multiples of 26 have no blended edge pixels; other sizes do.
    #[test]
    fn the_seam_rule_holds_in_resvg() {
        let r = Renderer::new();
        let empty = board_of("8/8/8/8/8/8/8/8 w - - 0 1");
        let light = [0xf0, 0xd9, 0xb5];
        let dark = [0xb5, 0x88, 0x63];
        for px in [390u32, 1144, 1170] {
            let out = r.render_svg(&crate::svg::board_svg(&empty, &req(px), &Colors::default()), px).unwrap();
            assert_eq!(
                seam_count(&out, light, dark),
                0,
                "{px} is a multiple of 26 and must have no seams"
            );
        }
        for px in [1152u32, 1180] {
            let out = r.render_svg(&crate::svg::board_svg(&empty, &req(px), &Colors::default()), px).unwrap();
            assert!(
                seam_count(&out, light, dark) > 0,
                "{px} is off-grid and should show the hairline this rule protects against"
            );
        }
    }

    /// Same request, same pixels. This is what makes an image cache key sound and
    /// what stops the picture being re-encoded and re-emitted for no reason --
    /// re-emitting an unchanged image makes the terminal clear the region first, so
    /// the board strobes at the frame rate.
    #[test]
    fn rendering_is_deterministic() {
        let r = Renderer::new();
        let b = board_of(START);
        let first = r.render(&b, &req(390), &Colors::default()).unwrap();
        for _ in 0..3 {
            assert_eq!(r.render(&b, &req(390), &Colors::default()).unwrap(), first);
        }
    }

    #[test]
    fn the_board_is_opaque_despite_the_margin_being_a_stroke() {
        let r = Renderer::new();
        let out = r.render(&board_of(START), &req(390), &Colors::default()).unwrap();
        // Centre and all four corners.
        for (x, y) in [(195, 195), (0, 0), (389, 0), (0, 389), (389, 389)] {
            assert_eq!(out.pixel(x, y)[3], 255, "({x},{y}) is not opaque");
        }
    }

    #[test]
    fn flipping_mirrors_the_picture() {
        let r = Renderer::new();
        let b = board_of(START);
        let normal = r.render(&b, &req(390), &Colors::default()).unwrap();
        let flipped =
            r.render(&b, &Request { flip: true, ..req(390) }, &Colors::default()).unwrap();
        assert_ne!(normal, flipped, "flipping must change the picture");
        // Sample through the middle of the back rank, where the PIECES are. The
        // empty squares are symmetric under a flip -- a8 and h1 are both dark --
        // so a strip that catches only square colours proves nothing, which is
        // exactly what an earlier version of this test did.
        let y = 15 + 22; // centre of the top rank
        let strip = |img: &Raster| -> Vec<[u8; 4]> {
            (20..img.px - 20).step_by(5).map(|x| img.pixel(x, y)).collect()
        };
        assert_ne!(
            strip(&normal),
            strip(&flipped),
            "the top rank holds black pieces one way up and white the other"
        );
    }

    #[test]
    fn the_last_move_highlight_changes_only_two_squares() {
        let r = Renderer::new();
        let b = board_of(START);
        let plain = r.render(&b, &req(390), &Colors::default()).unwrap();
        let lit = r
            .render(
                &b,
                &Request {
                    last: Some((shakmaty::Square::E2, shakmaty::Square::E4)),
                    ..req(390)
                },
                &Colors::default(),
            )
            .unwrap();
        // e4 is empty in the start position, so its centre should have changed to
        // the highlight colour, while a distant empty square should not have.
        let e4 = (15 + 4 * 45 + 22, 15 + 4 * 45 + 22);
        let a5 = (15 + 22, 15 + 3 * 45 + 22);
        assert_ne!(plain.pixel(e4.0, e4.1), lit.pixel(e4.0, e4.1), "e4 must be highlighted");
        assert_eq!(plain.pixel(a5.0, a5.1), lit.pixel(a5.0, a5.1), "a5 must be untouched");
    }

    #[test]
    fn halfblocks_produce_exactly_the_requested_grid() {
        let r = Renderer::new();
        let out = r.render(&board_of(START), &req(390), &Colors::default()).unwrap();
        for (cols, rows) in [(8u16, 4u16), (16, 8), (32, 16), (1, 1), (64, 32)] {
            let cells = halfblocks(&out, cols, rows);
            assert_eq!(cells.len(), rows as usize);
            assert!(cells.iter().all(|r| r.len() == cols as usize));
        }
    }

    #[test]
    fn halfblocks_of_a_board_are_not_all_one_colour() {
        let r = Renderer::new();
        let out = r.render(&board_of(START), &req(390), &Colors::default()).unwrap();
        let cells = halfblocks(&out, 16, 8);
        let flat: Vec<HalfCell> = cells.into_iter().flatten().collect();
        let distinct = flat.iter().collect::<std::collections::HashSet<_>>().len();
        assert!(distinct > 8, "a downsampled board should show structure, got {distinct} distinct cells");
    }
}
