//! Getting the picture onto the screen, and the discipline around it.
//!
//! Five rules, all of them paid for by a bug in the Python version:
//!
//! 1. **Emit only when the key changes.** Re-emitting an unchanged image makes the
//!    terminal clear the region before redrawing, so the board strobes at the frame
//!    rate. (Kitty placements do not strobe, but the rule costs nothing and keeps
//!    the sixel path honest.)
//! 2. **Never emit an image drawn for a different layout.** The layout rebuilds
//!    instantly and a render takes time, so the frame in between would paint the
//!    previous, larger board over a screen just wiped for the new one -- and its
//!    bottom and right edges land in rows the new layout never writes to, where
//!    nothing will ever repaint them. On screen: a strip of squares and half a pawn
//!    surviving forever. The pixel size is part of the key; compare it.
//! 3. **Flush the text layer before writing image bytes.** ratatui's pending frame
//!    is buffered, and if it flushes after the image it paints over it.
//! 4. **Never place an image on the last row.** A sixel image on the bottom line
//!    scrolls the screen.
//! 5. **`ESC[2J` on a geometry change** clears whatever did get stranded. Verified
//!    to erase image data in Konsole. With kitty, `a=d` deletion by id is better
//!    and is used when the probe confirmed it.

use anyhow::Result;
use base64::Engine as _;
use sf_board::Raster;
use std::io::Write;

use crate::caps::{Caps, Graphics};

/// Identifies a rendered picture. Two equal keys mean two identical images, so this
/// is what rule 1 and rule 2 compare.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ImageKey {
    pub board_fen: String,
    pub flip: bool,
    pub last: Option<String>,
    pub check: Option<String>,
    /// Rule 2 lives here: an image drawn at a different size is never emitted into
    /// the current layout.
    pub px: u32,
    /// And nor is one drawn for a different position on screen.
    pub at: (u16, u16),
}

/// The kitty image id we reuse. A single id with a single placement means a new
/// board replaces the old one in place rather than stacking.
const KITTY_IMAGE_ID: u32 = 7317;
/// The protocol's chunk limit, in base64 characters.
const CHUNK: usize = 4096;

/// Write a picture at a 1-based cell position.
///
/// `out` must be the same stream ratatui draws to, already flushed by the caller.
pub fn emit(out: &mut impl Write, caps: &Caps, raster: &Raster, at: (u16, u16)) -> Result<()> {
    match caps.graphics {
        Graphics::Kitty => emit_kitty(out, raster, at),
        Graphics::Sixel => emit_sixel(out, raster, at),
        // Half-blocks are drawn as ordinary cells by the board panel, not emitted.
        Graphics::HalfBlocks | Graphics::None => Ok(()),
    }
}

/// Transmit-and-place in one command, positioned by a cursor move.
///
/// `f=24` raw RGB rather than PNG: F2 established that Konsole accepts raw pixel
/// formats, so there is no deflate on the render path at all. The cost is about 4/3
/// the escape bytes of PNG at this flatness, which at one board change per move is
/// irrelevant.
fn emit_kitty(out: &mut impl Write, raster: &Raster, at: (u16, u16)) -> Result<()> {
    let rgb = raster.rgb();
    let b64 = base64::engine::general_purpose::STANDARD.encode(&rgb);
    let bytes = b64.as_bytes();

    // Position, then transmit. `q=2` suppresses both the OK and the errors, which
    // matters because a reply would otherwise land in the event stream and be read
    // as keystrokes.
    write!(out, "\x1b[{};{}H", at.0, at.1)?;
    let control = format!(
        "i={KITTY_IMAGE_ID},p=1,s={w},v={h},a=T,t=d,f=24,q=2",
        w = raster.px,
        h = raster.px
    );
    let mut first = true;
    let mut i = 0;
    while i < bytes.len() {
        let end = (i + CHUNK).min(bytes.len());
        let more = u8::from(end < bytes.len());
        let slice = std::str::from_utf8(&bytes[i..end])?;
        if first {
            write!(out, "\x1b_G{control},m={more};{slice}\x1b\\")?;
            first = false;
        } else {
            write!(out, "\x1b_Gm={more};{slice}\x1b\\")?;
        }
        i = end;
    }
    // Park the cursor at the origin so a stray write does not land on the picture.
    write!(out, "\x1b[H")?;
    out.flush()?;
    Ok(())
}

/// Sixel, for terminals without kitty graphics.
///
/// `diffusion: 0.0` is not a preference, it is a requirement. Error diffusion over a
/// picture this flat buys nothing, is recomputed from the whole image every render
/// so the noise lands differently after every move (the board visibly shimmers where
/// it should be identical), and it is three times slower: measured at 1144px, 286ms
/// with dithering against 96ms without.
fn emit_sixel(out: &mut impl Write, raster: &Raster, at: (u16, u16)) -> Result<()> {
    let payload = icy_sixel::sixel_encode(
        &raster.rgba,
        raster.px as usize,
        raster.px as usize,
        &icy_sixel::encoder::EncodeOptions {
            max_colors: 64,
            diffusion: 0.0,
            ..Default::default()
        },
    )
    .map_err(|e| anyhow::anyhow!("sixel encode: {e:?}"))?;
    write!(out, "\x1b[{};{}H", at.0, at.1)?;
    out.write_all(payload.as_bytes())?;
    write!(out, "\x1b[H")?;
    out.flush()?;
    Ok(())
}

/// Remove the picture. Rule 5.
pub fn clear(out: &mut impl Write, caps: &Caps) -> Result<()> {
    match caps.graphics {
        Graphics::Kitty if caps.kitty_delete => {
            write!(out, "\x1b_Ga=d,d=i,i={KITTY_IMAGE_ID},q=2\x1b\\")?;
        }
        Graphics::Kitty | Graphics::Sixel => {
            // The eraser of last resort, and the only one the Python had.
            write!(out, "\x1b[2J\x1b[H")?;
        }
        Graphics::HalfBlocks | Graphics::None => {}
    }
    out.flush()?;
    Ok(())
}

/// Tracks what is currently on screen so rules 1 and 2 are enforced in one place
/// rather than at every call site.
#[derive(Default)]
pub struct Painter {
    painted: Option<ImageKey>,
}

impl Painter {
    /// Emit `raster` if and only if it is both new and drawn for the current
    /// layout. Returns whether anything was written.
    pub fn paint(
        &mut self,
        out: &mut impl Write,
        caps: &Caps,
        key: &ImageKey,
        raster: &Raster,
    ) -> Result<bool> {
        // Rule 2: the image must have been drawn at the size the layout wants now.
        if raster.px != key.px {
            tracing::debug!(
                have = raster.px,
                want = key.px,
                "declining to paint a board drawn for another layout"
            );
            return Ok(false);
        }
        // Rule 1.
        if self.painted.as_ref() == Some(key) {
            return Ok(false);
        }
        emit(out, caps, raster, key.at)?;
        self.painted = Some(key.clone());
        Ok(true)
    }

    /// Forget what is on screen: after a resize, a redraw request, or a clear.
    pub fn forget(&mut self) {
        self.painted = None;
    }

    pub fn current(&self) -> Option<&ImageKey> {
        self.painted.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raster(px: u32) -> Raster {
        Raster { px, rgba: vec![0x40; (px * px * 4) as usize] }
    }

    fn key(px: u32) -> ImageKey {
        ImageKey {
            board_fen: "8/8/8/8/8/8/8/8".into(),
            flip: false,
            last: None,
            check: None,
            px,
            at: (4, 3),
        }
    }

    fn kitty_caps() -> Caps {
        Caps { graphics: Graphics::Kitty, kitty_confirmed: true, kitty_delete: true, ..Caps::default() }
    }

    #[test]
    fn an_unchanged_image_is_not_re_emitted() {
        let mut p = Painter::default();
        let mut out = Vec::new();
        assert!(p.paint(&mut out, &kitty_caps(), &key(26), &raster(26)).unwrap());
        let after_first = out.len();
        assert!(!p.paint(&mut out, &kitty_caps(), &key(26), &raster(26)).unwrap());
        assert_eq!(out.len(), after_first, "rule 1: no bytes for an unchanged board");
    }

    /// Rule 2. This is the bug that stranded a strip of squares and half a pawn on
    /// screen forever after a resize.
    #[test]
    fn an_image_drawn_for_another_layout_is_never_painted() {
        let mut p = Painter::default();
        let mut out = Vec::new();
        // The layout now wants 52px; the renderer has only finished the old 26px.
        assert!(!p.paint(&mut out, &kitty_caps(), &key(52), &raster(26)).unwrap());
        assert!(out.is_empty(), "better briefly absent than briefly wrong");
        assert!(p.current().is_none());
    }

    #[test]
    fn a_new_position_is_emitted() {
        let mut p = Painter::default();
        let mut out = Vec::new();
        p.paint(&mut out, &kitty_caps(), &key(26), &raster(26)).unwrap();
        let mut moved = key(26);
        moved.board_fen = "8/8/8/8/8/8/8/K7".into();
        let before = out.len();
        assert!(p.paint(&mut out, &kitty_caps(), &moved, &raster(26)).unwrap());
        assert!(out.len() > before);
    }

    #[test]
    fn moving_the_picture_re_emits_it() {
        let mut p = Painter::default();
        let mut out = Vec::new();
        p.paint(&mut out, &kitty_caps(), &key(26), &raster(26)).unwrap();
        let mut elsewhere = key(26);
        elsewhere.at = (9, 9);
        assert!(p.paint(&mut out, &kitty_caps(), &elsewhere, &raster(26)).unwrap());
    }

    #[test]
    fn forget_makes_the_next_paint_unconditional() {
        let mut p = Painter::default();
        let mut out = Vec::new();
        p.paint(&mut out, &kitty_caps(), &key(26), &raster(26)).unwrap();
        p.forget();
        assert!(p.paint(&mut out, &kitty_caps(), &key(26), &raster(26)).unwrap());
    }

    #[test]
    fn the_kitty_stream_is_well_formed() {
        let mut out = Vec::new();
        // 40x40 RGB is 4800 bytes -> 6400 base64 -> two chunks.
        emit_kitty(&mut out, &raster(40), (4, 3)).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.starts_with("\x1b[4;3H"), "positioned by a cursor move");
        assert_eq!(s.matches("\x1b_G").count(), 2, "two chunks");
        assert!(s.contains(",m=1;"), "the first chunk says more follows");
        assert!(s.contains("\x1b_Gm=0;"), "the last chunk terminates");
        assert!(s.contains("a=T"), "transmit and place in one command");
        assert!(s.contains("f=24"), "raw RGB: no PNG encode on the render path");
        assert!(s.contains("q=2"), "replies suppressed, or they land in the event stream");
        assert!(s.ends_with("\x1b[H"), "cursor parked away from the picture");
    }

    #[test]
    fn a_single_chunk_image_still_terminates() {
        let mut out = Vec::new();
        emit_kitty(&mut out, &raster(26), (1, 1)).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert_eq!(s.matches("\x1b_G").count(), 1);
        assert!(s.contains(",m=0;"), "a lone chunk is also the last chunk");
    }

    #[test]
    fn delete_prefers_the_placement_id_when_the_terminal_supports_it() {
        let mut out = Vec::new();
        clear(&mut out, &kitty_caps()).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("a=d,d=i"), "delete by id");
        assert!(!s.contains("\x1b[2J"), "no need for the blunt instrument");

        // And falls back when it does not.
        let mut out = Vec::new();
        clear(&mut out, &Caps { kitty_delete: false, ..kitty_caps() }).unwrap();
        assert!(String::from_utf8(out).unwrap().contains("\x1b[2J"));
    }

    #[test]
    fn half_blocks_emit_nothing_because_they_are_ordinary_cells() {
        let mut out = Vec::new();
        emit(&mut out, &Caps::default(), &raster(26), (1, 1)).unwrap();
        assert!(out.is_empty());
    }
}
