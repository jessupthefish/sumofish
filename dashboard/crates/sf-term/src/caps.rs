//! What can this terminal actually do?
//!
//! **Ask it. Do not assume.** The project's own Lab Note: sixel was ruled out here
//! on the belief that Konsole's support was off by default and unreliable. It was
//! neither, and that wrong assumption cost the entire half-block renderer. The
//! mirror of the mistake is equally available -- assuming kitty graphics work
//! because the merge request exists -- so this sends a real query and reads a real
//! answer, and falls back down a ladder when it does not get one.
//!
//! Measured on Konsole 26.04.3 in the M0 probe: kitty graphics supported, raw RGB
//! (`f=24`) and RGBA (`f=32`) both accepted, chunked transmission fine, and delete
//! by image id works. See `docs/probe-results.md`.

use anyhow::Result;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::time::{Duration, Instant};

/// How the board picture gets onto the screen, best first.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Graphics {
    /// The kitty graphics protocol. No palette, no dithering, placement by id, and
    /// explicit deletion -- so no quantisation shimmer and no full-screen wipe on
    /// a resize.
    Kitty,
    /// Sixel. Paletted, so it needs care: dithering must be off or the flat board
    /// visibly shimmers between frames, and re-emitting an unchanged image makes
    /// the terminal clear the region first, so the board strobes at the frame rate.
    Sixel,
    /// Unicode half-blocks from the same rasteriser. Works everywhere.
    #[default]
    HalfBlocks,
    /// The user asked for no picture at all.
    None,
}

impl Graphics {
    pub const fn name(self) -> &'static str {
        match self {
            Graphics::Kitty => "kitty",
            Graphics::Sixel => "sixel",
            Graphics::HalfBlocks => "halfblocks",
            Graphics::None => "none",
        }
    }
    pub fn parse(s: &str) -> Option<Graphics> {
        match s {
            "kitty" => Some(Graphics::Kitty),
            "sixel" => Some(Graphics::Sixel),
            "halfblocks" | "text" => Some(Graphics::HalfBlocks),
            "off" | "none" => Some(Graphics::None),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Caps {
    pub graphics: Graphics,
    /// Pixel size of one terminal cell. Needed to turn the board region's cells
    /// into an image size, and re-probed on every resize because a font change
    /// invalidates it and the picture would silently be the wrong shape.
    pub cell: sf_layout::CellSize,
    /// Whether the terminal answered the kitty capability query, as opposed to us
    /// falling back. Reported on screen so a surprise is diagnosable.
    pub kitty_confirmed: bool,
    /// Whether `a=d,d=i` deletion works. When false, `ESC[2J` on a geometry change
    /// stays the eraser, exactly as the Python did.
    pub kitty_delete: bool,
}

impl Default for Caps {
    fn default() -> Self {
        Caps {
            graphics: Graphics::HalfBlocks,
            cell: sf_layout::CellSize::default(),
            kitty_confirmed: false,
            kitty_delete: false,
        }
    }
}

/// Probe the terminal.
///
/// **Ordering matters.** This must run after entering the alternate screen and
/// **before** the event reader starts, or the capability replies get consumed as
/// input and the terminal looks like it supports nothing.
///
/// `force` short-circuits the query, for `--board=sixel` and for tests.
pub fn probe(force: Option<Graphics>) -> Caps {
    let mut caps = Caps::default();

    // Cell size first: needed whatever the graphics answer, and cheap.
    if let Some(cell) = query_cell_size() {
        caps.cell = cell;
    }

    if let Some(g) = force {
        caps.graphics = g;
        caps.kitty_confirmed = g == Graphics::Kitty;
        // A forced protocol gets the conservative eraser, since nothing was asked.
        caps.kitty_delete = false;
        tracing::info!(graphics = g.name(), "graphics forced by config");
        return caps;
    }

    match kitty_supported() {
        Ok(true) => {
            caps.graphics = Graphics::Kitty;
            caps.kitty_confirmed = true;
            caps.kitty_delete = kitty_delete_supported().unwrap_or(false);
        }
        Ok(false) | Err(_) => {
            // A `4` in the primary device attributes is the standard claim of
            // sixel support.
            if sixel_claimed().unwrap_or(false) {
                caps.graphics = Graphics::Sixel;
            }
        }
    }
    tracing::info!(
        graphics = caps.graphics.name(),
        cell_w = caps.cell.w,
        cell_h = caps.cell.h,
        kitty_delete = caps.kitty_delete,
        "terminal probed"
    );
    caps
}

/// `ESC_Gi=..,a=q,..` — "would you accept this?", which displays nothing. A
/// conforming terminal answers `ESC_Gi=<id>;OK ESC\`. Silence is a no.
fn kitty_supported() -> Result<bool> {
    let reply = with_tty(|tty| {
        // A 1x1 raw RGB pixel: the smallest legal transmission.
        let payload = base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            [0x7fu8, 0x7f, 0x7f],
        );
        ask_apc(tty, format!("\x1b_Gi=1,s=1,v=1,a=q,t=d,f=24;{payload}\x1b\\").as_bytes())
    })?;
    Ok(reply.contains("OK"))
}

fn kitty_delete_supported() -> Result<bool> {
    // There is no query for deletion, so transmit a 1x1, delete it, and ask
    // whether the delete was accepted. `q=1` suppresses the OK for success but
    // still reports errors, so an empty reply is the pass.
    let reply = with_tty(|tty| {
        let payload = base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            [0x7fu8, 0x7f, 0x7f],
        );
        tty.write_all(format!("\x1b_Gi=9997,s=1,v=1,a=t,t=d,f=24,q=2;{payload}\x1b\\").as_bytes())?;
        ask_apc(tty, b"\x1b_Ga=d,d=i,i=9997,q=1\x1b\\")
    })?;
    Ok(!reply.contains("ENOENT") && !reply.contains("EINVAL"))
}

fn sixel_claimed() -> Result<bool> {
    let reply = with_tty(|tty| ask(tty, b"\x1b[c", b'c'))?;
    Ok(reply.contains(";4") || reply.contains("?4"))
}

/// `ESC[16t` reports the cell size in pixels as `ESC[6;<h>;<w>t`.
fn query_cell_size() -> Option<sf_layout::CellSize> {
    let reply = with_tty(|tty| ask(tty, b"\x1b[16t", b't')).ok()?;
    let body = reply.trim_start_matches('\x1b').trim_start_matches('[');
    let parts: Vec<&str> = body.trim_end_matches('t').split(';').collect();
    if parts.len() != 3 || parts[0] != "6" {
        return None;
    }
    let h: u16 = parts[1].parse().ok()?;
    let w: u16 = parts[2].parse().ok()?;
    // Reject absurd answers rather than making the picture the wrong shape.
    (w > 1 && w < 64 && h > 1 && h < 128).then_some(sf_layout::CellSize { w, h })
}

fn with_tty<T>(f: impl FnOnce(&mut File) -> Result<T>) -> Result<T> {
    let mut tty = OpenOptions::new().read(true).write(true).open("/dev/tty")?;
    f(&mut tty)
}

fn ask(tty: &mut File, query: &[u8], terminator: u8) -> Result<String> {
    tty.write_all(query)?;
    tty.flush()?;
    read_until(tty, Duration::from_millis(300), |b| b.last() == Some(&terminator))
}

fn ask_apc(tty: &mut File, query: &[u8]) -> Result<String> {
    tty.write_all(query)?;
    tty.flush()?;
    read_until(tty, Duration::from_millis(400), |b| b.ends_with(b"\x1b\\"))
}

/// `os_read` on the raw fd, never a buffered reader. Python's text layer once
/// pulled a whole reply off the fd, returned one character and kept the rest, so
/// `select` reported nothing pending and the loop exited with a bare ESC -- which
/// reads exactly like a terminal that does not support the query, and the remainder
/// then surfaced inside the *next* query's answer.
fn read_until(tty: &mut File, timeout: Duration, done: impl Fn(&[u8]) -> bool) -> Result<String> {
    let deadline = Instant::now() + timeout;
    let mut buf = Vec::new();
    let mut chunk = [0u8; 256];
    while Instant::now() < deadline {
        if !readable(tty, Duration::from_millis(15)) {
            continue;
        }
        match tty.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                if done(&buf) {
                    break;
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PollFd {
    fd: i32,
    events: i16,
    revents: i16,
}

unsafe extern "C" {
    fn poll(fds: *mut PollFd, nfds: u64, timeout: i32) -> i32;
}

fn readable(tty: &File, timeout: Duration) -> bool {
    let mut fds = [PollFd { fd: tty.as_raw_fd(), events: 0x1, revents: 0 }];
    // SAFETY: a valid one-element pollfd array whose fd is owned and open for the
    // duration of the call.
    let n = unsafe { poll(fds.as_mut_ptr(), 1, timeout.as_millis() as i32) };
    n > 0 && (fds[0].revents & 0x1) != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graphics_names_round_trip() {
        for g in [Graphics::Kitty, Graphics::Sixel, Graphics::HalfBlocks, Graphics::None] {
            assert_eq!(Graphics::parse(g.name()), Some(g));
        }
        assert_eq!(Graphics::parse("text"), Some(Graphics::HalfBlocks));
        assert_eq!(Graphics::parse("off"), Some(Graphics::None));
        assert_eq!(Graphics::parse("nonsense"), None);
    }

    #[test]
    fn a_forced_protocol_skips_the_query() {
        // No tty in a test runner, so this also proves the probe does not hang or
        // panic when /dev/tty is unavailable.
        let caps = probe(Some(Graphics::Sixel));
        assert_eq!(caps.graphics, Graphics::Sixel);
        assert!(!caps.kitty_confirmed);
    }

    #[test]
    fn the_default_is_the_universal_fallback() {
        assert_eq!(Caps::default().graphics, Graphics::HalfBlocks);
    }
}
