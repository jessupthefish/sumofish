//! F1/F2/F3 — kitty graphics in Konsole.
//!
//! Konsole MR !594 (merged 2022-02-07, KDE Gear 22.04) added sixel, iTerm2 and
//! kitty graphics. It is self-described experimental, direct transfer only, no
//! animation, and "any format supported by QImage::loadFromData". This probe
//! turns those words into measurements, because the Lab Note is explicit:
//! **do not assume a terminal cannot do something without asking it.** Sixel was
//! once ruled out here on a belief and it cost the entire half-block renderer.
//!
//! Run this IN a Konsole window, never through a pipe:
//!     cargo run -p xtask -- probe-term
//!
//! Everything is written to /dev/tty rather than stdout, so the transcript stays
//! clean and a redirected stdout does not swallow the escapes. The reply is read
//! with `os_read` on the raw fd, never through a buffered reader: Python's text
//! layer once pulled a whole reply off the fd, returned one character and kept
//! the rest, which reads exactly like a terminal that does not support the query.

use anyhow::{Context, Result};
use base64::Engine as _;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::time::{Duration, Instant};

/// A 1x1 opaque pixel, as raw RGB. The smallest legal kitty transmission.
const ONE_PX_RGB: [u8; 3] = [0x7f, 0x7f, 0x7f];

/// Which visual stage to set up and then hold. Split into separate runs rather
/// than one timed sequence so each can be screenshotted without racing a sleep.
#[derive(PartialEq)]
enum Stage {
    None,
    /// F1d: transmit-and-place a test pattern, then hold.
    Display,
    /// F3: the same, plus text written on the image's own rows.
    Text,
    /// F1e: the same, then delete by image id, then hold.
    Delete,
}

pub fn run(args: &[String]) -> Result<()> {
    let stage = match args.iter().find_map(|a| a.strip_prefix("--stage=")) {
        Some("display") => Stage::Display,
        Some("text") => Stage::Text,
        Some("delete") => Stage::Delete,
        Some(other) => anyhow::bail!("unknown stage {other}"),
        None => Stage::None,
    };
    let hold = args
        .iter()
        .find_map(|a| a.strip_prefix("--hold="))
        .and_then(|s| s.parse().ok())
        .unwrap_or(25u64);

    let mut tty = OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/tty")
        .context("open /dev/tty — run this in a real terminal, not a pipe")?;

    crossterm::terminal::enable_raw_mode().context("enable raw mode")?;
    let result = probe(&mut tty, stage, hold);
    let _ = crossterm::terminal::disable_raw_mode();
    let report = result?;

    // Raw mode is off again, so ordinary println is safe.
    println!("\n{report}");
    std::fs::create_dir_all("docs").ok();
    std::fs::write("docs/probe-f1.md", &report).context("write docs/probe-f1.md")?;
    println!("wrote docs/probe-f1.md");
    Ok(())
}

fn probe(tty: &mut File, stage: Stage, hold: u64) -> Result<String> {
    let mut r = String::from("## F1/F2/F3 — kitty graphics in this terminal\n\n");
    r.push_str(&format!(
        "- `TERM={}`\n- `KONSOLE_VERSION={}`\n- `$TERM_PROGRAM={}`\n\n",
        env("TERM"),
        env("KONSOLE_VERSION"),
        env("TERM_PROGRAM"),
    ));

    // --- baseline: does it answer a primary device attributes query at all? ---
    // `4` in the reply list is the standard claim of sixel support.
    let da = ask(tty, b"\x1b[c", b'c', Duration::from_millis(400))?;
    r.push_str(&format!(
        "**DA1** (`ESC[c`) → `{}`  \nsixel claimed (a `4` in the list): **{}**\n\n",
        escape_vis(&da),
        if da.contains(";4") || da.contains("?4") { "yes" } else { "no" }
    ));

    // --- F1a: the kitty capability query ---
    // a=q asks "would you accept this?" without displaying anything. A conforming
    // terminal answers ESC_Gi=<id>;OK ESC\ . Anything else, including silence,
    // is a no. This is the positive query the Lab Note demands, rather than an
    // assumption in either direction.
    let q = kitty_cmd("i=31,s=1,v=1,a=q,t=d,f=24", &ONE_PX_RGB);
    let reply = ask_apc(tty, q.as_bytes(), Duration::from_millis(600))?;
    let f24_ok = reply.contains("OK");
    r.push_str(&format!(
        "**F1a — capability query** `a=q,f=24` → `{}`  \nverdict: **{}**\n\n",
        escape_vis(&reply),
        if f24_ok { "kitty graphics SUPPORTED" } else { "no reply / not supported" }
    ));

    // --- F2: does it accept raw RGBA (f=32)? ---
    // If yes, the PNG encode leaves the hot path entirely, at ~4x the escape bytes.
    let q32 = kitty_cmd("i=32,s=1,v=1,a=q,t=d,f=32", &[0x7f, 0x7f, 0x7f, 0xff]);
    let reply32 = ask_apc(tty, q32.as_bytes(), Duration::from_millis(600))?;
    let f32_ok = reply32.contains("OK");
    r.push_str(&format!(
        "**F2 — raw RGBA** `a=q,f=32` → `{}`  \nverdict: **{}**\n\n",
        escape_vis(&reply32),
        if f32_ok { "f=32 accepted; PNG encode is optional" } else { "f=32 rejected; PNG (f=100) is the path" }
    ));

    // --- F1b: does it accept PNG (f=100)? ---
    let png = one_px_png();
    let q100 = kitty_cmd("i=33,a=q,t=d,f=100", &png);
    let reply100 = ask_apc(tty, q100.as_bytes(), Duration::from_millis(600))?;
    let f100_ok = reply100.contains("OK");
    r.push_str(&format!(
        "**F1b — PNG** `a=q,f=100` → `{}`  \nverdict: **{}**\n\n",
        escape_vis(&reply100),
        if f100_ok { "f=100 accepted" } else { "f=100 rejected" }
    ));

    // --- F1c: chunk size tolerance ---
    // Direct transfer means the whole payload is base64 in the escape stream.
    // The protocol's chunk limit is 4096 base64 bytes; check that a multi-chunk
    // transmission is accepted, because a 1144px board is ~27 chunks and if
    // chunking is broken the whole approach dies here rather than in M2.
    if f100_ok || f24_ok {
        let big = vec![0x40u8; 64 * 64 * 3]; // 12KiB raw -> 16KiB base64 -> 4 chunks
        let sent = kitty_chunked("i=34,s=64,v=64,a=q,t=d,f=24", &big, 4096);
        let replyc = ask_apc(tty, sent.as_bytes(), Duration::from_millis(900))?;
        r.push_str(&format!(
            "**F1c — 4-chunk transmission** (64x64 RGB, 4096B chunks) → `{}`  \nverdict: **{}**\n\n",
            escape_vis(&replyc),
            if replyc.contains("OK") { "chunking works" } else { "CHUNKING FAILED — investigate before M2" }
        ));
    }

    if stage == Stage::None {
        r.push_str("Run with `--stage=display|text|delete` for the visual tests (F1d, F3, F1e).\n");
        return Ok(r);
    }

    // --- F1d: transmit and place ---
    // `a=T` transmits and places in one command, positioned by a cursor move —
    // which is how the current sixel path places the board, and the only
    // placement Konsole's direct-transfer implementation can be relied on for.
    // f=24 (raw RGB) because F2 says it is accepted, so there is no reason to
    // pay for a deflate on the hot path.
    let (w, h) = (240u32, 240u32);
    write!(tty, "\x1b[2J\x1b[H")?; // ESC[2J: the eraser of last resort
    write!(tty, "\x1b[3;3H")?; // row 3, col 3, 1-based
    let show = kitty_chunked(&format!("i=41,p=1,s={w},v={h},a=T,t=d,f=24,q=2"), &test_pattern(w, h), 4096);
    tty.write_all(show.as_bytes())?;
    tty.flush()?;
    r.push_str("**F1d — display**: 240x240 pattern placed at row 3 col 3 via `a=T,t=d,f=24,q=2`.\n");
    r.push_str("Expect a magenta border, a green diagonal and a flat grey field. Any banding,\n");
    r.push_str("speckle or colour shift would mean the transfer is being quantised.\n\n");

    // --- F3: does sibling text on the image's own rows damage it? ---
    // `rich` rewrites a whole line when any cell in it changes, which erased the
    // sixel board one row at a time and produced two separate corruption Lab
    // Notes. Before trusting `CellDiffOption::Skip` to fix that, establish
    // whether this terminal damages the picture when text lands *beside* it.
    if stage == Stage::Text || stage == Stage::Delete {
        for row in 4..14 {
            write!(tty, "\x1b[{row};40Hsibling text on image row {row}    ")?;
        }
        tty.flush()?;
        r.push_str("**F3 — sibling text** written at column 40 for rows 4..14, i.e. on the rows the\n");
        r.push_str("image occupies. If the picture is intact, the region tolerates neighbouring text\n");
        r.push_str("and `CellDiffOption::Skip` only has to stop ratatui writing *inside* the rect.\n\n");
    }

    // --- F1e: delete by image id ---
    if stage == Stage::Delete {
        tty.write_all(b"\x1b_Ga=d,d=i,i=41,q=2\x1b\\")?;
        write!(tty, "\x1b[20;1Hsent a=d,d=i,i=41 -- the image should be gone")?;
        tty.flush()?;
        r.push_str("**F1e — delete** `a=d,d=i,i=41` sent. Gone means placement ids can be reused\n");
        r.push_str("without `ESC[2J`. Still there means `ESC[2J` on geometry change stays the\n");
        r.push_str("eraser, exactly as today, and R3 has failed.\n\n");
    }

    // Hold so the window can be screenshotted, then leave the terminal clean.
    write!(tty, "\x1b[24;1Hholding {hold}s for a screenshot...")?;
    tty.flush()?;
    std::thread::sleep(Duration::from_secs(hold));
    write!(tty, "\x1b[2J\x1b[H")?;
    tty.write_all(b"\x1b_Ga=d,d=A,q=2\x1b\\")?; // delete all placements, best effort
    tty.flush()?;

    Ok(r)
}

// ---------------------------------------------------------------- escapes

/// One unchunked kitty command: `ESC _ G <control> ; <base64 payload> ESC \`
fn kitty_cmd(control: &str, payload: &[u8]) -> String {
    let b64 = base64::engine::general_purpose::STANDARD.encode(payload);
    format!("\x1b_G{control};{b64}\x1b\\")
}

/// Chunked transmission. `m=1` means "more follows", `m=0` terminates. The
/// control block is only sent with the first chunk; subsequent chunks carry `m`
/// alone. Chunk size is in base64 characters, and the protocol caps it at 4096.
fn kitty_chunked(control: &str, payload: &[u8], chunk: usize) -> String {
    let b64 = base64::engine::general_purpose::STANDARD.encode(payload);
    let bytes = b64.as_bytes();
    let mut out = String::new();
    let mut first = true;
    let mut i = 0;
    while i < bytes.len() {
        let end = (i + chunk).min(bytes.len());
        let more = if end < bytes.len() { 1 } else { 0 };
        let slice = std::str::from_utf8(&bytes[i..end]).unwrap();
        if first {
            out.push_str(&format!("\x1b_G{control},m={more};{slice}\x1b\\"));
            first = false;
        } else {
            out.push_str(&format!("\x1b_Gm={more};{slice}\x1b\\"));
        }
        i = end;
    }
    out
}

/// Write a query and read until `terminator` or timeout. `os_read` on the raw fd,
/// never a buffered reader — see the module docs.
fn ask(tty: &mut File, query: &[u8], terminator: u8, timeout: Duration) -> Result<String> {
    tty.write_all(query)?;
    tty.flush()?;
    read_until(tty, timeout, |buf| buf.last() == Some(&terminator))
}

/// The same, for an APC reply which ends `ESC \`.
fn ask_apc(tty: &mut File, query: &[u8], timeout: Duration) -> Result<String> {
    tty.write_all(query)?;
    tty.flush()?;
    read_until(tty, timeout, |buf| buf.ends_with(b"\x1b\\"))
}

fn read_until(tty: &mut File, timeout: Duration, done: impl Fn(&[u8]) -> bool) -> Result<String> {
    let deadline = Instant::now() + timeout;
    let mut buf = Vec::new();
    let mut chunk = [0u8; 256];
    while Instant::now() < deadline {
        if !readable(tty, Duration::from_millis(20)) {
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

/// `poll(2)` on the tty fd. Avoids a busy spin and avoids pulling in a whole
/// async runtime for a one-shot probe.
fn readable(tty: &File, timeout: Duration) -> bool {
    let mut fds = [libc_pollfd(tty.as_raw_fd())];
    // SAFETY: fds is a valid, correctly-sized array of pollfd for the duration
    // of the call, and the fd is owned by `tty` and open.
    let n = unsafe { poll(fds.as_mut_ptr(), 1, timeout.as_millis() as i32) };
    n > 0 && (fds[0].revents & 0x1) != 0 // POLLIN
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PollFd {
    fd: i32,
    events: i16,
    revents: i16,
}
fn libc_pollfd(fd: i32) -> PollFd {
    PollFd { fd, events: 0x1, revents: 0 }
}
unsafe extern "C" {
    fn poll(fds: *mut PollFd, nfds: u64, timeout: i32) -> i32;
}

// ---------------------------------------------------------------- payloads

/// A pattern whose damage is obvious at a glance: magenta border, green
/// diagonal, mid-grey field. Palette banding, dithering noise and partial
/// erasure are all visible in one screenshot.
fn test_pattern(w: u32, h: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let edge = x < 3 || y < 3 || x + 3 >= w || y + 3 >= h;
            let diag = x.abs_diff(y) < 4;
            out.extend_from_slice(match (edge, diag) {
                (true, _) => &[0xff, 0x00, 0xff],
                (_, true) => &[0x00, 0xff, 0x00],
                _ => &[0x60, 0x60, 0x60],
            });
        }
    }
    out
}

fn one_px_png() -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, 1, 1);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        let mut w = enc.write_header().expect("png header");
        w.write_image_data(&ONE_PX_RGB).expect("png data");
    }
    out
}

// ---------------------------------------------------------------- misc

fn env(k: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| "(unset)".into())
}

/// Render control bytes readable so the report is copy-pasteable.
fn escape_vis(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\x1b' => "<ESC>".to_string(),
            c if c.is_control() => format!("<{:02x}>", c as u32),
            c => c.to_string(),
        })
        .collect()
}
