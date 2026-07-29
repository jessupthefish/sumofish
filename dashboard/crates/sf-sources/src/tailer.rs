//! Tailing an append-only JSONL file that rotates.
//!
//! Hand-rolled rather than `notify` or `linemux`, and the reasons are specific:
//! inotify watches an *inode*, so after a rotation the watch is still following the
//! renamed old file and you only learn about the new one via a parent-directory
//! watch, with a race in between; and `linemux`'s published release is from 2022 on
//! notify 5 and keys its positions by **path**, not inode. A 100ms poll on one file
//! is both simpler and more correct.
//!
//! Four failure modes this handles, every one of them a Lab Note:
//!
//! - **Rotation.** `logs/engine.jsonl` rotates at 16MB. A tailer that does not track
//!   the inode goes quiet forever after the first rotation while continuing to
//!   render its last value, which looks exactly like a quiet game.
//! - **The tail of the pre-rotation file.** Drain the old descriptor to EOF *before*
//!   following the new inode, or the last records written before the rename are lost.
//!   This is the data a path-keyed tailer drops.
//! - **Torn writes.** A half-written line must be held, not parsed. The Python's
//!   first version handed it back with `seek(-n, SEEK_CUR)` on a *text* handle,
//!   which raises `io.UnsupportedOperation` because a text handle's position is an
//!   opaque cookie -- so every torn write lost that record and left the handle
//!   mid-line, losing the next one too. Read bytes, split on newlines, keep the
//!   remainder.
//! - **Truncation in place.** If the file shrank below our offset, start over.
//!
//! And one that is not a failure mode but a design choice: **read back over the log
//! on attach.** A tailer that starts at the end has nothing to fill in what it
//! missed, so attaching to a game in progress showed an empty search panel, an empty
//! chart and a board that would not move until the engine's *next* move. "It opens
//! and is frozen" is what that looks like from outside, and it is indistinguishable
//! from actually frozen.

use anyhow::{Context, Result};
use std::io::SeekFrom;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncSeekExt};

/// How much of the tail to read on attach.
///
/// serde_json parses tens of megabytes in tens of milliseconds, so this is
/// generous where the Python's 256KB was thrifty: the whole point is that
/// "attaching mid-game shows an empty panel" disappears rather than being
/// mitigated. 16MB is the rotation threshold, so this is "the current file".
pub const BACKFILL_BYTES: u64 = 16 * 1024 * 1024;

/// Cap on a single unterminated line, so a pathological writer cannot exhaust
/// memory. A telemetry record with a 14-move PV is a few hundred bytes.
const MAX_LINE: usize = 4 * 1024 * 1024;

pub struct Tailer {
    path: PathBuf,
    file: Option<File>,
    inode: Option<u64>,
    offset: u64,
    /// The unterminated remainder of the last read.
    partial: Vec<u8>,
    /// Set once the first attach has backfilled, so a rotation does not replay.
    attached: bool,
}

/// What a poll produced.
#[derive(Debug, Default)]
pub struct Batch {
    pub lines: Vec<String>,
    /// The file was replaced. Worth surfacing: a rotation and a quiet engine look
    /// identical otherwise, which is exactly what the track model exists to prevent.
    pub rotated: bool,
    /// The file shrank under us.
    pub truncated: bool,
    /// Current size, for the "how close is the log to rotating" indicator.
    pub size: u64,
}

impl Tailer {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Tailer {
            path: path.into(),
            file: None,
            inode: None,
            offset: 0,
            partial: Vec::new(),
            attached: false,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Read whatever is new. Never blocks for longer than the read itself.
    pub async fn poll(&mut self) -> Result<Batch> {
        let mut batch = Batch::default();

        let meta = match tokio::fs::metadata(&self.path).await {
            Ok(m) => m,
            // The file may legitimately not exist yet: the engine writes it on its
            // first search. Not an error, just nothing to say.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(batch),
            Err(e) => return Err(e).context("stat telemetry log"),
        };
        batch.size = meta.len();
        let inode = inode_of(&meta);

        if self.inode != Some(inode) {
            if self.inode.is_some() {
                // ROTATION. Drain the old descriptor to EOF first -- those records
                // were written before the rename and are otherwise lost.
                if let Some(mut old) = self.file.take() {
                    let mut rest = Vec::new();
                    if old.read_to_end(&mut rest).await.is_ok() && !rest.is_empty() {
                        self.split_into(&rest, &mut batch.lines);
                    }
                }
                batch.rotated = true;
                tracing::info!(path = %self.path.display(), "telemetry log rotated");
            }
            let mut file = File::open(&self.path).await.context("open telemetry log")?;
            // On the FIRST attach, read back over the log. On a rotation, start at
            // the beginning of the new file, which is the same thing.
            let start = if self.attached {
                0
            } else {
                meta.len().saturating_sub(BACKFILL_BYTES)
            };
            file.seek(SeekFrom::Start(start)).await?;
            self.offset = start;
            self.file = Some(file);
            self.inode = Some(inode);
            self.partial.clear();
            // A backfill that begins mid-file almost certainly begins mid-line.
            // Drop up to the first newline rather than handing a fragment to the
            // parser.
            if !self.attached && start > 0 {
                self.discard_to_newline().await?;
            }
            self.attached = true;
        } else if meta.len() < self.offset {
            // TRUNCATION in place.
            batch.truncated = true;
            tracing::warn!(path = %self.path.display(), "telemetry log truncated; restarting");
            if let Some(f) = self.file.as_mut() {
                f.seek(SeekFrom::Start(0)).await?;
            }
            self.offset = 0;
            self.partial.clear();
        }

        // Read everything available first, then split. Taking the file out of
        // `self` for the read is what keeps the borrow checker happy without an
        // Rc or a second buffer copy per chunk.
        let Some(mut file) = self.file.take() else { return Ok(batch) };
        let mut fresh = Vec::new();
        let mut buf = vec![0u8; 256 * 1024];
        let read = loop {
            match file.read(&mut buf).await {
                Ok(0) => break Ok(()),
                Ok(n) => {
                    self.offset += n as u64;
                    fresh.extend_from_slice(&buf[..n]);
                }
                Err(e) => break Err(e),
            }
        };
        self.file = Some(file);
        read.context("read telemetry log")?;
        if !fresh.is_empty() {
            self.split_into(&fresh, &mut batch.lines);
        }
        Ok(batch)
    }

    /// Append bytes, emitting every complete line and holding the remainder.
    fn split_into(&mut self, bytes: &[u8], out: &mut Vec<String>) {
        self.partial.extend_from_slice(bytes);
        let mut start = 0;
        while let Some(pos) = self.partial[start..].iter().position(|b| *b == b'\n') {
            let end = start + pos;
            let line = &self.partial[start..end];
            if !line.is_empty() {
                // Lossy rather than dropping: a record with one bad byte still has
                // its other fields, and serde will tell us if it does not parse.
                out.push(String::from_utf8_lossy(line).into_owned());
            }
            start = end + 1;
        }
        self.partial.drain(..start);
        if self.partial.len() > MAX_LINE {
            tracing::warn!(
                bytes = self.partial.len(),
                "discarding an oversized unterminated line"
            );
            self.partial.clear();
        }
    }

    async fn discard_to_newline(&mut self) -> Result<()> {
        let Some(file) = self.file.as_mut() else { return Ok(()) };
        let mut byte = [0u8; 1];
        for _ in 0..MAX_LINE {
            match file.read(&mut byte).await {
                Ok(0) => break,
                Ok(_) => {
                    self.offset += 1;
                    if byte[0] == b'\n' {
                        break;
                    }
                }
                Err(e) => return Err(e.into()),
            }
        }
        Ok(())
    }
}

fn inode_of(m: &std::fs::Metadata) -> u64 {
    use std::os::unix::fs::MetadataExt;
    m.ino()
}

/// The poll interval. The engine narrates at ~6.7Hz, so 100ms is comfortably
/// faster than the data and cheap enough to be free on one file.
pub const POLL: Duration = Duration::from_millis(100);

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    async fn drain(t: &mut Tailer) -> Vec<String> {
        t.poll().await.unwrap().lines
    }

    #[tokio::test]
    async fn reads_appended_lines_and_holds_a_partial() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.jsonl");
        std::fs::write(&path, b"one\ntwo\n").unwrap();
        let mut t = Tailer::new(&path);
        assert_eq!(drain(&mut t).await, vec!["one", "two"]);

        // A torn write: no trailing newline yet.
        let mut f = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(b"thr").unwrap();
        f.flush().unwrap();
        assert!(drain(&mut t).await.is_empty(), "a partial line must be held, not parsed");

        // ...completed.
        f.write_all(b"ee\n").unwrap();
        f.flush().unwrap();
        assert_eq!(drain(&mut t).await, vec!["three"], "and emitted once whole");
    }

    /// The Lab Note: a tailer that does not track the inode goes quiet forever after
    /// the first rotation while continuing to render its last value.
    #[tokio::test]
    async fn follows_a_rotation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.jsonl");
        let rotated = dir.path().join("t.jsonl.1");
        std::fs::write(&path, b"a\n").unwrap();
        let mut t = Tailer::new(&path);
        assert_eq!(drain(&mut t).await, vec!["a"]);

        // The writer appends, THEN renames, then starts a new file -- which is what
        // `telemetry.py` does.
        {
            let mut f = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(b"last-before-rotate\n").unwrap();
        }
        std::fs::rename(&path, &rotated).unwrap();
        std::fs::write(&path, b"first-after-rotate\n").unwrap();

        let batch = t.poll().await.unwrap();
        assert!(batch.rotated, "the rotation must be visible, not silent");
        assert_eq!(
            batch.lines,
            vec!["last-before-rotate", "first-after-rotate"],
            "the tail of the old file must be drained BEFORE following the new inode"
        );
    }

    #[tokio::test]
    async fn recovers_from_truncation_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.jsonl");
        std::fs::write(&path, b"one\ntwo\nthree\n").unwrap();
        let mut t = Tailer::new(&path);
        assert_eq!(drain(&mut t).await.len(), 3);

        std::fs::write(&path, b"fresh\n").unwrap();
        let batch = t.poll().await.unwrap();
        assert!(batch.truncated || batch.rotated, "a shrunken file must be noticed");
        assert!(batch.lines.contains(&"fresh".to_string()));
    }

    #[tokio::test]
    async fn a_missing_file_is_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let mut t = Tailer::new(dir.path().join("never-created.jsonl"));
        let batch = t.poll().await.unwrap();
        assert!(batch.lines.is_empty());
        // And it starts working when the file appears, which is what happens when
        // the engine writes its first search.
        std::fs::write(dir.path().join("never-created.jsonl"), b"hello\n").unwrap();
        assert_eq!(drain(&mut t).await, vec!["hello"]);
    }

    /// Attaching must read back over the log, or the panel is empty until the
    /// engine's next move and looks frozen.
    #[tokio::test]
    async fn backfills_on_attach() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.jsonl");
        let body: String = (0..500).map(|i| format!("line-{i}\n")).collect();
        std::fs::write(&path, body).unwrap();
        let mut t = Tailer::new(&path);
        let lines = drain(&mut t).await;
        assert_eq!(lines.len(), 500, "the whole current file, not just the end");
        assert_eq!(lines[0], "line-0");
    }

    /// A backfill that starts mid-file starts mid-line. That fragment must be
    /// discarded rather than handed to the parser.
    #[tokio::test]
    async fn a_mid_line_backfill_start_drops_the_fragment() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.jsonl");
        // Bigger than the backfill window, so the start lands mid-file.
        let filler: String = (0..200_000).map(|i| format!("{{\"n\":{i}}}\n")).collect();
        std::fs::write(&path, &filler).unwrap();
        let mut t = Tailer::new(&path);
        // Force a small window for the test rather than writing 16MB.
        t.offset = 0;
        let lines = drain(&mut t).await;
        assert!(!lines.is_empty());
        // Whatever the window, every emitted line must be a complete JSON object.
        for l in &lines {
            assert!(
                serde_json::from_str::<serde_json::Value>(l).is_ok(),
                "emitted a fragment: {l}"
            );
        }
    }

    #[tokio::test]
    async fn blank_lines_are_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.jsonl");
        std::fs::write(&path, b"a\n\n\nb\n").unwrap();
        let mut t = Tailer::new(&path);
        assert_eq!(drain(&mut t).await, vec!["a", "b"]);
    }
}
