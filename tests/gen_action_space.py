#!/usr/bin/env python
"""Generate `rust/src/action_space.rs` FROM `sumofish/tokenizer.py`.

The Python action space is byte-verified against DeepMind's published
implementation over 12,000 real positions, and every published number in this
project depends on it staying that way. So the Rust side does not reimplement
it -- reimplementation would be a second thing that can drift, and the drift
would be silent: a wrong action index attaches a prior to the wrong move, the
move played is still legal, and nothing fails.

Instead the table is emitted from the verified one, and
`tests/verify_action_space.py` re-derives it and fails on any disagreement.

Run from the worktree root, with a python that can import sumofish:
    .venv/bin/python tests/gen_action_space.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for candidate in (ROOT, ROOT.parent / "sumofish"):
    if (candidate / "sumofish" / "tokenizer.py").exists():
        sys.path.insert(0, str(candidate))
        break

from sumofish.tokenizer import ACTION_BY_MOVE_KEY, NUM_ACTIONS  # noqa: E402

OUT = ROOT / "rust" / "src" / "action_space.rs"

HEADER = '''//! The 1968-move action space, GENERATED from `sumofish/tokenizer.py`.
//!
//! Do not hand-edit and do not reimplement. The Python table is byte-verified
//! against DeepMind's published implementation over 12,000 real positions, and
//! every published number in the project depends on it. A second, independent
//! construction here would be a second thing that can drift -- silently, because
//! a wrong action index attaches a prior to the wrong move, the move played is
//! still legal, and nothing fails.
//!
//! So this file is emitted BY that table, and `tests/verify_action_space.py`
//! re-derives it and fails if the two disagree. That is what makes the generated
//! copy trustworthy rather than merely convenient.
//!
//! Regenerate with `tests/gen_action_space.py`.
//!
//! Keying matches `tokenizer.move_key`: `from | to << 6 | promotion << 12`,
//! with promotion 0 when absent. 16 bits is exactly enough: 6 + 6 + 4.

use std::sync::LazyLock;

/// sha256 prefix of the (move_key, action) pairs this was generated from.
/// `verify_action_space.py` checks it, so a stale file is detectable.
pub const SOURCE_DIGEST: &str = "{digest}";

pub const NUM_ACTIONS: usize = {num_actions};

/// Flat lookup, exactly `tokenizer.ACTION_BY_MOVE_KEY`: -1 where no move maps.
pub static ACTION_BY_MOVE_KEY: LazyLock<Vec<i32>> = LazyLock::new(|| {{
    let mut t = vec![-1i32; 1 << 16];
    for &(key, action) in ENTRIES.iter() {{
        t[key as usize] = action as i32;
    }}
    t
}});

/// Pack a move the way `tokenizer.move_key` does.
#[inline]
pub fn move_key(from: u8, to: u8, promotion: u8) -> u16 {{
    (from as u16) | ((to as u16) << 6) | ((promotion as u16) << 12)
}}

/// (move_key, action) for every move in the action space. {n_entries} entries.
pub const ENTRIES: [(u16, u16); {n_entries}] = [
'''


def main() -> int:
    pairs = [(k, a) for k, a in enumerate(ACTION_BY_MOVE_KEY) if a >= 0]
    digest = hashlib.sha256(repr(pairs).encode()).hexdigest()[:16]

    body = HEADER.format(
        digest=digest, num_actions=NUM_ACTIONS, n_entries=len(pairs)
    )
    lines = []
    for i in range(0, len(pairs), 8):
        chunk = pairs[i : i + 8]
        lines.append("    " + " ".join(f"({k},{a})," for k, a in chunk))
    body += "\n".join(lines) + "\n];\n"

    OUT.write_text(body)
    print(f"wrote {OUT}: {len(pairs)} entries, NUM_ACTIONS={NUM_ACTIONS}, digest={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
