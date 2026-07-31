#!/usr/bin/env python
"""Verify `rust/src/action_space.rs` still matches `sumofish/tokenizer.py`.

The Rust table is GENERATED from the Python one (see `gen_action_space.py`), so
the risk is not a bad translation, it is a STALE file: the tokenizer changes and
the generated copy quietly does not. A wrong action index attaches a prior to the
wrong move, every move played is still legal, and nothing else fails.

This re-derives the table and compares, including the digest recorded in the
generated header.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for candidate in (ROOT, ROOT.parent / "sumofish"):
    if (candidate / "sumofish" / "tokenizer.py").exists():
        sys.path.insert(0, str(candidate))
        break
from sumofish.tokenizer import ACTION_BY_MOVE_KEY, NUM_ACTIONS  # noqa: E402

SRC = ROOT / "rust" / "src" / "action_space.rs"


def main() -> int:
    text = SRC.read_text()

    pairs = [(k, a) for k, a in enumerate(ACTION_BY_MOVE_KEY) if a >= 0]
    digest = hashlib.sha256(repr(pairs).encode()).hexdigest()[:16]

    got_digest = re.search(r'SOURCE_DIGEST: &str = "([0-9a-f]*)"', text).group(1)
    got_num = int(re.search(r"NUM_ACTIONS: usize = (\d+)", text).group(1))
    got_pairs = [
        (int(k), int(a)) for k, a in re.findall(r"\((\d+),(\d+)\)", text)
    ]

    problems = []
    if got_num != NUM_ACTIONS:
        problems.append(f"NUM_ACTIONS: rust={got_num} python={NUM_ACTIONS}")
    if got_pairs != pairs:
        problems.append(
            f"entries differ: rust has {len(got_pairs)}, python has {len(pairs)}"
        )
        for (gk, ga), (pk, pa) in zip(got_pairs, pairs):
            if (gk, ga) != (pk, pa):
                problems.append(f"  first divergence: rust=({gk},{ga}) python=({pk},{pa})")
                break
    if got_digest != digest:
        problems.append(
            f"digest: rust={got_digest} python={digest} "
            "(regenerate with tests/gen_action_space.py)"
        )

    if problems:
        for p in problems:
            print(f"FAIL {p}")
        return 1
    print(f"OK: action space in sync, {len(pairs)} entries, digest {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
