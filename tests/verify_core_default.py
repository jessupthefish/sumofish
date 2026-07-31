#!/usr/bin/env python
"""Is Rust actually the default core, and is `python` still a real opt-out?

Flipped 2026-07-31: `select_mcts_class()` used to default to Python with Rust
opt-in via `CHESSGPU_CORE=rust`. Rust has been running every rated game since
2026-07-30 (see STATE.md) on the strength of the identity proof against
`sumofish.mcts`, not on the env var, so it earned the default. Python was NOT
retired -- it's still the oracle `tests/identity_*.py`, `scripts/match.py` and
`scripts/acceptance.py` compare the Rust core against, still fully reachable
via `CHESSGPU_CORE=python` -- it just stopped being what an absent env var
silently falls back to. This checks exactly that one thing changed.

No GPU, no engine boot -- `select_mcts_class()` only branches on
`os.environ` and returns classes; nothing here is ever instantiated.

Usage:
    tests/verify_core_default.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from contextlib import contextmanager

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sumofish.mcts import MCTS
from sumofish.rust_mcts import RustMCTS, select_mcts_class


@contextmanager
def core_env(value):
    """Set (or unset, for `None`) CHESSGPU_CORE, and always restore it."""
    had = "CHESSGPU_CORE" in os.environ
    old = os.environ.get("CHESSGPU_CORE")
    try:
        if value is None:
            os.environ.pop("CHESSGPU_CORE", None)
        else:
            os.environ["CHESSGPU_CORE"] = value
        yield
    finally:
        if had:
            os.environ["CHESSGPU_CORE"] = old
        else:
            os.environ.pop("CHESSGPU_CORE", None)


def check(name: str, got, want) -> bool:
    ok = got == want
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: got={got} want={want}")
    return ok


def main() -> int:
    results = []

    with core_env(None):
        results.append(check("unset CHESSGPU_CORE selects rust", select_mcts_class(), (RustMCTS, "rust")))

    with core_env(""):
        results.append(check("empty CHESSGPU_CORE selects rust", select_mcts_class(), (RustMCTS, "rust")))

    with core_env("rust"):
        results.append(check("CHESSGPU_CORE=rust selects rust", select_mcts_class(), (RustMCTS, "rust")))

    with core_env("RUST"):
        results.append(check("CHESSGPU_CORE is case-insensitive", select_mcts_class(), (RustMCTS, "rust")))

    with core_env("python"):
        results.append(
            check("CHESSGPU_CORE=python still opts out to the oracle", select_mcts_class(), (MCTS, "python"))
        )

    with core_env("rusty"):
        try:
            select_mcts_class()
            results.append(check("a typo raises rather than silently picking one", "no raise", "ValueError"))
        except ValueError:
            results.append(check("a typo raises rather than silently picking one", "ValueError", "ValueError"))

    if all(results):
        print(f"\nOK: {len(results)}/{len(results)} checks passed")
        return 0
    print(f"\nFAIL: {sum(results)}/{len(results)} checks passed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
