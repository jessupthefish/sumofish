#!/usr/bin/env python
"""Does the engine warn when a Rust-only flag is set but the Python core is running?

No GPU, no engine boot -- `ignored_rust_flags()` is a pure function over an
explicit env mapping, same shape as `verify_think_time.py` /
`verify_data_frac.py`.

The bug it guards against: CHESSGPU_DEDUP/COMPILE/MATE_DISTANCE/VLOSS_FIX are
constructor args the Rust MCTS takes and the Python MCTS does not. Setting one
of them without also setting CHESSGPU_CORE=rust was a silent no-op -- nothing
at runtime (boot telemetry, the dashboard, the engine's own output) showed the
mistake. Same invisible-failure shape as the root_q/top()/pv() bugs found
elsewhere in this project the same night this was caught, just for a
config-typo instead of a missing binding.

Usage:
    tests/verify_rust_flag_guard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chessgpu.engines.search_engine import RUST_ONLY_FLAGS, ignored_rust_flags


def check(name: str, got, want, describe=lambda g: "") -> bool:
    ok = got == want
    status = "ok" if ok else "FAIL"
    print(f"  [{status}] {name}: got={got} want={want} {describe(got)}")
    return ok


def main() -> int:
    results = []

    results.append(check(
        "nothing set, nothing ignored",
        ignored_rust_flags({}),
        [],
    ))

    results.append(check(
        "every flag explicitly off is not reported as ignored",
        ignored_rust_flags({name: "0" for name in RUST_ONLY_FLAGS}),
        [],
    ))

    # The actual regression: a rollback edit that sets the flag to a falsy-
    # looking string must not be reported as ignored (it isn't ON, so there's
    # nothing to warn about) -- this also indirectly checks the same
    # true/false parsing bug documented in env_flag's own history.
    results.append(check(
        "'false' parses as off, not reported",
        ignored_rust_flags({"CHESSGPU_VLOSS_FIX": "false"}),
        [],
    ))

    # The real scenario: CHESSGPU_VLOSS_FIX=1 with no CHESSGPU_CORE set (the
    # exact deployment shape this project's own systemd units use for a
    # staged-but-not-yet-core-switched flag).
    results.append(check(
        "one flag on is reported by name",
        ignored_rust_flags({"CHESSGPU_VLOSS_FIX": "1"}),
        ["CHESSGPU_VLOSS_FIX"],
    ))

    # Multiple flags on: all of them surface, in RUST_ONLY_FLAGS's own order,
    # so the warning message is deterministic rather than dict-order-dependent.
    results.append(check(
        "multiple flags on are all reported, in a stable order",
        ignored_rust_flags({"CHESSGPU_VLOSS_FIX": "1", "CHESSGPU_DEDUP": "yes"}),
        ["CHESSGPU_DEDUP", "CHESSGPU_VLOSS_FIX"],
    ))

    # An unrelated env var (e.g. CHESSGPU_CORE itself, or CHESSGPU_SIMS) must
    # not be mistaken for one of the four -- only the exact four names count.
    results.append(check(
        "unrelated CHESSGPU_* vars are not mistaken for a Rust-only flag",
        ignored_rust_flags({"CHESSGPU_CORE": "rust", "CHESSGPU_SIMS": "5000"}),
        [],
    ))

    if all(results):
        print(f"\nOK: {len(results)}/{len(results)} checks passed")
        return 0
    print(f"\nFAIL: {sum(results)}/{len(results)} checks passed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
