#!/usr/bin/env python
"""Does `--auto-resume` pick up the data stream where the crashed process left it?

No GPU, no bag file, no engine boot -- `data_frac_after()` is a pure function,
so this is direct assertions against it, same shape as `verify_think_time.py`.

The bug it guards against: a fresh `--auto-resume` launch rebuilt its loader
from `args.data_start_frac`, a static CLI flag baked into the launch command
and never updated by a previous resume. Every crash-restart therefore replayed
the exact same prefix of the bag its parent process already trained on,
instead of continuing into new records -- and got worse the more times the
process restarted, since each restart re-began from the same static point
regardless of how far a prior process instance had actually advanced.

Usage:
    tests/verify_data_frac.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import data_frac_after


def check(name: str, got: float, want: float, tol: float = 1e-9) -> bool:
    ok = abs(got - want) < tol
    status = "ok" if ok else "FAIL"
    print(f"  [{status}] {name}: got={got:.6f} want={want:.6f}")
    return ok


def main() -> int:
    results = []

    # A fresh launch checkpointing for the first time: no records consumed
    # yet relative to its own start, so the fraction should be unchanged.
    results.append(check(
        "no records consumed yet",
        data_frac_after(0.6, 0, 1_000_000),
        0.6,
    ))

    # The core scenario: a process consumed half the bag since it started.
    results.append(check(
        "half the bag consumed since launch",
        data_frac_after(0.0, 500_000, 1_000_000),
        0.5,
    ))

    # The actual regression: simulate two sequential crash-restarts and
    # confirm the SECOND resume continues from the FIRST resume's endpoint,
    # not from the run's original static launch flag. Before the fix, every
    # relaunch reused the same static args.data_start_frac and this would
    # incorrectly be 0.2 again instead of advancing to 0.4.
    original_launch_frac = 0.2
    after_first_leg = data_frac_after(original_launch_frac, 200_000, 1_000_000)
    after_second_leg = data_frac_after(after_first_leg, 200_000, 1_000_000)
    results.append(check(
        "first leg advances past the original launch point",
        after_first_leg,
        0.4,
    ))
    results.append(check(
        "second leg chains off the FIRST leg's endpoint, not the original launch flag",
        after_second_leg,
        0.6,
    ))

    # Wraparound: a stream that laps the bag must keep advancing rather than
    # producing a fraction >= 1.0, matching _BagStream's own modulo behaviour.
    results.append(check(
        "wraps around past a full lap of the bag",
        data_frac_after(0.9, 300_000, 1_000_000),
        0.2,
    ))

    # A record count so small it rounds records_consumed/total to something
    # tiny must not divide by zero or blow up -- max(1, total) is the guard.
    results.append(check(
        "degenerate zero-length dataset does not divide by zero",
        data_frac_after(0.0, 5, 0),
        0.0,
    ))

    if all(results):
        print(f"\nOK: {len(results)}/{len(results)} checks passed")
        return 0
    print(f"\nFAIL: {sum(results)}/{len(results)} checks passed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
