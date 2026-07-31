#!/usr/bin/env python
"""Does `think_time()`'s floor ever ask for more than the clock actually has?

No GPU, no network, no engine boot -- this is a pure function, so this is a
handful of direct assertions rather than an oracle script with a position
corpus. It still lives in `tests/` and `run_all.sh`, because the bug it
guards against was found live, not by review: `MIN_SECONDS` (the floor,
"always look at something") was applied AFTER the ceiling clamp
(`remaining_s * MAX_FRACTION`), so at very low remaining time the floor could
win and hand back a budget bigger than the side to move's entire remaining
clock. `sumofish-bot.service` plays live blitz/bullet, where a sub-200ms
clock is a real, repeated scramble, not a corner case -- and losing on time
is a strictly worse outcome than an instamove, per this module's own
docstring.

Usage:
    tests/verify_think_time.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sumofish.engines.search_engine import MAX_FRACTION, MIN_SECONDS, think_time
from sumofish.uci import Limits


def check(name: str, limits: Limits, side: chess.Color, predicate, describe) -> bool:
    budget = think_time(limits, side)
    ok = predicate(budget)
    status = "ok" if ok else "FAIL"
    print(f"  [{status}] {name}: budget={budget * 1000:.2f}ms  ({describe(budget)})")
    return ok


def main() -> int:
    results = []

    # The exact regression: 30ms left, no increment. Before the fix,
    # think_time(wtime=30, winc=0) returned 0.05 (50ms) -- more than the
    # entire remaining clock.
    results.append(check(
        "the reported scenario (wtime=30ms, winc=0)",
        Limits(wtime=30, btime=30, winc=0, binc=0), chess.WHITE,
        lambda b: b * 1000 < 30,
        lambda b: "must be strictly less than the 30ms actually remaining",
    ))

    # An even more extreme scramble: does the absolute clamp hold as
    # remaining_s shrinks further, not just at one hand-picked value?
    for ms in (20, 10, 5, 2, 1):
        # <=, not <: the invariant is "never MORE than what's left", not
        # "always strictly less". At remaining_s this small the epsilon floor
        # can legitimately equal the whole remaining clock (e.g. 1ms left,
        # 1ms requested) -- using all of an already-catastrophic clock is not
        # the bug this guards against; asking for MORE than it is.
        results.append(check(
            f"extreme scramble (wtime={ms}ms)",
            Limits(wtime=ms, btime=ms, winc=0, binc=0), chess.WHITE,
            lambda b, ms=ms: 0 < b * 1000 <= ms,
            lambda b: "must stay in (0, remaining] -- never zero, never over",
        ))

    # A real increment should not let the budget exceed the clock either.
    results.append(check(
        "low time with a real increment (wtime=50ms, winc=100ms)",
        Limits(wtime=50, btime=50, winc=100, binc=100), chess.WHITE,
        lambda b: b * 1000 < 50,
        lambda b: "must stay under the 50ms remaining even with a big increment",
    ))

    # Normal play must be unaffected: this fix should only bite in the
    # low-time regime, not change ordinary time management.
    results.append(check(
        "ordinary time control (wtime=30000ms, winc=0)",
        Limits(wtime=30_000, btime=30_000, winc=0, binc=0), chess.WHITE,
        lambda b: abs(b - 30_000 / 1000 / 30) < 0.05,
        lambda b: f"should be ~{30_000 / 1000 / 30:.2f}s, DIVISOR=30, unaffected by the clamp",
    ))

    # The ceiling (MAX_FRACTION) must still bind in the regime it was meant
    # for -- this fix must not have loosened it.
    results.append(check(
        "ceiling still binds well above the danger zone (wtime=1000ms)",
        Limits(wtime=1_000, btime=1_000, winc=0, binc=0), chess.WHITE,
        lambda b: b <= 1.0 * MAX_FRACTION + 1e-9,
        lambda b: f"must not exceed the {MAX_FRACTION:.0%} ceiling",
    ))

    # The floor (MIN_SECONDS) must still apply when the clock can afford it.
    results.append(check(
        "floor still applies when the clock can afford it (wtime=2000ms)",
        Limits(wtime=2_000, btime=2_000, winc=0, binc=0), chess.WHITE,
        lambda b: b >= MIN_SECONDS - 1e-9,
        lambda b: f"must be at least the {MIN_SECONDS * 1000:.0f}ms floor",
    ))

    if all(results):
        print(f"\nOK: {len(results)}/{len(results)} checks passed")
        return 0
    print(f"\nFAIL: {sum(results)}/{len(results)} checks passed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
