#!/usr/bin/env python
"""Instamove and early stopping. No GPU required.

Both exist because `scripts/operating_point.py` measured, on 2026-08-13 over 336
logged moves, that the engine spent 100% of its allowance on EVERY move and
never once took under a second -- including in positions with a single legal
reply, where it thought for 19 seconds.

The property that has to hold for early stopping to be allowed anywhere near a
rated game: **stopping must never change WHICH move is played, only when.** The
move is `max(visits)` in both cases, so it is enough that the rule only fires
when the runner-up cannot catch the leader even if every remaining simulation
were handed to it.

That is checked two ways here, because a hand-picked example is not a proof:
  * directly, on cases chosen to sit either side of the boundary;
  * as an ALGEBRAIC bound. With `lead <= done` always true (the leader cannot
    lead by more visits than have been played), the rule
        lead > SAFETY * rate * remaining
    cannot fire before a fraction SAFETY/(1+SAFETY) of the budget has elapsed.
    At SAFETY=1.25 that is 5/9 = 0.5556. A regression that made the rule fire
    earlier than that would be a real loss of safety margin, and it would not
    show up in any single example.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chess  # noqa: E402

from sumofish.engines.search_engine import (  # noqa: E402
    EARLY_STOP_MIN_FRACTION, EARLY_STOP_SAFETY, INSTAMOVE_SECONDS, decided,
    think_time,
)


def main() -> int:
    failures: list[str] = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    # -- a forced move is decided before it is searched --------------------
    check(decided([100], 100, 5.0, 5.0, 10.0),
          "a single candidate must be decided immediately")
    check(decided([], 0, 5.0, 5.0, 10.0),
          "no candidates must not keep searching")

    # -- the rate estimate is not trusted early ----------------------------
    early = EARLY_STOP_MIN_FRACTION * 10.0 * 0.5
    check(not decided([10_000, 1], 10_001, early, 0.001, 10.0),
          "must not stop before EARLY_STOP_MIN_FRACTION of the budget, however "
          "lopsided the position looks")

    # -- the boundary, either side -----------------------------------------
    # done=1000 in 5.0s -> rate 200/s; remaining 1.0s -> 200 more sims possible;
    # threshold = 1.25 * 200 = 250.
    check(not decided([1000 + 250, 1000], 1000, 5.0, 1.0, 10.0),
          "lead exactly at the threshold must NOT stop (strict inequality)")
    check(decided([1000 + 251, 1000], 1000, 5.0, 1.0, 10.0),
          "lead one visit past the threshold must stop")

    # -- a close race never stops ------------------------------------------
    check(not decided([520, 480], 1000, 9.0, 1.0, 10.0),
          "a 40-visit lead with 200 simulations left must not stop")

    # -- the algebraic bound, swept ----------------------------------------
    #
    # lead <= done always. So with rate = done/elapsed, the most extreme
    # possible position (leader has every visit) still cannot satisfy
    #     done > SAFETY * (done/elapsed) * remaining
    # until elapsed > SAFETY * remaining, i.e. until a fraction
    # SAFETY/(1+SAFETY) of the budget has gone.
    budget = 10.0
    bound = EARLY_STOP_SAFETY / (1.0 + EARLY_STOP_SAFETY)
    fired_before_bound = []
    for i in range(1, 1000):
        frac = i / 1000.0
        elapsed = frac * budget
        remaining = budget - elapsed
        done = 100_000
        # The most lopsided root possible: one move has everything but a visit.
        if decided([done - 1, 1], done, elapsed, remaining, budget) and frac < bound:
            fired_before_bound.append(frac)
    check(not fired_before_bound,
          f"the rule fired at elapsed fractions {fired_before_bound[:5]}, below "
          f"the algebraic floor of {bound:.4f}. Either EARLY_STOP_SAFETY moved "
          f"or the inequality did.")

    # And it must actually be reachable, or it is a no-op wearing a feature's hat.
    check(decided([100_000 - 1, 1], 100_000, 0.95 * budget, 0.05 * budget, budget),
          "the rule never fires even in the most lopsided possible position, "
          "which would make early stopping dead code")

    # -- the game clock ----------------------------------------------------
    #
    # match.py's Clock is what makes any of the above measurable: --time is
    # seconds per MOVE, and under it time saved on one move goes nowhere, which
    # is exactly what early stopping exists to exploit.
    sys.path.insert(0, str(ROOT / "scripts"))
    from match import Clock  # noqa: E402

    c = Clock(60.0, 1.0)
    check(c.remaining[chess.WHITE] == 60.0 and c.remaining[chess.BLACK] == 60.0,
          "both sides start on the base time")

    # A normal move: bill it, then add the increment.
    check(c.charge(chess.WHITE, 5.0), "a 5s move on a 60s clock must not flag")
    check(abs(c.remaining[chess.WHITE] - 56.0) < 1e-9,
          f"60 - 5 + 1 should be 56, got {c.remaining[chess.WHITE]}")
    check(abs(c.spent[chess.WHITE] - 5.0) < 1e-9, "spent must track the raw cost")
    check(c.remaining[chess.BLACK] == 60.0, "charging one side must not touch the other")

    # The increment is added AFTER the deduction and only on survival. Any other
    # order makes it impossible to lose on time in a game with an increment.
    c2 = Clock(10.0, 5.0)
    check(not c2.charge(chess.WHITE, 12.0),
          "overrunning the clock must flag even when the increment would cover it")
    check(c2.remaining[chess.WHITE] < 0,
          "a flagged clock stays negative rather than being topped up")

    # Exactly on the buzzer is not a flag.
    c3 = Clock(10.0, 0.0)
    check(c3.charge(chess.WHITE, 10.0), "spending exactly the clock must not flag")

    # Limits are milliseconds, converted in one place only.
    lim = Clock(60.0, 1.5).limits()
    check(lim.wtime == 60_000 and lim.btime == 60_000,
          f"clock -> Limits must be ms, got wtime={lim.wtime}")
    check(lim.winc == 1500 and lim.binc == 1500,
          f"increment -> Limits must be ms, got winc={lim.winc}")
    check(think_time(lim, chess.WHITE) > 0, "think_time must accept a Clock's Limits")

    check(INSTAMOVE_SECONDS > 0,
          "a forced move gets a SMALL budget, not zero: rust/src/tree.rs "
          "reroot() declines after more than 2 plies, so skipping the search "
          "entirely discards the tree on the following move")

    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"verify_time_management: OK (early stop cannot fire before "
          f"{bound:.4f} of the budget, cannot change the move, and is reachable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
