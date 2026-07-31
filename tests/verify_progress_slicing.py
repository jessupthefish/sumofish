#!/usr/bin/env python
"""Does `RustMCTS.search()` actually report progress live now, or still not at all?

No GPU, no real networks: `RustMCTS` is constructed via `object.__new__` and
given a plain deterministic evaluator function directly, bypassing
`__init__`'s torch requirement (it has no `__slots__`, so this is a legitimate
minimal-construction technique, the same one this test needs to stay fast and
CI-friendly).

The bug this guards against: `RustMCTS.search()`'s `on_progress` parameter was
accepted and silently ignored -- the Rust core wrote only the FINAL "move"
telemetry record per move, so a dashboard watching the engine think showed the
PREVIOUS move's finished search for the whole next move's think time,
including a move that had already been played, still shown as a "candidate".
Fixed by slicing the search into short `continue_search()` calls between
which `on_progress` can be called safely. The Rust-side identity guarantee
(slicing changes nothing about the result) is proven separately in
`rust::tree::continue_search_tests`; this file is the Python orchestration
layer on top of it -- does it actually call back, does `done` move forward,
does an exception in the callback stay contained.

Usage:
    tests/verify_progress_slicing.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sumofish.rust_mcts import RustMCTS


def deterministic_evaluator(fens: list[str], actions: list[list[int]]):
    """Uniform priors, a fixed value -- exactly what the Rust unit tests'
    `FixedEval` does, so both sides of the FFI boundary are exercised by the
    same kind of evaluator this feature was designed around."""
    priors = [[1.0 / max(1, len(a))] * len(a) for a in actions]
    values = [0.5] * len(fens)
    return priors, values


def make_mcts(batch: int = 8, simulations: int = 100_000_000) -> RustMCTS:
    m = object.__new__(RustMCTS)
    m.simulations = simulations
    m._evaluate = deterministic_evaluator
    import sumofish_core as core

    m._core = core.Mcts(batch=batch, reuse=True)
    return m


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "ok" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))
    return ok


def main() -> int:
    results = []

    # The actual regression: on_progress must be called more than once during
    # a real (short but non-trivial) think, and `done` must move forward.
    mcts = make_mcts()
    board = chess.Board()
    calls: list[int] = []

    def on_progress(_root, done):
        calls.append(done)

    start = time.perf_counter()
    root, visits = mcts.search(
        board, deadline=start + 0.4, on_progress=on_progress, progress_every=0.05,
    )
    results.append(check(
        "on_progress is called more than once in a 0.4s search at 0.05s slices",
        len(calls) >= 2,
        f"got {len(calls)} calls: {calls}",
    ))
    results.append(check(
        "done is non-decreasing across calls",
        calls == sorted(calls),
        f"{calls}",
    ))
    results.append(check(
        "search still returns real visits",
        bool(visits) and sum(visits.values()) > 0,
        f"{len(visits)} moves, {sum(visits.values())} total visits",
    ))
    results.append(check(
        "the final visit total is at least the last reported `done`",
        sum(visits.values()) >= (calls[-1] if calls else 0),
    ))

    # Exception safety: search_engine's own telemetry emission must never be
    # able to cost the engine a move, mirroring sumofish/mcts.py's own
    # `report()`'s try/except around on_progress.
    mcts2 = make_mcts()
    board2 = chess.Board()

    def raising_progress(_root, _done):
        raise RuntimeError("telemetry backend is down")

    start2 = time.perf_counter()
    try:
        _root2, visits2 = mcts2.search(
            board2, deadline=start2 + 0.2, on_progress=raising_progress, progress_every=0.05,
        )
        results.append(check(
            "a raising on_progress does not propagate out of search()",
            bool(visits2) and sum(visits2.values()) > 0,
        ))
    except RuntimeError:
        results.append(check("a raising on_progress does not propagate out of search()", False))

    # No deadline, or no on_progress: must take the old, unsliced path (no
    # crash, sensible result) -- this is the fast path every OTHER caller
    # (scripts/match.py, smoke.py, acceptance.py) still uses. A small
    # simulation count here, deliberately: with deadline=None there is no
    # clock to bound the search at all, so unlike every other case in this
    # file `self.simulations` IS the only limiter, and the production value
    # (100M) is only ever safe paired with a deadline, which this case is
    # specifically testing the absence of.
    mcts3 = make_mcts(simulations=200)
    board3 = chess.Board()
    _root3, visits3 = mcts3.search(board3, deadline=None)
    results.append(check(
        "deadline=None (no clock) still returns real visits",
        bool(visits3) and sum(visits3.values()) > 0,
    ))

    mcts4 = make_mcts()
    board4 = chess.Board()
    _root4, visits4 = mcts4.search(board4, deadline=time.perf_counter() + 0.2, on_progress=None)
    results.append(check(
        "on_progress=None with a real deadline still returns real visits",
        bool(visits4) and sum(visits4.values()) > 0,
    ))

    if all(results):
        print(f"\nOK: {len(results)}/{len(results)} checks passed")
        return 0
    print(f"\nFAIL: {sum(results)}/{len(results)} checks passed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
