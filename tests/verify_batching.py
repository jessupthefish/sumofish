#!/usr/bin/env python
"""Batching must be invisible to the search. No GPU required.

`sumofish/batching.py` exists to make matches faster. The only reason it is
allowed near the lab is that it cannot change what a game plays: it groups
forward passes, it does not alter what any game asks for or the order it asks
in. A speedup that changed the games would make every future measurement
incomparable with the archive, which costs far more than the time it saves.

So that claim gets asserted rather than assumed, against a fake evaluator whose
output is a pure function of its input. Batched and unbatched must agree
exactly, row for row, no matter how the rows were grouped.

Also covered, because both are hangs rather than errors in the prototype and a
harness that hangs at hour six of a 2000-game match reads as slow progress:
  * an exception in the forward pass must reach every waiter;
  * stop() with work pending must wake every waiter.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sumofish.batching import BatchedEvaluator  # noqa: E402


def fake_raw(fens, actions):
    """Deterministic pure function of each row, independent of batching."""
    priors = [[len(f) * 1.0 + i for i in a] for f, a in zip(fens, actions)]
    values = [float(sum(ord(c) for c in f) % 1000) for f in fens]
    return priors, values


def main() -> int:
    failures: list[str] = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    rows = [(f"fen-{i}" + "x" * (i % 7), [i, i + 1, i + 2]) for i in range(48)]
    want_p, want_v = fake_raw([f for f, _ in rows], [a for _, a in rows])

    # -- 1. Batched across 8 threads must equal one unbatched call, exactly.
    for concurrency in (1, 4, 8):
        got: dict[int, tuple] = {}
        b = BatchedEvaluator(fake_raw, active_games=concurrency, max_wait=0.005)
        errs: list[BaseException] = []

        def worker(idx: int) -> None:
            try:
                f, a = rows[idx]
                p, v = b.evaluate([f], [a])
                got[idx] = (p[0], v[0])
            except BaseException as exc:  # noqa: BLE001
                errs.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(rows))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        check(not any(t.is_alive() for t in threads),
              f"concurrency={concurrency}: a worker never returned (deadlock)")
        check(not errs, f"concurrency={concurrency}: workers raised {errs[:1]}")
        check(len(got) == len(rows),
              f"concurrency={concurrency}: {len(got)}/{len(rows)} rows came back")
        for i in got:
            check(got[i][0] == want_p[i] and got[i][1] == want_v[i],
                  f"concurrency={concurrency} row {i}: batching CHANGED the "
                  f"evaluation, {got[i]} vs {(want_p[i], want_v[i])}")
        check(b.forward_passes <= len(rows),
              "more forward passes than rows, which is not batching")
        b.stop()

    # -- 2. A failing forward pass must reach every waiter, not hang them.
    def boom(fens, actions):
        raise ValueError("forward pass exploded")

    b = BatchedEvaluator(boom, active_games=4, max_wait=0.005)
    seen: list[str] = []

    def failer() -> None:
        try:
            b.evaluate(["f"], [[1]])
            seen.append("returned-normally")
        except ValueError:
            seen.append("raised")
        except BaseException:  # noqa: BLE001
            seen.append("raised-wrong-type")

    ts = [threading.Thread(target=failer) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=15)
    check(not any(t.is_alive() for t in ts),
          "a waiter hung when the forward pass raised")
    check(seen.count("raised") == 4, f"expected 4 raises, got {seen}")
    b.stop()

    # -- 3. stop() with work pending must wake waiters rather than strand them.
    started = threading.Event()

    def slow(fens, actions):
        started.set()
        threading.Event().wait(3)
        return [[0.0]], [0.0]

    b = BatchedEvaluator(slow, active_games=99, max_wait=5.0)
    out: list[str] = []

    def waiter() -> None:
        try:
            b.evaluate(["f"], [[1]])
            out.append("returned")
        except BaseException:  # noqa: BLE001
            out.append("raised")

    t = threading.Thread(target=waiter)
    t.start()
    threading.Event().wait(0.2)
    b.stop()
    t.join(timeout=15)
    check(not t.is_alive(), "stop() left a waiter blocked forever")
    check(out == ["raised"], f"stop() should fail pending waiters, got {out}")

    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("verify_batching: OK (batching does not change evaluations; "
          "errors and stop() reach every waiter)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
