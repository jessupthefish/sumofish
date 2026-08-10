"""Share one forward pass across several concurrently-playing games.

`CHESSGPU_BATCH` already batches LEAVES from one game's own tree into a single
forward pass, but it is 1:1 with one `core.Mcts`, so N games running at once
issue N separate small forward passes. The engine is launch-bound well below
~128 rows (one row costs about what 128 costs), so those passes are mostly
waiting on kernel launch rather than arithmetic. `scripts/batch_payoff.py`
measured the end-to-end payoff at **2.45x games/hour** at concurrency 8.

This is that prototype's `CrossGameBatcher`, made production-worthy: exceptions
propagate instead of deadlocking, shutdown wakes every waiter, and the active
count cannot drift.

# The property that makes this safe for MEASUREMENT, not just for speed

Batching across games changes **when** forward passes happen, never **what any
single game asks for or the order it asks in**. Each game still runs its own
`core.Mcts.search()` and still sees its own evaluations, in its own sequence,
with the same values. Grouping the passes is invisible to the tree.

So a batched match must produce **games identical to a serial one** for the
same seed. That is a testable claim rather than a hope, and it is the whole
licence for using this in the lab: a speedup that changed the games would make
every measurement incomparable with the archive. `tests/verify_batching.py`
asserts it on real evaluations. If that test ever fails, the batcher is wrong
and the throughput is worthless, in that order.

# What this deliberately does NOT do

It does not schedule, prioritise or reorder games, and it does not try to keep
the batch full. A game that stalls is bounded by `max_wait` and nothing else;
the flusher fires on a timer regardless. Cleverness here buys little (the win
is launch overhead, which any reasonably-sized batch recovers) and costs the
determinism property above, which is the expensive thing to get back.
"""

from __future__ import annotations

import threading
import time


class BatchedEvaluator:
    """One forward pass over the union of several games' pending requests.

    Drop-in for `RustMCTS._evaluate`: same `(fens, actions) -> (priors, values)`
    signature, so integration is an assignment and no search code changes.

        batcher = BatchedEvaluator(raw_evaluate, active_games=8)
        for mcts in per_game_mcts:
            mcts._evaluate = batcher.evaluate
        ...
        batcher.game_finished()   # exactly once per game, when it ends
        batcher.stop()
    """

    def __init__(self, raw_evaluate, active_games: int, max_wait: float = 0.02):
        if active_games < 1:
            raise ValueError(f"active_games must be >= 1, got {active_games}")
        self._raw = raw_evaluate
        self.max_wait = max_wait
        self._lock = threading.Lock()
        self._active = active_games
        self._pending: list[tuple[list[str], list[list[int]], dict]] = []
        self._stopped = False
        # Diagnostics. rows/passes is the number that justifies this existing:
        # at 1.0 it is doing nothing a per-game batcher was not already doing.
        self.rows_sent = 0
        self.forward_passes = 0
        self._flusher = threading.Thread(target=self._flush_loop, daemon=True,
                                         name="batched-evaluator")
        self._flusher.start()

    # -- lifecycle ---------------------------------------------------------

    def game_finished(self) -> None:
        """One game has ended. Call exactly once per game.

        Without this the flush-when-everyone-has-asked fast path waits for
        requests from games that will never make another one, and every
        remaining game falls back to the `max_wait` timer. That still
        terminates, which is why the prototype's drift was invisible; it just
        quietly gives back most of the speedup near the end of a match.
        """
        with self._lock:
            self._active = max(0, self._active - 1)
            if self._pending and len(self._pending) >= max(self._active, 1):
                self._flush_locked()

    def stop(self) -> None:
        """Stop the flusher and fail any waiter still blocked.

        A waiter blocked on an Event that nobody will ever set is a hang, and a
        harness that hangs at the end of a 2000-game match is worse than one
        that crashes, because it looks like slow progress.
        """
        with self._lock:
            self._stopped = True
            for _, _, holder in self._pending:
                holder["error"] = RuntimeError("evaluator stopped with work pending")
                holder["event"].set()
            self._pending = []

    # -- the batch ---------------------------------------------------------

    def _flush_loop(self) -> None:
        while not self._stopped:
            time.sleep(self.max_wait)
            with self._lock:
                if self._pending and not self._stopped:
                    self._flush_locked()

    def _flush_locked(self) -> None:
        """Caller holds `self._lock`."""
        batch, self._pending = self._pending, []
        fens: list[str] = []
        actions: list[list[int]] = []
        spans = []
        for f, a, holder in batch:
            start = len(fens)
            fens.extend(f)
            actions.extend(a)
            spans.append((start, len(fens), holder))
        try:
            priors, values = self._raw(fens, actions)
        except BaseException as exc:  # noqa: BLE001 -- must reach every waiter
            # One bad batch must not strand N threads. Hand the exception to
            # each caller so it surfaces on the thread that can report which
            # game it belongs to.
            for _, _, holder in spans:
                holder["error"] = exc
                holder["event"].set()
            return
        self.rows_sent += len(fens)
        self.forward_passes += 1
        for start, end, holder in spans:
            holder["result"] = (priors[start:end], values[start:end])
            holder["event"].set()

    def evaluate(self, fens: list[str], actions: list[list[int]]):
        """Submit and block until this caller's slice comes back."""
        if self._stopped:
            raise RuntimeError("evaluate() after stop()")
        holder: dict = {"event": threading.Event()}
        with self._lock:
            self._pending.append((fens, actions, holder))
            # Fast path: everyone still playing has asked, so waiting on the
            # timer would only add latency.
            if len(self._pending) >= max(self._active, 1):
                self._flush_locked()
        holder["event"].wait()
        if "error" in holder:
            raise holder["error"]
        return holder["result"]

    @property
    def rows_per_pass(self) -> float:
        return self.rows_sent / self.forward_passes if self.forward_passes else 0.0
