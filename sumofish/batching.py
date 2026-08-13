"""Share one forward pass across several concurrently-playing games.

`CHESSGPU_BATCH` already batches LEAVES from one game's own tree into a single
forward pass, but it is 1:1 with one `core.Mcts`, so N games running at once
issue N separate small forward passes. The engine is launch-bound well below
~128 rows (one row costs about what 128 costs), so those passes are mostly
waiting on kernel launch rather than arithmetic. `scripts/batch_payoff.py`
measured the end-to-end payoff at **2.45x games/hour** at concurrency 8.

# The reproducibility problem, and its actual mechanism

Batching across games changes **when** forward passes happen, never **what any
single game asks for or the order it asks in**. Each game still runs its own
`core.Mcts.search()` and still sees its own evaluations, in its own sequence.
So the question is only whether the NUMBERS come back the same.

**They do not, and until 2026-08-13 this docstring named the wrong reason.** It
said the grouping was decided by thread scheduling, so "different grouping,
different last-bit values, different games". `scripts/batch_invariance.py`
measured it against the real nets over 256 positions, and grouping is
irrelevant:

  1. **Row COUNT changes the output.** Against a one-row-per-pass reference the
     prior changes on ~250/256 positions once a pass carries 44 or more rows,
     and the argmax of the prior moves on 3/256. The value holds to one ULP
     until 128 rows, where it moves on 251/256 by up to 7.8e-04. Two kernel
     switches, at 40->44 and at 127->128.
  2. **WHICH positions share a pass does not matter at all.** Shuffling the
     partners at a fixed 64 rows gives bit-identical priors, 0/256 different,
     over three independent shuffles.
  3. **A repeated shape is bit-identical.**

The output is a deterministic function of the row count alone. That distinction
is the whole design, **because a row count is something you can fix and a thread
schedule is not.**

# What `fixed_rows` does

With `fixed_rows=N` every forward pass carries exactly N rows: the flush
concatenates its pending requests and chunks them into N-row passes, and the
final short chunk is padded up by the evaluator. The row count then stops
depending on how many games happened to have a request pending when the timer
fired, and a batched run becomes bit-reproducible against another batched run at
the same N.

**The evaluator MUST be built with a matching `pad_to`:**

    raw = make_evaluator(policy, value, pad_to=N)
    batcher = BatchedEvaluator(raw, active_games=8, fixed_rows=N)

Without that, the final chunk of each flush is short, its shape varies, and the
guarantee is silently gone: everything still runs and the numbers still look
reasonable. `short_passes` counts those, and `assert_reproducible()` raises if
any occurred, so the failure is loud where it matters. The padding is close to
free for the same reason batching pays at all, and `make_evaluator` repeats the
last row rather than zero-padding, because an all-zero token sequence is not a
legal position and produces NaNs that read as a model bug.

# What this still does NOT buy

**Comparability with the existing archive.** Every result in `runs/matches` was
measured by an unbatched search sending variable row counts, and no choice of
`fixed_rows` reproduces that. A batched run is comparable to other batched runs
at the same N, and to nothing else. Before any batched number is quoted beside
an archived one, that has to be said out loud, or the batched arm has to be
re-run unbatched.

# What this deliberately does NOT do

It does not schedule, prioritise or reorder games, and it does not try to keep
the batch full. A game that stalls is bounded by `max_wait` and nothing else;
the flusher fires on a timer regardless. Cleverness here buys little (the win is
launch overhead, which any reasonably-sized batch recovers) and costs the
determinism above, which is the expensive thing to get back.
"""

from __future__ import annotations

import threading
import time


class BatchedEvaluator:
    """One forward pass over the union of several games' pending requests.

    Drop-in for `RustMCTS._evaluate`: same `(fens, actions) -> (priors, values)`
    signature, so integration is an assignment and no search code changes.

        raw = make_evaluator(policy, value, pad_to=256)
        batcher = BatchedEvaluator(raw, active_games=8, fixed_rows=256)
        for mcts in per_game_mcts:
            mcts._evaluate = batcher.evaluate
        ...
        batcher.game_finished()   # exactly once per game, when it ends
        batcher.stop()
        batcher.assert_reproducible()
    """

    def __init__(self, raw_evaluate, active_games: int, max_wait: float = 0.02,
                 fixed_rows: int | None = None):
        if active_games < 1:
            raise ValueError(f"active_games must be >= 1, got {active_games}")
        if fixed_rows is not None and fixed_rows < 1:
            raise ValueError(f"fixed_rows must be >= 1, got {fixed_rows}")
        self._raw = raw_evaluate
        self.max_wait = max_wait
        self.fixed_rows = fixed_rows
        self._lock = threading.Lock()
        self._active = active_games
        self._pending: list[tuple[list[str], list[list[int]], dict]] = []
        self._stopped = False
        # Diagnostics. rows/passes is the number that justifies this existing:
        # at 1.0 it is doing nothing a per-game batcher was not already doing.
        self.rows_sent = 0
        self.forward_passes = 0
        # Passes that did NOT carry exactly `fixed_rows` rows. Every one of
        # these depends on the evaluator's `pad_to` to restore the shape, so a
        # nonzero count with no matching `pad_to` means the reproducibility
        # guarantee is gone. See `assert_reproducible`.
        self.short_passes = 0
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

    def assert_reproducible(self) -> None:
        """Raise if any forward pass had a shape the padding cannot have fixed.

        Call after a run whose numbers are going to be compared to another run.
        Silence here is the only evidence that `fixed_rows` and the evaluator's
        `pad_to` actually agreed; nothing else in the stack checks it, and a
        mismatch changes results without changing behaviour.
        """
        if self.fixed_rows is None:
            raise RuntimeError(
                "fixed_rows was not set, so row counts varied with thread "
                "scheduling and this run is not reproducible. See "
                "scripts/batch_invariance.py.")
        if self.short_passes:
            raise RuntimeError(
                f"{self.short_passes} of {self.forward_passes} forward passes "
                f"did not carry exactly {self.fixed_rows} rows. They are "
                f"reproducible only if the evaluator was built with "
                f"pad_to={self.fixed_rows}; if it was, this check cannot see it "
                f"and you may clear short_passes deliberately.")

    # -- the batch ---------------------------------------------------------

    def _flush_loop(self) -> None:
        while not self._stopped:
            time.sleep(self.max_wait)
            with self._lock:
                if self._pending and not self._stopped:
                    self._flush_locked()

    def _forward(self, fens: list[str], actions: list[list[int]]):
        """Issue the pass, in exact `fixed_rows` chunks when one is set."""
        if self.fixed_rows is None:
            self.rows_sent += len(fens)
            self.forward_passes += 1
            return self._raw(fens, actions)
        priors: list = []
        values: list = []
        for i in range(0, len(fens), self.fixed_rows):
            chunk_f = fens[i:i + self.fixed_rows]
            chunk_a = actions[i:i + self.fixed_rows]
            p, v = self._raw(chunk_f, chunk_a)
            # The evaluator pads UP to pad_to and returns only the real rows, so
            # a short chunk still comes back the right length. Slicing defensively
            # anyway: a padded evaluator that ever returned its padding rows would
            # otherwise misalign every span after it, silently.
            priors.extend(p[:len(chunk_f)])
            values.extend(v[:len(chunk_f)])
            self.rows_sent += len(chunk_f)
            self.forward_passes += 1
            if len(chunk_f) != self.fixed_rows:
                self.short_passes += 1
        return priors, values

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
            priors, values = self._forward(fens, actions)
        except BaseException as exc:  # noqa: BLE001 -- must reach every waiter
            # One bad batch must not strand N threads. Hand the exception to
            # each caller so it surfaces on the thread that can report which
            # game it belongs to.
            for _, _, holder in spans:
                holder["error"] = exc
                holder["event"].set()
            return
        for start, end, holder in spans:
            holder["result"] = (priors[start:end], values[start:end])
            holder["event"].set()

    def evaluate(self, fens: list[str], actions: list[list[int]]):
        """Submit and block until this caller's slice comes back."""
        holder: dict = {"event": threading.Event()}
        with self._lock:
            # The stopped check belongs INSIDE the lock. Outside it, this
            # interleaving strands the caller forever: it passes the check,
            # stop() then sets _stopped and drains an empty _pending, this
            # thread appends to the drained list, and with _active >= 2 the
            # fast path below does not fire. _flush_loop has already exited on
            # _stopped, so nothing will ever set the event and the wait() at
            # the bottom blocks for good. That reads as "the harness got slow
            # at hour six", which is the exact failure this class was written
            # to remove.
            if self._stopped:
                raise RuntimeError("evaluate() after stop()")
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
