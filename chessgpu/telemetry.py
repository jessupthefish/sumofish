"""The engine's narration channel.

The search knows things nobody can see: which moves it is actually considering,
how the evaluation is moving, how many positions a second it is getting through,
how much of its time budget is gone. All of that lives inside `MCTS` for the
duration of one move and is then discarded. This module is how it gets out.

## Why an append-only file and not a socket or a fifo

Because the engine must never be able to block on whether anyone is watching.

`open(fifo, "w")` blocks until a reader attaches. Put that inside `choose()` and
a dashboard that happens not to be running turns into an unbounded stall on a
running chess clock. A socket has the mirror problem: it needs a connected peer
or a buffer that can fill, and either way the failure mode is the engine
waiting on a spectator.

An append-only file has neither property. `write()` to a local file with a
reader that may or may not exist is the same operation either way, and a tailer
that dies loses nothing that is not still on disk. The repo already does this
twice (`logs/rating.jsonl`, `runs/*/log.jsonl`), so it is also the format that
needs no explaining.

## Why a thread

Even a non-blocking write is a syscall on the critical path, and the critical
path here is measured against a clock that can lose a game. `Telemetry.emit()`
puts a dict on a bounded queue and returns; a daemon thread does the encoding
and the I/O. When the queue is full it drops the oldest progress record rather
than blocking, because a dropped frame of a 6 Hz progress feed costs nothing and
a stalled search costs the game.

Records that matter (`move`, `game`, `boot`) are marked durable and are never
dropped -- the queue is sized so that could only happen if the disk stopped
answering entirely, at which point the drop is the least of it.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path

# Records are small (a PV, five candidate moves, a dozen scalars: ~400 bytes).
# 512 of them is a couple of hundred KB of slack, which is far more than a
# healthy writer ever accumulates and still bounded if the disk hangs.
QUEUE_DEPTH = 512

# Roll over at 16 MB. A busy game writes ~60 KB, so this is ~250 games of
# history before the previous file is replaced.
MAX_BYTES = 16 * 1024 * 1024


class Telemetry:
    """Fire-and-forget JSON-lines writer. Never raises into the caller."""

    def __init__(self, path: str | os.PathLike | None) -> None:
        self.path = Path(path) if path else None
        self.dropped = 0
        self._q: queue.Queue[tuple[bool, dict]] = queue.Queue(maxsize=QUEUE_DEPTH)
        self._fh = None
        self._thread = None
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_if_large()
            self._fh = self.path.open("a", buffering=1)   # line buffered
        except OSError:
            # No telemetry is an acceptable outcome. A dead engine is not.
            self.path = None
            return
        self._thread = threading.Thread(
            target=self._drain, name="telemetry", daemon=True
        )
        self._thread.start()

    @property
    def enabled(self) -> bool:
        """False when there is nowhere to write, so callers can skip the work
        of *producing* a record as well as the work of writing it. Set
        `CHESSGPU_TELEMETRY=` (empty) to turn narration off entirely."""
        return self.path is not None

    # ---- producer side, called from the search thread --------------------

    def emit(self, record: dict, durable: bool = False) -> None:
        """Queue a record. Returns immediately, always."""
        if self.path is None:
            return
        record.setdefault("t", time.time())
        try:
            self._q.put_nowait((durable, record))
        except queue.Full:
            if durable:
                # Make room by discarding one droppable record, then retry once.
                try:
                    self._q.get_nowait()
                    self._q.put_nowait((durable, record))
                    self.dropped += 1
                    return
                except (queue.Empty, queue.Full):
                    pass
            self.dropped += 1

    # ---- consumer side, its own thread -----------------------------------

    def _drain(self) -> None:
        while True:
            try:
                _durable, record = self._q.get()
                line = json.dumps(record, separators=(",", ":"))
                self._fh.write(line + "\n")
            except Exception:
                # A broken telemetry thread must not take the engine with it.
                # Sleep so a persistent failure cannot become a hot loop.
                time.sleep(1.0)

    def _rotate_if_large(self) -> None:
        try:
            if self.path.exists() and self.path.stat().st_size > MAX_BYTES:
                self.path.replace(self.path.with_suffix(self.path.suffix + ".1"))
        except OSError:
            pass


def perspective(win_prob: float, white_to_move: bool) -> float:
    """Convert a side-to-move-relative probability to White's frame.

    Every value in `mcts.py` is from the perspective of whoever is about to
    move at that node. Charting those raw across a game gives a sawtooth that
    flips sign every ply and means nothing. Anything that persists or plots a
    value converts here first, and the record carries both numbers so no
    downstream reader has to remember which one it is holding.
    """
    return win_prob if white_to_move else 1.0 - win_prob
