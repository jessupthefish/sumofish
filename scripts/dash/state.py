"""Everything the dashboard knows, and how fresh each part of it is.

The old watch loop had one bug shape repeated three times: it fetched, and if
the fetch failed it kept the previous value and drew it exactly as if it were
new. A dead connection rendered pixel-identical to a live one. You could stare
at a frozen board for twenty minutes and have no way to tell.

So staleness is not a display decision here, it is part of the data model. You
cannot read a value out of this module without also being handed how old it is
and whether its source is currently healthy. `Field.track` collapses that into
the three states a plot display would use:

    LIVE     updated within its expected interval
    COAST    late, so this is the last known value and it may have moved
    LOST     the source is erroring or has been silent for a long multiple

The render layer colours by track and prints the age. Nothing else is required
of it, and nothing can bypass it.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field

LIVE, COAST, LOST = "live", "coast", "lost"


class Smooth:
    """A number that moves toward its target instead of jumping to it.

    Exponential approach rather than a fixed step, so a big swing starts fast
    and settles gently, and the motion takes the same time whatever the size
    of the change. `tau` is the time constant: roughly, how long to cover two
    thirds of the remaining distance.

    Framerate independent on purpose -- it eases against the wall clock, not
    against a frame count -- so it looks the same whether the loop is keeping
    up or not.
    """

    def __init__(self, tau: float = 0.35) -> None:
        self.tau = tau
        self.value: float | None = None
        self._last = 0.0

    def advance(self, target: float | None) -> float | None:
        now = time.monotonic()
        dt, self._last = (now - self._last if self._last else 0.0), now
        if target is None:
            self.value = None
            return None
        if self.value is None or dt <= 0:
            self.value = target
            return self.value
        # 1 - e^(-dt/tau): the fraction of the remaining gap to close this
        # frame, for however long this frame happened to take.
        self.value += (target - self.value) * (1.0 - math.exp(-dt / self.tau))
        if abs(target - self.value) < 5e-4:
            self.value = target
        return self.value


@dataclass
class Field:
    """One value, plus the provenance needed to render it honestly."""

    name: str
    interval: float                 # how often a healthy source updates this
    value: object = None
    updated: float = 0.0            # wall clock of the last successful update
    error: str | None = None        # last failure, cleared on success
    fills: int = 0                  # successful updates, ever

    def set(self, value) -> None:
        self.value = value
        self.updated = time.time()
        self.error = None
        self.fills += 1

    def fail(self, why: str) -> None:
        self.error = why

    @property
    def age(self) -> float:
        return float("inf") if not self.updated else time.time() - self.updated

    @property
    def track(self) -> str:
        """Is this number being fed, coasting, or gone."""
        if self.fills == 0:
            return LOST
        age = self.age
        if self.error and age > self.interval * 2:
            return LOST
        if age > self.interval * 6:
            return LOST
        if age > self.interval * 2:
            return COAST
        return LIVE


def ago(seconds: float) -> str:
    if seconds == float("inf"):
        return "never"
    if seconds < 90:
        return f"{int(seconds)}s"
    if seconds < 5400:
        return f"{int(seconds / 60)}m"
    return f"{int(seconds / 3600)}h"


@dataclass
class Tape:
    """A rolling log, so the view has memory past the current frame.

    Without one, the only answer to "what happened while I was looking away"
    is a W/D/L counter that already forgot. Bounded on purpose: this is a
    display, not storage, and the PGN and the jsonl files are the record.
    """

    limit: int = 400
    lines: deque = field(default_factory=lambda: deque(maxlen=400))

    def add(self, kind: str, text: str) -> None:
        self.lines.append((time.time(), kind, text))

    def tail(self, n: int) -> list[tuple[float, str, str]]:
        return list(self.lines)[-n:]


class State:
    """The single object every source writes and the renderer reads.

    One lock, held only for the assignment. Sources do their I/O outside it,
    so a lichess request that takes eight seconds to time out never holds the
    renderer up -- which is the other half of the old design's problem, where
    the fetch happened inline in the draw loop.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # One event per source, so a source that learns something can nudge
        # another out of its sleep instead of leaving it to find out on its own
        # schedule. Used when a game ends: the stream knows immediately, the
        # poller would otherwise take its full interval to catch up.
        self._wake: dict[str, threading.Event] = {}
        self.tape = Tape()
        self.started = time.time()
        # ply -> P(White wins), for the game in progress. Kept in White's
        # frame, never the side-to-move frame the search works in, because a
        # curve of alternating perspectives is a sawtooth rather than a trend.
        self.curve: dict[int, float] = {}
        # The evaluation gauge's animated position, advanced once per frame.
        self.eval_smooth = Smooth()
        self.fields: dict[str, Field] = {}
        for name, interval in (
            # These are "how often a healthy source updates this", which is
            # what `Field.track` grades staleness against. They have to track
            # the real cadences or the live/coast/lost tag lies.
            ("profile", 45.0),      # polled every 45s
            ("playing", 6.0),       # polled every 2s
            # Pushed per move, so silence is normal and not a fault: the gap
            # between moves is a player thinking. Generous enough not to cry
            # wolf during a long think at the slowest time control this bot
            # accepts, tight enough that a genuinely dead stream still shows.
            ("game", 90.0),
            # Written at ~6Hz *while searching* and not at all between moves,
            # so the gap is however long the opponent takes.
            ("engine", 30.0),
            ("engine_board", 30.0),   # the position, from that same feed
            ("train", 20.0),        # training run log
            ("rating_log", 120.0),  # local rating samples with deploy markers
            ("gpu", 3.0),
            ("units", 8.0),
            ("finished", 3600.0),   # summary of the last completed game
        ):
            self.fields[name] = Field(name, interval)

    # ---- access ----------------------------------------------------------

    def set(self, name: str, value) -> None:
        with self._lock:
            self.fields[name].set(value)

    def fail(self, name: str, why: str) -> None:
        with self._lock:
            self.fields[name].fail(why)

    def get(self, name: str, default=None):
        with self._lock:
            v = self.fields[name].value
            return default if v is None else v

    def field(self, name: str) -> Field:
        with self._lock:
            return self.fields[name]

    def waker(self, name: str) -> threading.Event:
        with self._lock:
            return self._wake.setdefault(name, threading.Event())

    def wake(self, name: str) -> None:
        self.waker(name).set()

    def note(self, kind: str, text: str) -> None:
        with self._lock:
            self.tape.add(kind, text)

    def record_eval(self, ply: int, wp_white: float) -> None:
        """Add a point to the game's evaluation curve.

        A ply lower than what we already hold means a new game started, so the
        curve resets. Without that check the next game's opening would be
        plotted on top of the last game's endgame.
        """
        with self._lock:
            if self.curve and ply < max(self.curve):
                self.curve = {}
            self.curve[ply] = wp_white

    def curve_series(self) -> list[float]:
        with self._lock:
            return [self.curve[p] for p in sorted(self.curve)]

    @property
    def health(self) -> str:
        """Worst track across the sources that matter for watching a game."""
        ranks = {LIVE: 0, COAST: 1, LOST: 2}
        worst = LIVE
        for name in ("profile", "playing", "engine"):
            t = self.fields[name].track
            if ranks[t] > ranks[worst]:
                worst = t
        return worst
