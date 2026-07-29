"""Two views of the same board, and which one to believe.

Measured on a live game, thirteen consecutive plies: **the lichess public game
stream runs a median 8.7 seconds behind the engine's own view**, and very
consistently -- 8.4s to 9.1s. That is not jitter and it is not this program;
it is what the public spectator feed does. Whatever the reason (a delay on
watching games in progress is the obvious one), it cannot be optimised away
from this side.

So the stream is the wrong source for "what is the position right now". The
engine's telemetry is written the instant a search starts, is local, and
carries the FEN being searched -- which is the position *after* the opponent
moved. It is the freshest thing available by nearly nine seconds.

    engine telemetry   the position, now. Primary.
    lichess stream     clocks, the move list, who is playing, and the
                       authoritative record. Slower, and still needed.

`EngineBoard` keeps a board driven by the telemetry. The only thing the
telemetry does not say is *which move* produced each position, and that is
worth having for the last-move highlight, so consecutive records are bridged
by searching for the move or moves between them. Consecutive records differ by
one or two plies -- the opponent's move and ours -- so that search stays
shallow.

Anything further apart is a record from somewhere else: the bot plays two
games at once and both engines write to the same file. `EngineBoard` holds
those rather than adopting them, and the lichess stream -- late, but
authoritative about which positions this game has been in -- decides which are
ours. That is also what makes stale telemetry from a finished game harmless.
"""

from __future__ import annotations

from collections import deque

import chess

# Consecutive telemetry records are one or two plies apart: the opponent's
# reply, then ours. Beyond that we are not looking at the next position in the
# same game, so reset rather than guess.
MAX_BRIDGE_PLIES = 2


def _bridge(start: chess.Board, target_fen: str, depth: int) -> list[chess.Move] | None:
    """The move(s) from `start` to `target_fen`, or None if not within depth.

    Breadth is the legal move count, so one ply is ~40 boards and two ~1600.
    Bounded by `MAX_BRIDGE_PLIES`, so it cannot get expensive.
    """
    if start.board_fen() == target_fen:
        return []
    if depth == 0:
        return None
    for move in start.legal_moves:
        start.push(move)
        try:
            rest = _bridge(start, target_fen, depth - 1)
            if rest is not None:
                return [move] + rest
        finally:
            start.pop()
    return None


class EngineBoard:
    """The current position, as the engine sees it, for **one** game.

    Fed one telemetry record at a time. Holds a board and the move that last
    changed it, and nothing else: the stream owns the clocks and the move list
    because it is the only source that has them.

    ## Why this has to reject records rather than follow all of them

    `config/lichess-bot.yml` sets `concurrency: 2`, lichess-bot spawns one
    engine process per game, and every one of them appends to the same
    `logs/engine.jsonl`. So the file is not a stream, it is two streams
    interleaved, and nothing in a record said which game it belonged to.
    Measured over one log: 468 places where consecutive searches alternate
    between two positions, and stretches like ply 7 of one game, then ply 0 of
    another, then back.

    Read as one stream that produced: a search panel flipping between two
    games, an evaluation curve holding both games' plies at once -- which is
    what "the graph says one side is winning when it is not" was -- and a
    board picture that jumped to the other game, because a position too far
    away to bridge was treated as "the game moved on" and adopted.

    The fix is an anchor the other game cannot fake: the position history of
    the game the dashboard is actually showing, straight from the lichess
    stream. A record is ours if the stream has seen that position, or if it
    continues from the last position we accepted. Anything else belongs to the
    other board and is dropped.

    The stream runs ~9s late (see the module docstring), so confirmation
    arrives late -- which is why the continuation rule exists, and why an
    anchor that turns out to be wrong self-corrects within about nine seconds
    rather than sticking.
    """

    def __init__(self) -> None:
        self.board: chess.Board | None = None
        self.last: chess.Move | None = None
        # Records that could be ours but cannot be proved so yet, newest last.
        # The engine is ahead of the stream, so a position it is searching now
        # is one the stream will not describe for another nine seconds --
        # which means at the moment a record arrives there is often nothing to
        # confirm it against. Hold it; the confirmation is coming.
        self._orphans: deque = deque(maxlen=48)

    def reset(self) -> None:
        """Forget the chain. Called when the game on screen changes."""
        self.board, self.last = None, None
        self._orphans.clear()

    def reanchor(self, history: frozenset) -> list[dict]:
        """Adopt held records the stream has since confirmed. Returns them.

        This is what makes a wrong guess temporary. Two games in the same
        opening are in the same position for a few plies, so the chain can
        start out following the wrong one; and on a mid-game attach there is
        no chain at all and every record arrives unconfirmable. Both recover
        here, one stream-lag later, when a position the dashboard's own game
        has actually been in turns up among the held records.
        """
        if not history or not self._orphans:
            return []
        held = list(self._orphans)
        for i, rec in enumerate(held):
            placement = (rec.get("fen") or " ").split(" ")[0]
            if placement not in history:
                continue
            self.board, self.last = None, None
            self._orphans.clear()
            # Replay from the confirmed one: the rest bridge onto it in order,
            # and anything that does not belong is dropped again.
            return [r for r in held[i:] if self.update(r, history)]
        return []

    def update(self, record: dict, history: frozenset = frozenset()) -> bool:
        """Absorb a record. False means it belongs to another game.

        `history` is every position the streamed game has been in, as
        placement-only FENs. Empty means there is no game on screen to
        disagree with, and then every record is accepted as before.
        """
        fen = record.get("fen")
        if not fen:
            return False
        try:
            target = chess.Board(fen)
        except ValueError:
            return False

        bridge = None
        if self.board is not None:
            bridge = _bridge(self.board.copy(), target.board_fen(), MAX_BRIDGE_PLIES)

        if bridge is not None:
            # Continues the chain we are already following.
            for move in bridge:
                self.last = move
                self.board.push(move)
            return True

        if history and target.board_fen() not in history:
            # Too far from our game to be the next position in it, and the
            # game on screen has never been in this position. Either it is the
            # other board, or it is ours and the stream has not got there yet.
            # Hold it rather than deciding: `reanchor` sorts it out when the
            # stream catches up, and until then the board on screen stays
            # where it was instead of jumping to someone else's game.
            self._orphans.append(record)
            return False

        # Either the stream confirms this position is ours, or there is no
        # stream to ask. Start the chain here; the move that produced it is
        # not knowable from one record.
        self.board, self.last = target, None
        return True

        # A finished search has chosen, and that move is played as far as the
        # engine is concerned; the stream will confirm it in its own time.
        if record.get("done") and record.get("uci"):
            try:
                move = chess.Move.from_uci(record["uci"])
            except ValueError:
                return
            if move in self.board.legal_moves:
                self.board.push(move)
                self.last = move

    def snapshot(self) -> dict | None:
        if self.board is None:
            return None
        return {"board": self.board.copy(), "last": self.last,
                "ply": self.board.ply()}
