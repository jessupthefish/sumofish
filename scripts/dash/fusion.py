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
shallow. Anything further apart is treated as a new game rather than guessed
at, which is also what makes stale telemetry from a finished game harmless.
"""

from __future__ import annotations

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
    """The current position, as the engine sees it.

    Fed one telemetry record at a time. Holds a board and the move that last
    changed it, and nothing else: the stream owns the clocks and the move list
    because it is the only source that has them.
    """

    def __init__(self) -> None:
        self.board: chess.Board | None = None
        self.last: chess.Move | None = None

    def update(self, record: dict) -> None:
        fen = record.get("fen")
        if not fen:
            return
        try:
            target = chess.Board(fen)
        except ValueError:
            return

        bridge = None
        if self.board is not None:
            bridge = _bridge(self.board.copy(), target.board_fen(), MAX_BRIDGE_PLIES)

        if bridge is None:
            # First record, or a jump too large to be the next position in
            # this game. Start again from what we were told, and admit we do
            # not know what move produced it.
            self.board, self.last = target, None
        else:
            for move in bridge:
                self.last = move
                self.board.push(move)

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
