"""Minimal UCI protocol loop.

The protocol handling lives here once. Engines supply a `chooser`: a callable
that takes a board plus the parsed `go` limits and returns a move. Phase 0 uses
a random chooser; the neural engine drops into the same slot unchanged.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from typing import Callable, Protocol

import chess


@dataclass
class Limits:
    """Parsed `go` arguments. All times in milliseconds. None means unset."""

    wtime: int | None = None
    btime: int | None = None
    winc: int = 0
    binc: int = 0
    movetime: int | None = None
    depth: int | None = None
    nodes: int | None = None
    movestogo: int | None = None
    infinite: bool = False

    def time_for(self, side: chess.Color) -> int | None:
        """Remaining clock for `side`, or None if the GUI didn't send one."""
        return self.wtime if side == chess.WHITE else self.btime

    def inc_for(self, side: chess.Color) -> int:
        return self.winc if side == chess.WHITE else self.binc


class Chooser(Protocol):
    def __call__(self, board: chess.Board, limits: Limits) -> chess.Move: ...


def _warn(msg: str) -> None:
    """Loud on stderr. lichess-bot surfaces engine stderr in its logs; a silent
    failure here is indistinguishable from a bad model."""
    print(f"[uci] {msg}", file=sys.stderr, flush=True)


def _parse_go(tokens: list[str]) -> Limits:
    limits = Limits()
    int_fields = {
        "wtime": "wtime",
        "btime": "btime",
        "winc": "winc",
        "binc": "binc",
        "movetime": "movetime",
        "depth": "depth",
        "nodes": "nodes",
        "movestogo": "movestogo",
    }
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "infinite":
            limits.infinite = True
            i += 1
        elif tok in int_fields and i + 1 < len(tokens):
            try:
                setattr(limits, int_fields[tok], int(tokens[i + 1]))
            except ValueError:
                pass
            i += 2
        else:
            i += 1
    return limits


def _apply_position(tokens: list[str]) -> chess.Board | None:
    """Handle `position [startpos | fen <6 fields>] [moves ...]`.

    Returns None if the position could not be reconstructed faithfully.

    Silently substituting *a* board for *the* board is the worst failure mode
    available here: the engine happily answers for a position the GUI is not
    in, lichess rejects the move, the game is lost, and nothing is logged. So
    every divergence is loud on stderr and returns None, and the caller
    refuses to move rather than guessing.
    """
    if not tokens:
        _warn("empty `position` command")
        return None

    if tokens[0] == "startpos":
        board = chess.Board()
        rest = tokens[1:]
    elif tokens[0] == "fen":
        try:
            cut = tokens.index("moves")
        except ValueError:
            cut = len(tokens)
        fen = " ".join(tokens[1:cut])
        try:
            board = chess.Board(fen)
        except ValueError as e:
            _warn(f"unparseable FEN {fen!r}: {e}")
            return None
        rest = tokens[cut:]
    else:
        _warn(f"unrecognised `position` form: {tokens[0]!r}")
        return None

    if rest and rest[0] == "moves":
        for i, uci in enumerate(rest[1:]):
            try:
                board.push_uci(uci)
            except ValueError as e:
                _warn(f"illegal move {uci!r} at index {i} of the move list: {e}")
                return None
    elif rest:
        _warn(f"expected `moves`, got {rest[0]!r}")
        return None
    return board


def run(chooser: Chooser, name: str, author: str) -> None:
    board: chess.Board | None = chess.Board()

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        tokens = line.split()
        cmd, args = tokens[0], tokens[1:]

        if cmd == "uci":
            print(f"id name {name}", flush=True)
            print(f"id author {author}", flush=True)
            print("uciok", flush=True)

        elif cmd == "isready":
            print("readyok", flush=True)

        elif cmd == "ucinewgame":
            board = chess.Board()

        elif cmd == "position":
            board = _apply_position(args)

        elif cmd == "go":
            if board is None:
                # We do not know the position. Answering would mean sending a
                # move for a board the GUI is not on.
                _warn("`go` with no valid position; refusing to move")
                continue
            limits = _parse_go(args)
            try:
                move = chooser(board=board, limits=limits)
            except Exception:
                # Anything at all: CUDA OOM, a driver reset, a tokenizer edge
                # case. Dying here forfeits the game; a bad legal move does
                # not. The engine must outlive its own bugs.
                _warn("chooser raised, falling back to the first legal move")
                traceback.print_exc(file=sys.stderr)
                legal = list(board.legal_moves)
                if not legal:
                    _warn("no legal moves; nothing to send")
                    continue
                move = legal[0]
            print(f"bestmove {move.uci()}", flush=True)

        elif cmd in ("quit", "stop"):
            if cmd == "quit":
                return

        elif cmd == "setoption":
            pass  # No options yet.
