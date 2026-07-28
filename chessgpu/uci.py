"""Minimal UCI protocol loop.

The protocol handling lives here once. Engines supply a `chooser`: a callable
that takes a board plus the parsed `go` limits and returns a move. Phase 0 uses
a random chooser; the neural engine drops into the same slot unchanged.
"""

from __future__ import annotations

import sys
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


def _apply_position(tokens: list[str]) -> chess.Board:
    """Handle `position [startpos | fen <6 fields>] [moves ...]`."""
    if not tokens:
        return chess.Board()

    if tokens[0] == "startpos":
        board = chess.Board()
        rest = tokens[1:]
    elif tokens[0] == "fen":
        # A FEN is six space-separated fields, but be lenient: some GUIs omit
        # the trailing counters. Take everything up to `moves`.
        try:
            cut = tokens.index("moves")
        except ValueError:
            cut = len(tokens)
        board = chess.Board(" ".join(tokens[1:cut]))
        rest = tokens[cut:]
    else:
        return chess.Board()

    if rest and rest[0] == "moves":
        for uci in rest[1:]:
            try:
                board.push_uci(uci)
            except ValueError:
                # A move we can't parse means our state has diverged from the
                # GUI's. Nothing good comes from guessing past that point.
                break
    return board


def run(chooser: Chooser, name: str, author: str) -> None:
    board = chess.Board()

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
            limits = _parse_go(args)
            move = chooser(board=board, limits=limits)
            print(f"bestmove {move.uci()}", flush=True)

        elif cmd in ("quit", "stop"):
            if cmd == "quit":
                return

        elif cmd == "setoption":
            pass  # No options yet.
