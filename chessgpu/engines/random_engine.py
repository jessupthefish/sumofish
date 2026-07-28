"""Phase 0 engine: picks a uniformly random legal move.

Exists to prove the lichess deployment path end to end before any model does.
When it loses every game, that is the expected result and not a bug.
"""

from __future__ import annotations

import random
import sys

import chess

from chessgpu.uci import Limits, run


def choose(board: chess.Board, limits: Limits) -> chess.Move:
    legal = list(board.legal_moves)
    if not legal:
        # UCI has no clean way to say "no moves"; the GUI should never ask.
        return chess.Move.null()
    return random.choice(legal)


def main() -> None:
    run(choose, name="chess-gpu random", author="Jessupthefish")


if __name__ == "__main__":
    sys.exit(main())
