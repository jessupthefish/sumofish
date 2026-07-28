"""Puzzle evaluation. This is the correctness gate for the whole project.

Protocol is a faithful port of `searchless_chess/src/puzzles.py`, including the
non-obvious rule: when the model plays something other than the expected move,
the puzzle still counts as solved if that move is checkmate. Lichess puzzles
sometimes admit a faster mate than the recorded line.

Board starts at the end of the puzzle's PGN. Even-indexed moves in `Moves` are
the opponent's and get pushed; odd-indexed moves are ours and get predicted.

Reference numbers from the paper, so a port that is silently wrong is visible.
Note WHICH prediction target each belongs to, because they are not comparable:

    action-value       83.3%      <- the headline 88.9%/2054/2895 lineage
    state-value        77.5%
    behavioral cloning 65.7%      <- what SumoFish currently trains

Those are the paper's own Table 2 ablation at fixed architecture. Comparing a
behavioral-cloning model against the action-value number is an apples-to-oranges
error that makes a healthy run look broken; it was made here once already.

Scale matters as much as target: the paper trains 20M steps at batch 4096, i.e.
81.9B positions. A 300k-step run at batch 1024 is 307M, or 1/267th of that.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn
import pandas as pd


@dataclass
class PuzzleResult:
    solved: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.solved / self.total if self.total else 0.0

    def __str__(self) -> str:
        return f"{self.solved}/{self.total} = {self.accuracy:.4f}"


def load_puzzles(path: str | Path, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=limit)
    return df


def _starting_board(pgn: str) -> chess.Board:
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        raise ValueError(f"could not parse PGN: {pgn[:80]}")
    return game.end().board()


class _PuzzleState:
    """Tracks one puzzle as it is stepped forward in lockstep with others."""

    __slots__ = ("board", "moves", "idx", "done", "solved")

    def __init__(self, board: chess.Board, moves: list[str]) -> None:
        self.board = board
        self.moves = moves
        self.idx = 0
        self.done = False
        self.solved = False
        self._advance()

    def _advance(self) -> None:
        """Push opponent moves until it is our turn, or the puzzle is over."""
        while self.idx < len(self.moves) and self.idx % 2 == 0:
            self.board.push(chess.Move.from_uci(self.moves[self.idx]))
            self.idx += 1
        if self.idx >= len(self.moves):
            self.done, self.solved = True, True

    def submit(self, predicted: chess.Move) -> None:
        expected = self.moves[self.idx]
        if predicted.uci() != expected:
            # Upstream's rule: a deviation is forgiven only if it mates.
            self.board.push(predicted)
            self.done = True
            self.solved = self.board.is_checkmate()
            return
        self.board.push(chess.Move.from_uci(expected))
        self.idx += 1
        self._advance()


def evaluate_puzzles(policy, puzzles: pd.DataFrame, batch_size: int = 512) -> PuzzleResult:
    """Run `policy` over `puzzles`, batching forward passes across puzzles.

    Puzzles have different lengths, so they are stepped in lockstep and drop out
    as they finish. This keeps the GPU busy; evaluating one position at a time
    is roughly an order of magnitude slower and the metric is identical.
    """
    solved = 0
    total = 0

    rows = list(puzzles.itertuples(index=False))
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        states: list[_PuzzleState] = []
        for row in chunk:
            try:
                states.append(_PuzzleState(_starting_board(row.PGN), row.Moves.split(" ")))
            except (ValueError, AssertionError):
                continue  # Malformed row; excluded from the denominator.

        total += len(states)
        while True:
            active = [s for s in states if not s.done]
            if not active:
                break
            predictions = policy.play_batch([s.board for s in active])
            for state, move in zip(active, predictions, strict=True):
                state.submit(move)

        solved += sum(s.solved for s in states)

    return PuzzleResult(solved=solved, total=total)
