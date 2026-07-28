"""Choosing moves with a value head instead of a policy head.

This is a different kind of engine from `policy.py`, and the difference is the
whole point of the state-value phase.

The behavioral-cloning model answers "what move would Stockfish play here?" in
one forward pass. It never forms an opinion about the position. It cannot tell
you whether it is winning, cannot offer a draw, cannot resign, and cannot be
searched, because there is nothing to search *toward*.

The value model answers "how good is this position for the side to move?" To
pick a move you make each legal move and ask the question about the resulting
position, then take the move that leaves the opponent worst off. That is still
zero search -- one ply of make-move-and-evaluate is not a tree -- but it is a
real evaluation function, and everything later is built on it: MCTS, calibrated
difficulty by picking a move a known amount worse than best, resignation,
draw offers, and an engine that can say how sure it is.

Cost: ~30 legal moves per position, batched into one forward pass. Measured at
about 5 ms, against 3.7 ms for the single-position policy net. Effectively
free.

The sign convention is the one people get wrong: the model predicts the win
probability of *the side to move*. After we make our move it is the opponent's
turn, so the returned value is THEIR win probability, and we want to MINIMISE
it.
"""

from __future__ import annotations

import chess
import numpy as np
import torch

from chessgpu.hlgauss import HLGauss
from chessgpu.model import ChessTransformer
from chessgpu.tokenizer import tokenize


class ValuePolicy:
    """Picks moves by evaluating the position after each legal move."""

    def __init__(
        self,
        model: ChessTransformer,
        hlgauss: HLGauss,
        device: str = "cuda:0",
        temperature: float = 0.0,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model = model.to(device).eval()
        self.hl = hlgauss.to(device)
        self.device = device
        self.temperature = temperature
        self.dtype = dtype

    @torch.inference_mode()
    def evaluate(self, boards: list[chess.Board]) -> tuple[np.ndarray, np.ndarray]:
        """Win probability for the side to move, and the model's uncertainty."""
        tokens = np.stack([tokenize(b.fen()) for b in boards])
        x = torch.from_numpy(tokens).long().to(self.device, non_blocking=True)
        with torch.autocast(self.device.split(":")[0], dtype=self.dtype):
            logits = self.model(x).float()
        return (
            self.hl.expectation(logits).cpu().numpy(),
            self.hl.confidence(logits).cpu().numpy(),
        )

    def value_of(self, board: chess.Board) -> float:
        """Win probability for the side to move in `board`."""
        return float(self.evaluate([board])[0][0])

    def rank_moves(self, board: chess.Board) -> list[tuple[chess.Move, float]]:
        """Every legal move with OUR win probability after playing it, best first.

        Terminal positions are scored exactly rather than asked of the model:
        the net has never been trained to recognise that a game is over, and
        guessing at checkmate when it is a lookup would be silly.
        """
        legal = list(board.legal_moves)
        if not legal:
            return []

        scores: dict[chess.Move, float] = {}
        to_evaluate: list[chess.Move] = []
        children: list[chess.Board] = []

        for mv in legal:
            board.push(mv)
            outcome = board.outcome(claim_draw=True)
            if outcome is not None:
                # Scored from our side: we just moved, so a checkmate here is
                # ours. A draw is 0.5 regardless of who claims it.
                scores[mv] = 1.0 if outcome.winner is not None else 0.5
            else:
                to_evaluate.append(mv)
                children.append(board.copy(stack=False))
            board.pop()

        if children:
            # The model returns the CHILD side-to-move's win probability, i.e.
            # the opponent's. Ours is one minus that.
            values, _ = self.evaluate(children)
            for mv, v in zip(to_evaluate, values, strict=True):
                scores[mv] = 1.0 - float(v)

        return sorted(scores.items(), key=lambda kv: -kv[1])

    def play(self, board: chess.Board) -> chess.Move:
        ranked = self.rank_moves(board)
        if not ranked:
            raise ValueError(f"no legal moves in {board.fen()}")
        if self.temperature <= 0:
            return ranked[0][0]
        moves = [m for m, _ in ranked]
        vals = np.array([v for _, v in ranked], dtype=np.float64)
        p = np.exp((vals - vals.max()) / self.temperature)
        p /= p.sum()
        return moves[int(np.random.choice(len(moves), p=p))]

    def play_batch(self, boards: list[chess.Board]) -> list[chess.Move]:
        """Same interface as NeuralPolicy so the evaluator works unchanged.

        Not batched across boards: each board needs its own variable-length set
        of children. Batching within a board is where the win already is.
        """
        return [self.play(b) for b in boards]

    def play_with_commentary(self, board: chess.Board) -> tuple[chess.Move, dict]:
        """A move plus what the model thought, for chat and logging."""
        ranked = self.rank_moves(board)
        move, best = ranked[0]
        _, conf = self.evaluate([board])
        return move, {
            "win_prob": best,
            "uncertainty": float(conf[0]),
            "margin": best - (ranked[1][1] if len(ranked) > 1 else best),
            "considered": len(ranked),
            "top": [(m.uci(), round(v, 3)) for m, v in ranked[:3]],
        }
