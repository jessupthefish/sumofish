"""Turning model outputs into moves.

The model scores all 1968 actions in the action space, most of which are
illegal in any given position. Move selection masks to the legal set and takes
the argmax. That masking is not a detail: unmasked, the model will happily
propose a move that does not exist in the position, and there is no search to
catch it.
"""

from __future__ import annotations

import chess
import numpy as np
import torch

from sumofish.model import ChessTransformer
from sumofish.tokenizer import MOVE_TO_ACTION, tokenize_board


class NeuralPolicy:
    """Picks moves from a trained model. No search, one forward pass."""

    def __init__(
        self,
        model: ChessTransformer,
        device: str = "cuda:0",
        temperature: float = 0.0,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.temperature = temperature
        self.dtype = dtype

    @torch.inference_mode()
    def _logprobs(self, boards: list[chess.Board]) -> torch.Tensor:
        """Raw logits, not log-probs.

        Kept under this name because callers only ever rank with it, and argmax
        plus the temperature softmax below are both invariant to the missing
        normalizer. Use model(x, log_softmax=True) if a true probability is
        needed.
        """
        tokens = np.stack([tokenize_board(b) for b in boards])
        x = torch.from_numpy(tokens).long().to(self.device, non_blocking=True)
        with torch.autocast(self.device.split(":")[0], dtype=self.dtype):
            return self.model(x).float()

    # Rough piece values, used only to decide whether a draw is good for us.
    _VALUE = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
    }

    def _material_edge(self, board: chess.Board) -> int:
        us = board.turn
        edge = 0
        for piece in board.piece_map().values():
            v = self._VALUE[piece.piece_type]
            edge += v if piece.color == us else -v
        return edge

    def _drawish(self, board: chess.Board, move: chess.Move) -> bool:
        """Would this move hand the opponent a draw claim?

        Must be can_claim_draw(), not is_repetition(3). The latter only asks
        whether the position has *already* occurred three times, which is not
        what ends games: a draw is claimable as soon as a repetition is
        available. Using is_repetition here made the guard fire zero times in
        thirty games while fourteen of them ended in threefold repetition.
        """
        board.push(move)
        try:
            return board.can_claim_draw()
        finally:
            board.pop()

    def play_batch(self, boards: list[chess.Board]) -> list[chess.Move]:
        """One move per board. Batched, because the GPU is idle otherwise.

        Includes a repetition guard, which is worth a lot here. The model sees
        only the current position: 77 tokens with no move history in them. It
        is structurally incapable of noticing that it is repeating. Measured
        against random opponents it won every game on material by an average of
        +20 points and still drew 14 of 30 by threefold, shuffling in positions
        it was completely winning.

        A searching engine gets this for free from its repetition table. With no
        search, the policy has to supply it. So when we are clearly ahead,
        moves that let the opponent claim a draw go to the back of the queue.
        When we are behind, a draw is a good outcome and the guard stays off.
        """
        if not boards:
            return []
        logprobs = self._logprobs(boards).cpu().numpy()

        moves: list[chess.Move] = []
        for row, board in zip(logprobs, boards, strict=True):
            legal = list(board.legal_moves)
            if not legal:
                raise ValueError(f"no legal moves in {board.fen()}")
            # A promotion is encoded with its piece suffix; python-chess emits
            # queen promotions with the suffix too, so uci() lines up directly.
            scores = np.full(len(legal), -np.inf, dtype=np.float32)
            for i, mv in enumerate(legal):
                action = MOVE_TO_ACTION.get(mv.uci())
                if action is not None:
                    scores[i] = row[action]
            if not np.isfinite(scores).any():
                # Cannot happen for standard chess (verified over random walks),
                # but falling back beats raising mid-game on lichess.
                moves.append(legal[0])
                continue

            # Repetition guard. Only when ahead: if we are losing, a draw claim
            # is the best available result and should not be avoided. Demote
            # rather than forbid, so a position where every move repeats still
            # produces a legal move.
            if len(legal) > 1 and self._material_edge(board) >= 2:
                penalty = np.array(
                    [self._drawish(board, mv) for mv in legal], dtype=bool
                )
                if penalty.any() and not penalty.all():
                    scores = np.where(penalty, scores - 1e4, scores)
            if self.temperature > 0:
                p = np.exp((scores - scores.max()) / self.temperature)
                p[~np.isfinite(scores)] = 0.0
                p /= p.sum()
                moves.append(legal[int(np.random.choice(len(legal), p=p))])
            else:
                moves.append(legal[int(np.argmax(scores))])
        return moves

    def play(self, board: chess.Board) -> chess.Move:
        return self.play_batch([board])[0]
