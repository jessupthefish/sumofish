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

from chessgpu.model import ChessTransformer
from chessgpu.tokenizer import MOVE_TO_ACTION, tokenize


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
        tokens = np.stack([tokenize(b.fen()) for b in boards])
        x = torch.from_numpy(tokens).long().to(self.device, non_blocking=True)
        with torch.autocast(self.device.split(":")[0], dtype=self.dtype):
            return self.model(x).float()

    def play_batch(self, boards: list[chess.Board]) -> list[chess.Move]:
        """One move per board. Batched, because the GPU is idle otherwise."""
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
