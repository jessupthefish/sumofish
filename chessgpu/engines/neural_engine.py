"""SumoFish's real engine: a transformer, one forward pass, no search.

Time management is nearly trivial here. A conventional engine has to decide how
long to think; this one has a fixed cost per move regardless of the clock, so
the only real job is to not crash and to answer promptly. That is a genuine
structural advantage in blitz and the main reason a home-hosted bot on a
residential connection is viable at all.

    bin/sumofish            uses runs/current.pt
    CHESSGPU_CKPT=... bin/sumofish
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import chess
import torch

from chessgpu.model import ChessTransformer, ModelConfig
from chessgpu.policy import NeuralPolicy
from chessgpu.uci import Limits, run

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CKPT = ROOT / "runs" / "current.pt"


def load_policy(
    ckpt_path: str | Path,
    device: str | None = None,
    prefer_ema: bool = True,
) -> tuple[NeuralPolicy, dict]:
    """Load a checkpoint into a ready-to-play policy.

    Prefers the EMA weights: they are the ones the training run evaluates and
    promotes on, so playing the raw weights would mean shipping something that
    was never measured.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["cfg"])
    model = ChessTransformer(cfg)

    state = ckpt.get("ema") if prefer_ema and ckpt.get("ema") else ckpt["model"]
    model.load_state_dict({k: v.to(torch.float32) for k, v in state.items()})

    info = {
        "step": ckpt.get("step"),
        "params": model.num_parameters(),
        "ema": state is ckpt.get("ema"),
        "device": device,
    }
    return NeuralPolicy(model, device=device), info


def main() -> None:
    ckpt_path = os.environ.get("CHESSGPU_CKPT", str(DEFAULT_CKPT))
    if not Path(ckpt_path).exists():
        # Fail loudly on stderr rather than silently playing badly. lichess-bot
        # surfaces a startup failure; a quietly broken engine looks like a bad
        # model and wastes an evening.
        print(f"checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    policy, info = load_policy(ckpt_path)
    print(
        f"loaded {ckpt_path} step={info['step']} params={info['params']:,} "
        f"ema={info['ema']} device={info['device']}",
        file=sys.stderr,
    )

    # Warm the kernels now, during UCI handshake, rather than on the first move
    # of a real game where the clock is running.
    policy.play(chess.Board())

    def choose(board: chess.Board, limits: Limits) -> chess.Move:
        return policy.play(board)

    run(choose, name=f"SumoFish {info['params']//1_000_000}M", author="Jessupthefish")


if __name__ == "__main__":
    main()
