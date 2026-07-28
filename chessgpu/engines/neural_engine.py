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
    # Sampling temperature. At 0 the engine is fully deterministic and plays
    # literally the same game every time from the same opening -- measured at
    # 1 unique game in 10 self-plays, which is the thing that stops a human
    # ever wanting a rematch. At 0.15 it is 10/10 unique with no measurable
    # strength cost (T=0 through T=0.5 are statistically indistinguishable
    # head-to-head). This is a diversity knob, not a difficulty knob.
    policy.temperature = float(os.environ.get("CHESSGPU_TEMPERATURE", "0.15"))
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

    def on_option(name: str, value: str) -> bool:
        if name.lower() != "temperature":
            return False
        try:
            policy.temperature = max(0.0, float(value))
        except ValueError:
            print(f"bad Temperature value {value!r}", file=sys.stderr, flush=True)
            return True
        print(f"temperature -> {policy.temperature}", file=sys.stderr, flush=True)
        return True

    run(
        choose,
        name=f"SumoFish {info['params']//1_000_000}M",
        author="Jessupthefish",
        options=[("option name Temperature type string default 0.15", "")],
        on_option=on_option,
    )


if __name__ == "__main__":
    main()
