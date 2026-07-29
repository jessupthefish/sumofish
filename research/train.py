#!/usr/bin/env python
"""THE EDITABLE FILE. An agent rewrites this; the harness only runs it.

Contract with research/run.py, do not break it:

  * Train for at most TIME_BUDGET_S seconds of wall clock, then stop.
  * Evaluate on the fixed held-out slice using the frozen eval below.
  * Print exactly one line: `RESULT {"bpm": <float>, ...}` and nothing else
    after it.
  * Be deterministic given SEED.

The metric is bits per move: cross-entropy over the 1968-action space in
base 2. Lower is better. Random is log2(1968) = 10.94 bits. It is the chess
analogue of Karpathy's bits-per-byte, and it stays comparable if you change
the architecture, the optimizer, or the batch size.

Everything imported from `chessgpu` is frozen infrastructure and is verified
byte-exact against DeepMind's reference. Do not edit it, and do not
reimplement the tokenizer here. Change the model and the training recipe.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chessgpu.data import make_loader  # noqa: E402
from chessgpu.tokenizer import NUM_ACTIONS, SEQUENCE_LENGTH, VOCAB_SIZE  # noqa: E402
# The metric, from outside the editable surface. This import is ABOVE the marker
# so `check_frozen` covers it, and `run.py` also rejects a `def evaluate` below
# the marker, because a later definition would shadow this name.
from research import harness_eval  # noqa: E402

# ============================================================================
# FROZEN. run.py enforces these; changing them makes results incomparable.
# ============================================================================
TIME_BUDGET_S = 300
SEED = 1337
EVAL_BATCHES = 40
EVAL_BATCH_SIZE = 1024
TRAIN_BAG = ROOT / "data/train/behavioral_cloning_data.bag"
VAL_BAG = ROOT / "data/test/behavioral_cloning_data.bag"

# ============================================================================
# EDITABLE. Everything below here is fair game.
# ============================================================================
BATCH_SIZE = 1024
LR = 3e-4
WARMUP = 300
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
WORKERS = 10

EMBED_DIM = 256
NUM_LAYERS = 8
NUM_HEADS = 8
WIDENING = 4
CAUSAL = True          # upstream trains causally; see note in chessgpu/model.py
POST_LN = True


def positions(seq_len: int, dim: int, max_timescale: float = 1e4) -> Tensor:
    freqs = torch.arange(0, dim + 1, 2, dtype=torch.float32)
    inv = max_timescale ** (-freqs / dim)
    pos = torch.arange(seq_len, dtype=torch.float32)[:, None] * inv[None, :]
    return torch.cat([pos.sin(), pos.cos()], dim=-1)[:, :dim]


class Block(nn.Module):
    def __init__(self, d: int, heads: int, widening: int, causal: bool) -> None:
        super().__init__()
        self.h, self.hd, self.causal = heads, d // heads, causal
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        ffn = d * widening
        self.gate = nn.Linear(d, ffn, bias=False)
        self.up = nn.Linear(d, ffn, bias=False)
        self.down = nn.Linear(ffn, d, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q = q.view(b, t, self.h, self.hd).transpose(1, 2)
        k = k.view(b, t, self.h, self.hd).transpose(1, 2)
        v = v.view(b, t, self.h, self.hd).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        h = self.ln2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        d = EMBED_DIM
        self.d = d
        self.embed = nn.Embedding(VOCAB_SIZE, d)
        self.blocks = nn.ModuleList(
            Block(d, NUM_HEADS, WIDENING, CAUSAL) for _ in range(NUM_LAYERS)
        )
        self.ln = nn.LayerNorm(d) if POST_LN else nn.Identity()
        self.head = nn.Linear(d, NUM_ACTIONS)
        self.register_buffer("pos", positions(SEQUENCE_LENGTH + 1, d), persistent=False)
        self.apply(self._init)

    def _init(self, m: nn.Module) -> None:
        if isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02, a=-0.04, b=0.04)
        elif isinstance(m, nn.Linear):
            std = 1.0 / math.sqrt(m.weight.shape[1])
            nn.init.trunc_normal_(m.weight, std=std, a=-2 * std, b=2 * std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, tokens: Tensor) -> Tensor:
        x = torch.cat([tokens.new_zeros((tokens.shape[0], 1)), tokens], dim=1)
        h = self.embed(x) * math.sqrt(self.d) + self.pos.to(self.embed.weight.dtype)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln(h)[:, -1])


def lr_at(step: int, total: int) -> float:
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    p = min(1.0, (step - WARMUP) / max(1, total - WARMUP))
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def evaluate(model: nn.Module, device: str) -> float:
    """Held-out bits per move. Delegates to `research/harness_eval.py`.

    Kept as a one-line shim so existing call sites still read `evaluate(...)`.
    The body it used to contain is the yardstick and now lives outside this file;
    see that module for why. Editing this shim to compute anything else is
    exactly the move the harness exists to prevent, and `run.py` checks for it.
    """
    return harness_eval.evaluate(
        model, device,
        make_loader=make_loader,
        val_bag=VAL_BAG,
        eval_batches=EVAL_BATCHES,
        eval_batch_size=EVAL_BATCH_SIZE,
        seed=SEED,
    )


def main() -> None:
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda:0"

    model = Model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    net = torch.compile(model)

    loader = make_loader(
        TRAIN_BAG, batch_size=BATCH_SIZE, num_workers=WORKERS, seed=SEED
    )
    batches = iter(loader)

    # The step budget is a guess used only to shape the LR schedule; the wall
    # clock is what actually ends the run.
    est_steps = max(200, int(TIME_BUDGET_S * 10_000 / BATCH_SIZE))

    start = time.perf_counter()
    step = 0
    while time.perf_counter() - start < TIME_BUDGET_S:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, est_steps)
        tokens, actions = next(batches)
        tokens = tokens.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net(tokens)
        loss = F.cross_entropy(logits.float(), actions)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        step += 1

    train_seconds = time.perf_counter() - start
    bpm = evaluate(model, device)

    print(
        "RESULT "
        + json.dumps(
            {
                "bpm": round(bpm, 5),
                "params": n_params,
                "steps": step,
                "positions": step * BATCH_SIZE,
                "train_seconds": round(train_seconds, 1),
                "config": {
                    "batch_size": BATCH_SIZE, "lr": LR, "warmup": WARMUP,
                    "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
                    "embed_dim": EMBED_DIM, "num_layers": NUM_LAYERS,
                    "num_heads": NUM_HEADS, "widening": WIDENING,
                    "causal": CAUSAL, "post_ln": POST_LN,
                },
            }
        )
    )


if __name__ == "__main__":
    main()
