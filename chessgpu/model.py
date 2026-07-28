"""The transformer. LLaMA-shaped decoder, ported from DeepMind's Haiku version.

Architecture, matching `searchless_chess/src/transformer.py`:

    tokens -> shift_right (prepend BOS=0, drop last)
           -> Embed * sqrt(d) + sinusoidal positions
           -> N x [ h += Attn(LN(h)) ; h += SwiGLU(LN(h)) ]
           -> LN
           -> Linear(d -> output_size)          (this one has a bias)
           -> log_softmax

No biases anywhere except the final projection. Pre-norm. MLP widening 4x with
SwiGLU gating, so the MLP is three matrices, not two.

Only the final position's output is ever used: the loss is cross-entropy on the
move given the position, and every earlier position is masked out. See the note
on `causal` below, which is the most interesting knob in this file.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from chessgpu.tokenizer import NUM_ACTIONS, SEQUENCE_LENGTH, VOCAB_SIZE


@dataclasses.dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    output_size: int = NUM_ACTIONS
    embedding_dim: int = 256
    num_layers: int = 8
    num_heads: int = 8
    widening_factor: int = 4
    emb_init_scale: float = 0.02
    apply_post_ln: bool = True
    # Upstream trains causally, because it is written as a generic sequence
    # model. For this task the mask is arguably dead weight: only the last
    # position's logits are used, and that position already attends to the
    # whole board. All the causal mask does is stop the 77 state tokens from
    # seeing each other. Kept True to reproduce their checkpoints; flipping it
    # is the first experiment worth running.
    causal: bool = True

    @property
    def head_dim(self) -> int:
        if self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        return self.embedding_dim // self.num_heads


# Parameter counts confirmed by building them: 9.0M / 136.2M / 270.0M.
PRESETS: dict[str, ModelConfig] = {
    "9M": ModelConfig(embedding_dim=256, num_layers=8, num_heads=8),
    "136M": ModelConfig(embedding_dim=1024, num_layers=8, num_heads=8),
    "270M": ModelConfig(embedding_dim=1024, num_layers=16, num_heads=8),
    # Something small enough to iterate on in seconds while debugging.
    "tiny": ModelConfig(embedding_dim=64, num_layers=2, num_heads=4),
}


def sinusoid_position_encoding(
    sequence_length: int, hidden_size: int, max_timescale: float = 1e4
) -> np.ndarray:
    """Upstream's sinusoidal encoding, quirk included.

    `freqs` runs over `arange(0, d+1, 2)`, which has d/2 + 1 entries rather than
    the usual d/2. The concatenated sin/cos block is therefore d+2 wide and gets
    truncated back to d. That means the last cos channels are dropped and the
    sin/cos pairing is off by one relative to the textbook formulation. It is
    reproduced exactly because the pretrained checkpoints were trained with it.
    """
    freqs = np.arange(0, hidden_size + 1, 2)
    inv_freq = max_timescale ** (-freqs / hidden_size)
    pos_seq = np.arange(sequence_length)
    sinusoid_inp = np.einsum("i,j->ij", pos_seq, inv_freq)
    embeddings = np.concatenate([np.sin(sinusoid_inp), np.cos(sinusoid_inp)], axis=-1)
    return embeddings[:, :hidden_size]


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, h = cfg.embedding_dim, cfg.num_heads * cfg.head_dim
        self.q = nn.Linear(d, h, bias=False)
        self.k = nn.Linear(d, h, bias=False)
        self.v = nn.Linear(d, h, bias=False)
        self.out = nn.Linear(h, d, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        nh, hd = self.cfg.num_heads, self.cfg.head_dim
        # [B, T, H*D] -> [B, H, T, D]
        q = self.q(x).view(b, t, nh, hd).transpose(1, 2)
        k = self.k(x).view(b, t, nh, hd).transpose(1, 2)
        v = self.v(x).view(b, t, nh, hd).transpose(1, 2)
        # SDPA's default scale is 1/sqrt(head_dim), which is what upstream uses.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.cfg.causal)
        y = y.transpose(1, 2).reshape(b, t, nh * hd)
        return self.out(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.embedding_dim
        ffn = d * cfg.widening_factor
        self.gate = nn.Linear(d, ffn, bias=False)
        self.up = nn.Linear(d, ffn, bias=False)
        self.down = nn.Linear(ffn, d, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(cfg.embedding_dim)
        self.attn = MultiHeadAttention(cfg)
        self.ln_mlp = nn.LayerNorm(cfg.embedding_dim)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_attn(x))
        return x + self.mlp(self.ln_mlp(x))


class ChessTransformer(nn.Module):
    """Maps 77 board tokens to a distribution over the 1968-move action space."""

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.num_layers))
        self.ln_out = nn.LayerNorm(cfg.embedding_dim) if cfg.apply_post_ln else nn.Identity()
        self.head = nn.Linear(cfg.embedding_dim, cfg.output_size, bias=True)

        # Sequence is 77 state tokens plus one predicted token, and shift_right
        # keeps the length the same, so positions run to 78.
        pos = sinusoid_position_encoding(SEQUENCE_LENGTH + 1, cfg.embedding_dim)
        self.register_buffer("pos", torch.from_numpy(pos).float(), persistent=False)

        self.apply(self._init)

    def _init(self, module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(
                module.weight, std=self.cfg.emb_init_scale, a=-2 * self.cfg.emb_init_scale,
                b=2 * self.cfg.emb_init_scale,
            )
        elif isinstance(module, nn.Linear):
            # Haiku's Linear default is truncated normal with stddev 1/sqrt(fan_in).
            fan_in = module.weight.shape[1]
            std = 1.0 / math.sqrt(fan_in)
            nn.init.trunc_normal_(module.weight, std=std, a=-2 * std, b=2 * std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens: Tensor, *, log_softmax: bool = False) -> Tensor:
        """tokens: int64 [B, 77]. Returns [B, output_size].

        Upstream builds the 78-long sequence [state..., target] and shifts it
        right, so the model input is [BOS, state...]. We only ever read the
        final position, so we construct that input directly rather than
        carrying a target we would immediately discard.

        Raw logits by default. Upstream returns log-probs, but argmax is
        invariant to the softmax and F.cross_entropy wants logits, so
        normalizing here would just burn memory and a kernel on every step.
        Pass log_softmax=True when the calibrated probabilities are actually
        wanted, e.g. temperature sampling or a value head later.
        """
        b, t = tokens.shape
        if t != SEQUENCE_LENGTH:
            raise ValueError(f"expected {SEQUENCE_LENGTH} tokens, got {t}")

        bos = tokens.new_zeros((b, 1))
        x = torch.cat([bos, tokens], dim=1)          # [B, 78], the shift_right input

        h = self.embed(x) * math.sqrt(self.cfg.embedding_dim)
        h = h + self.pos[: x.shape[1]].to(h.dtype)
        for block in self.blocks:
            h = block(h)
        h = self.ln_out(h)
        logits = self.head(h[:, -1])                 # only the last position matters
        if log_softmax:
            return F.log_softmax(logits.float(), dim=-1)
        return logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build(preset: str = "9M", **overrides) -> ChessTransformer:
    cfg = dataclasses.replace(PRESETS[preset], **overrides)
    return ChessTransformer(cfg)
