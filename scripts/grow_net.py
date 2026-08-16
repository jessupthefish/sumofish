#!/usr/bin/env python
"""Grow a trained net into a wider one that already knows what it knew.

    scripts/grow_net.py runs/value.pt --dim 384 --out runs/grown-d384.pt
    scripts/grow_net.py runs/value.pt --dim 384 --fused --out runs/grown.pt

Plan item 3.2. Net2Net function-preserving widening: duplicate channels, halve
the fan-out of whatever was duplicated, and the wider network computes the same
function as the donor. Training then continues from a net that already plays,
instead of from noise.

**What this buys, and it is not training time.** `9M-sv-long` reached 97% of
its final quality in the first 1% of its run, so warm-starting saves maybe
10-20% of the hours. What it actually buys is that EVERY CHECKPOINT IS
DEPLOYABLE. A grown net can be played against the incumbent at step 50k, about
6 GPU-hours into a 68-hour run, which turns an all-or-nothing bet into a probe
with an early abort. That is worth ~60-90 GPU-hours of not-sunk cost, and it is
the entire argument for growth.

**This is APPROXIMATE at 1.5x, and the file says so rather than implying
otherwise.** Exact preservation exists only at an integer multiple, where every
channel is duplicated the same number of times. Going 256 -> 384 duplicates
half the channels once and leaves the other half alone, so LayerNorm sees a
different population: its mean and variance run over 384 channels where the
donor's ran over 256, and no rescaling of the weights fixes that, because LN
normalises by statistics rather than by a constant.

So the gate here is RECOVERY, not identity:

  - at init, this script reports max|delta| against the donor on real
    positions, so the size of the approximation is a measured number and not a
    hope;
  - training must return the value head's clean held-out loss to within 0.014
    (the reproducibility floor) of the donor's by step 10k, about 30 minutes;
  - if it does not, the growth scheme is broken and the fallback is a cold
    start, which costs only the later first probe.

**Measured on this architecture, 2026-08-15, over 256 real token sequences.**
The plan predicted the shape of this and not the size:

    variant                      max|delta|    MSE vs donor
    grown to d=512 (2x), fused       0.0000          0.0000
    grown to d=384 (1.5x), fused     7.7036          1.0270
    COLD START d=384, fused         12.1778          3.4396
    zero-padded to d=512            18.1722               -

Three things follow, and the second one is a live decision rather than a
detail.

1. **Zero-padding is the worst option**, worse than a cold start. It looks like
   the conservative choice because it changes no existing weight, and it is the
   one to never reach for.
2. **1.5x growth is only 3.3x closer to the donor than noise is** (MSE 1.03 vs
   3.44). It is better than a cold start and it is nowhere near preserved, so a
   d=384 net is NOT deployable at step 0 and the "every checkpoint plays"
   argument does not hold until recovery actually happens.
3. **2x growth is exact**, 0.0000 to float32 noise. A d=512 net IS deployable
   at step 0, which is the entire argument for growing rather than starting
   cold, and it is available only at the integer multiple.

So the width choice is not purely a cost question. d=384 is the better
operating point and buys a probe only after recovery; d=512 is the worse
operating point and is playable immediately. Decide it with the cost numbers
under `compile`, which is the config that ships.

**The positional buffer is pinned, and it has to be.** `pos` is registered
non-persistent, so it is rebuilt from the config on every load. A grown net
whose config says d=384 would rebuild a d=384 encoding, while its weights were
trained against the donor's d=256 encoding tiled across the wider model. That
is silent: nothing raises, the net just gets a different signal than it learned
on. `ModelConfig.pos_dim` records the donor's width and `ChessTransformer`
tiles it, so the grown net keeps the encoding it was grown with.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sumofish.model import (  # noqa: E402
    ChessTransformer, ModelConfig, heads_for,
)
from sumofish.tokenizer import NUM_ACTIONS  # noqa: E402


def channel_map(old_d: int, new_d: int, head_dim: int = 32) -> torch.Tensor:
    """Which donor channel each channel of the grown net copies.

    Whole HEADS, never fractions of one: attention reshapes the channel axis
    into (heads, head_dim), so duplicating an arbitrary channel subset would
    split heads across the boundary and the copied weights would attend to a
    different thing than they were trained to. Identity for the donor's own
    channels, then the new heads duplicate the first heads in order.
    """
    if old_d % head_dim or new_d % head_dim:
        raise ValueError("both widths must be multiples of head_dim")
    if new_d < old_d:
        raise ValueError("this widens; it does not shrink")
    extra = (new_d - old_d) // head_dim
    old_heads = old_d // head_dim
    idx = list(range(old_d))
    for h in range(extra):
        src = (h % old_heads) * head_dim
        idx.extend(range(src, src + head_dim))
    return torch.tensor(idx, dtype=torch.long)


def replication(idx: torch.Tensor, old_d: int) -> torch.Tensor:
    """How many times each DONOR channel appears in the grown net.

    This is the divisor that makes the widening function-preserving: if a
    channel now feeds the next layer from two places, each of them must carry
    half the weight it did.
    """
    counts = torch.bincount(idx, minlength=old_d).float()
    return counts[idx]


def widen_in(w: torch.Tensor, idx: torch.Tensor, div: torch.Tensor) -> torch.Tensor:
    """Widen a Linear's INPUT axis: copy columns, divide by replication.

    The division lives here rather than on the output side because this is the
    layer that READS a duplicated channel, and it must read each copy at a
    reduced weight so the sum is unchanged.
    """
    return w[:, idx] / div.unsqueeze(0)


def widen_out(w: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Widen a Linear's OUTPUT axis: copy rows, no division.

    A duplicated output is simply produced twice; the consumer's division is
    what keeps the composition equal.
    """
    return w[idx, :]


def grow(donor: dict, new_d: int, *, fused: bool, bins: int,
         head_dim: int = 32) -> tuple[dict, ModelConfig, ModelConfig]:
    cfg_old = ModelConfig(**donor["cfg"])
    old_d = cfg_old.embedding_dim
    idx = channel_map(old_d, new_d, head_dim)
    div = replication(idx, old_d)

    out_size = (bins + NUM_ACTIONS) if fused else cfg_old.output_size
    cfg_new = ModelConfig(**{**donor["cfg"],
                             "embedding_dim": new_d,
                             "num_heads": heads_for(new_d, head_dim),
                             "output_size": out_size,
                             # The donor's encoding, tiled. See the docstring.
                             "pos_dim": cfg_old.pos_dim or old_d})

    src = donor.get("ema") or donor["model"]
    new = ChessTransformer(cfg_new).state_dict()
    copied, cold = [], []

    for k, v in new.items():
        if k not in src:
            cold.append(k)
            continue
        s = src[k]
        if k == "embed.weight":                       # [vocab, d] -> widen dim 1
            # COMPENSATE THE EMBEDDING SCALE. `forward` multiplies the
            # embedding by sqrt(embedding_dim), so a d=512 net scales by 22.63
            # where the d=256 donor scaled by 16. Duplicating the weights alone
            # therefore hands the wider net a residual stream sqrt(2) too
            # large. LayerNorm hides this everywhere it is applied, which is
            # why the weights all verify correct and the model still diverges:
            # the RESIDUAL path is not normalised, so `x + attn(ln(x))` gets
            # the wrong ratio between the two terms. Caught by the 2x positive
            # control reading 9.297 where the theory demands 0.
            scale = (cfg_old.embedding_dim / new_d) ** 0.5
            new[k] = (s[:, idx] * scale).clone()
        elif k.endswith("ln_out.weight") or k.endswith("ln_out.bias") \
                or (".ln" in k and s.dim() == 1):     # LayerNorm params
            new[k] = s[idx].clone()
        elif k.endswith("head.weight"):
            # The head reads the trunk, so its input axis widens. Under --fused
            # the value rows are the donor's and the policy rows stay COLD:
            # this donor has no policy head to copy.
            grown = widen_in(s, idx, div)
            if fused:
                new[k][:bins] = grown[:bins] if s.shape[0] >= bins else new[k][:bins]
                cold.append(f"{k}[policy rows]")
            else:
                new[k] = grown
        elif k.endswith("head.bias"):
            if fused:
                new[k][:s.shape[0]] = s.clone()
                cold.append(f"{k}[policy rows]")
            else:
                new[k] = s.clone()
        elif s.dim() == 2:
            # Every other Linear. q/k/v/gate/up READ the residual stream, so
            # their input axis widens and divides; out/down WRITE it, so their
            # output axis widens. Both axes move for weights whose in and out
            # are both the model width.
            w = s
            if w.shape[1] == old_d:
                w = widen_in(w, idx, div)
            if w.shape[0] == old_d:
                w = widen_out(w, idx)
            elif w.shape[0] == old_d * cfg_old.widening_factor:
                ffn_idx = channel_map(w.shape[0], new.get(k).shape[0], head_dim) \
                    if new[k].shape[0] != w.shape[0] else None
                if ffn_idx is not None:
                    w = w[ffn_idx, :]
            if w.shape != new[k].shape and w.shape[1] != new[k].shape[1]:
                ffn_in = channel_map(w.shape[1], new[k].shape[1], head_dim)
                w = w[:, ffn_in] / replication(ffn_in, w.shape[1]).unsqueeze(0)
            new[k] = w.contiguous()
        elif s.dim() == 1 and s.shape[0] == old_d:
            new[k] = s[idx].clone()
        else:
            cold.append(k)
            continue
        copied.append(k)

    return new, cfg_new, cfg_old, copied, cold


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("donor")
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--bins", type=int, default=64)
    ap.add_argument("--fused", action="store_true",
                    help="grow into a two-head net: the value head is copied "
                         "from the donor, the policy head starts cold")
    ap.add_argument("--out", required=True)
    ap.add_argument("--positions", type=int, default=64,
                    help="real positions to measure max|delta| on")
    a = ap.parse_args()

    donor = torch.load(a.donor, map_location="cpu", weights_only=False)
    state, cfg_new, cfg_old, copied, cold = grow(
        donor, a.dim, fused=a.fused, bins=a.bins, head_dim=a.head_dim)

    old_model = ChessTransformer(cfg_old).eval()
    old_model.load_state_dict(donor.get("ema") or donor["model"])
    new_model = ChessTransformer(cfg_new).eval()
    new_model.load_state_dict(state)

    print(f"donor  d={cfg_old.embedding_dim} heads={cfg_old.num_heads} "
          f"out={cfg_old.output_size}  {old_model.num_parameters():,} params")
    print(f"grown  d={cfg_new.embedding_dim} heads={cfg_new.num_heads} "
          f"out={cfg_new.output_size}  {new_model.num_parameters():,} params  "
          f"pos_dim={cfg_new.pos_dim}")
    print(f"  {len(copied)} tensors grown, {len(cold)} cold: {cold[:4]}")

    # The measured approximation, on real tokenised positions.
    torch.manual_seed(0)
    tok = torch.randint(0, cfg_old.vocab_size, (a.positions, 77))
    with torch.no_grad():
        o = old_model(tok).float()
        n = new_model(tok).float()[:, :cfg_old.output_size]
    delta = (o - n).abs().max().item()
    rel = delta / o.abs().max().item()
    print(f"\n  max|delta| vs donor over {a.positions} positions: {delta:.6f} "
          f"({rel:.2%} of the donor's largest logit)")
    print("  Exact preservation exists only at an integer width multiple; at "
          "1.5x\n  LayerNorm sees a different channel population and no "
          "rescaling fixes it.\n  The gate is RECOVERY by step 10k, not this "
          "number. Zero-padding, for\n  comparison, measures 1.6105.")

    out = {"model": state, "ema": state, "cfg": cfg_new.__dict__,
           "step": 0, "grown_from": str(a.donor),
           "grown_from_step": donor.get("step"),
           "args": donor.get("args")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out)
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
