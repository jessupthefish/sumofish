"""One place that turns checkpoint files into a (value, policy) pair.

Four call sites used to duplicate this: `search_engine.py::main`, the live
bot's entry point; `scripts/match.py`, the measurement instrument;
`scripts/smoke.py`, the promotion gate; and `scripts/bench_search.py`. They
duplicated it deliberately -- `search_engine.py` carries a comment saying
refactoring the bot's entry point was a bigger change than that night
warranted -- and the cost came due when the fused net arrived: three of the
four load a fused checkpoint into an engine that runs, plays legal moves, and
is silently wrong, because every one of them reads the bin count as
`ck["cfg"]["output_size"]`, which is 2032 on a fused net and 64 on a value net.

A FUSED checkpoint is one trunk with two heads and a `bins + NUM_ACTIONS`
output: value logits in `[:, :bins]`, policy logits in `[:, bins:]`. Loading it
correctly is three things, and missing any one of them is a different silent
failure:

1. `HLGauss(bins=...)` must get 64, not 2032. Getting this wrong raises
   ("size of tensor a (2032) must match tensor b (64)"), which is the *loud*
   failure and therefore the harmless one.
2. The policy prior must read columns `[bins:]`. Reading from column 0 instead
   -- which is what `NeuralPolicy` does unless told otherwise -- means every
   move id indexes 64 columns to the left of its own logit. That is a prior
   over the wrong moves, and it is not detectable from the outside: the search
   still runs, still masks to legal moves, still returns a legal move, and just
   plays worse for a reason nothing reports.
3. Both wrappers must be backed by the SAME module object. `rust_mcts.py`
   collapses the two forward passes into one only when `pmodel is vmodel`, and
   it is right to insist: fusing is a claim about what runs, not about what the
   file contains. Load the file twice and you get a fused net that pays the
   full 2032-wide forward TWICE per node, which is slower than the two-net
   engine it was built to beat.

Detection is from the checkpoint, never from a flag, for the reason
`rust_mcts.py` already gives: a flag can disagree with the weights and the
weights cannot disagree with themselves.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from sumofish.engines.neural_engine import load_policy
from sumofish.hlgauss import HLGauss
from sumofish.model import ChessTransformer, ModelConfig
from sumofish.tokenizer import NUM_ACTIONS
from sumofish.value_policy import ValuePolicy


def fused_bins(ck: dict) -> int | None:
    """Value-bin count if `ck` is a fused two-head checkpoint, else None.

    Two independent sources, and they must agree. `ck["args"]` records what the
    run was asked to build (`--target both`, `--value-bins 64`); `ck["cfg"]`
    records the shape that got built. Trusting `args` alone would believe a run
    that was configured for fusion and produced something else; trusting `cfg`
    alone would have to infer the split point by assuming NUM_ACTIONS, which is
    exactly the assumption that makes a wrong slice look like a working engine.
    So: derive from the width, cross-check against the recorded intent, and
    refuse rather than guess when they disagree.
    """
    cfg = ck.get("cfg") or {}
    out = cfg.get("output_size")
    if not isinstance(out, int) or out <= NUM_ACTIONS:
        return None
    bins = out - NUM_ACTIONS

    args = ck.get("args") or {}
    if isinstance(args, dict) and args.get("target") is not None:
        if args.get("target") != "both":
            raise ValueError(
                f"checkpoint output_size {out} looks fused ({bins} bins + "
                f"{NUM_ACTIONS} moves) but its run recorded --target "
                f"{args.get('target')!r}; refusing to guess the split"
            )
        recorded = args.get("value_bins")
        if recorded is not None and recorded != bins:
            raise ValueError(
                f"fused checkpoint disagrees with itself: output_size {out} "
                f"implies {bins} value bins, the run recorded {recorded}"
            )
    return bins


def read(path: str | Path, device: str = "cuda:0") -> dict:
    return torch.load(str(path), map_location=device, weights_only=False)


def build_model(ck: dict, device: str = "cuda:0") -> ChessTransformer:
    """The architecture out of the checkpoint, never out of a preset name.

    Filtered against the current dataclass fields for the reason
    `search_engine.py` learned in July: `ModelConfig(**cfg)` hard-binds every
    checkpoint ever written to today's signature, so renaming one field kills
    the engine at boot on files that are otherwise perfectly loadable.
    """
    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    model = ChessTransformer(
        ModelConfig(**{k: v for k, v in ck["cfg"].items() if k in fields})
    )
    # EMA weights, because they are what training evaluated, selected and
    # promoted on. Playing the raw weights ships something never measured.
    state = ck.get("ema") or ck["model"]
    model.load_state_dict({k: v.float() for k, v in state.items()})
    return model.to(device).eval()


def load_value(value_path: str | Path, device: str = "cuda:0") -> ValuePolicy:
    """The value side alone, for callers that never touch the priors.

    Fused-aware for the reason at the top of this file: the bin count is 64 on
    a fused checkpoint and `cfg["output_size"]` says 2032.
    """
    ck = read(value_path, device)
    bins = fused_bins(ck)
    model = build_model(ck, device)
    vp = ValuePolicy(
        model,
        HLGauss(bins=bins if bins is not None else ck["cfg"]["output_size"]),
        device=device,
    )
    vp.step = ck.get("step")
    return vp


def load_nets(
    value_path: str | Path,
    policy_path: str | Path | None = None,
    device: str = "cuda:0",
) -> tuple[ValuePolicy, object, dict]:
    """(value, policy, info) for either a fused checkpoint or two separate nets.

    `policy_path` is IGNORED when the value checkpoint is fused, and `info`
    says so out loud rather than leaving an operator to wonder which of two
    files is playing. There is no arrangement in which a fused value head and a
    foreign policy net are both wanted: the priors that trunk was trained
    alongside are in the same file.
    """
    ck = read(value_path, device)
    bins = fused_bins(ck)
    model = build_model(ck, device)

    info = {
        "fused": bins is not None,
        "value_path": str(value_path),
        "step": ck.get("step"),
        "params": model.num_parameters(),
        "bins": bins if bins is not None else ck["cfg"]["output_size"],
    }

    if bins is not None:
        value = ValuePolicy(model, HLGauss(bins=bins), device=device)
        # The SAME module, offset past the value columns. Same object, so
        # `rust_mcts` takes its one-forward path; correct offset, so the
        # Python core and any direct prior read are right too.
        from sumofish.policy import NeuralPolicy

        policy = NeuralPolicy(model, device=device, offset=bins)
        info["policy_path"] = str(value_path)
        info["policy_ignored"] = (
            str(policy_path) if policy_path is not None else None
        )
    else:
        value = ValuePolicy(
            model, HLGauss(bins=ck["cfg"]["output_size"]), device=device
        )
        if policy_path is None:
            raise ValueError(
                f"{value_path} is a value-only checkpoint; it needs a policy net"
            )
        policy = load_policy(str(policy_path), device=device)[0]
        info["policy_path"] = str(policy_path)

    value.step = ck.get("step")
    return value, policy, info
