#!/usr/bin/env python
"""How much dearer is a bigger net, inside the search that has to use it?

    scripts/bench_search.py --preset 136M --out runs/lab/forward-bench.json

The number this produces, `ratio`, is the one that decides whether scaling the
model is worth doing at all, and it is not the parameter count.

The deployed bot is clock-bound: `think_time` hands the search a number of
seconds, not a number of simulations. So a net that costs m times as much per
forward pass does not simply cost m times as much -- it costs the search
log2(m) doublings of simulations, every move, forever. If a doubling of search
is worth D Elo (measured separately by the exchange-rate matches), a bigger net
must win by D x log2(m) at equal simulations merely to break even at equal time.

m is measured here rather than assumed, for two reasons. Parameter count is a
bad proxy: the 9M runs at roughly 11% of the card's throughput, so a wider model
uses it better and costs less than its parameter ratio implies. And the network
is only ~9% of search wall clock after this session's work, so m as felt by the
search is much closer to 1 than m as felt by a benchmark of the model alone.
Measuring the model in isolation would overstate the cost by an order of
magnitude and kill a scale-up that was actually affordable.

Weights are random. This measures cost, not quality, and an untrained net of
the right shape costs exactly what a trained one costs.

WHICH CORE THIS MEASURES, 2026-07-31. Until today this file imported
`sumofish.mcts` directly and instantiated it unconditionally, so every number it
ever produced -- including the `scale_m = 2.17` still sitting in
`runs/lab/state.json` and feeding `decide_scale`'s break-even bar -- is the cost
ratio as felt by the PYTHON tree, measured before the Rust port. That is the
exact trap LAB-NOTES records ("do not extrapolate a speedup from a profile taken
before the previous speedup; re-profile after every port"), and it went
unnoticed because `m` is an input to the port's own justification rather than an
output of it.

It matters because `m` is not a property of the net, it is a property of the net
DIVIDED BY the tree around it: the paragraph above says so itself ("the network
is only ~9% of search wall clock"). That 9% was the Python profile. With the
tree in Rust the network dominates instead, so the same dearer net is diluted by
much less and `m` must come out WORSE. A stale `m` therefore understates the
cost of scaling, which is the direction that wrongly licenses a 35-hour run.

So the core is now selected by `select_mcts_class()` -- the same resolution the
engine and `match.py` use, Rust unless `CHESSGPU_CORE=python` -- and it is
recorded in the output. `--core` pins it explicitly for an A/B. Re-running with
`--core python` should reproduce the historical number and is how you check the
harness rather than the tree.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import re

import chess
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sumofish.engines.neural_engine import load_policy  # noqa: E402
from sumofish.hlgauss import HLGauss  # noqa: E402
from sumofish.model import PRESETS, ChessTransformer, ModelConfig  # noqa: E402
from sumofish.rust_mcts import select_mcts_class  # noqa: E402
from sumofish.value_policy import ValuePolicy  # noqa: E402

# A quiet middlegame with a typical branching factor. Not the opening, where
# the tree is shallow and the measurement is dominated by expansion.
POSITION = "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9"


class _FusedHead(torch.nn.Module):
    """A two-head net presented to the value path, which expects one head.

    `--fused` builds a trunk with a `bins + NUM_ACTIONS` output so that ONE
    forward pass produces both the value distribution and the policy logits,
    which is the whole proposal. But `ValuePolicy` hands the output straight to
    `HLGauss`, which expects exactly `bins` columns, so the raw fused model
    fails with "the size of tensor a (2032) must match the size of tensor b
    (64)" before a single node is searched.

    Slicing here is not a cheat, it is the measurement. The full 2032-wide head
    is computed -- that cost is real and a real fused net pays it -- and only
    the value columns are handed on, because in this arm the policy net's own
    forward has been zeroed and nothing downstream wants the policy half. What
    is measured is therefore one forward of the fused shape, which is exactly
    what the proposed net does per node.
    """

    def __init__(self, inner: torch.nn.Module, bins: int) -> None:
        super().__init__()
        self.inner = inner
        self.bins = bins

    def forward(self, x):                                   # noqa: ANN001
        return self.inner(x)[:, :self.bins]


class _FreeNet(torch.nn.Module):
    """Stands in for one of the two nets so that its forward pass costs nothing.

    Used only by `--decompose`. It returns a cached constant of the right shape,
    so the search still runs its whole tree, its tokenisation, its numpy prior
    softmax and the OTHER forward pass. The delta against the `both` arm is
    therefore that net's true serial cost inside the loop, which is the one
    quantity the historical two-point fit could not see.
    """

    def __init__(self, out: int, device: str) -> None:
        super().__init__()
        self.out, self.device_, self._cache = out, device, {}

    def forward(self, x):                                   # noqa: D102
        n = x.shape[0]
        t = self._cache.get(n)
        if t is None:
            t = torch.zeros(n, self.out, device=self.device_)
            self._cache[n] = t
        return t


_SPEC = re.compile(r"^d(\d+)(?:L(\d+))?(?:h(\d+))?$", re.I)


def shape(spec: str, out: int) -> ModelConfig:
    """A preset name, or `d384`, `d384L8`, `d512L8h16`.

    `--preset` used to take `choices=list(PRESETS)`, so no width between 256 and
    1024 could be benched at all -- which is why the cost of a 2x net was argued
    from a two-point fit for two weeks instead of being measured in a minute.
    Heads default to d/32 so head_dim stays 32, the value every preset uses and
    the one exact channel duplication needs.
    """
    if spec in PRESETS:
        return ModelConfig(**{**PRESETS[spec].__dict__, "output_size": out})
    m = _SPEC.match(spec)
    if not m:
        raise SystemExit(
            f"bad shape {spec!r}: give a preset ({', '.join(PRESETS)}) "
            f"or dNNN[LN][hN], e.g. d384 or d512L8h16")
    d = int(m.group(1))
    layers = int(m.group(2)) if m.group(2) else 8
    heads = int(m.group(3)) if m.group(3) else max(1, d // 32)
    if d % heads:
        raise SystemExit(f"{spec}: embedding_dim {d} not divisible by {heads} heads")
    return ModelConfig(**{**PRESETS["9M"].__dict__, "embedding_dim": d,
                          "num_layers": layers, "num_heads": heads,
                          "output_size": out})


def search_nps(mcts_cls, value: ValuePolicy, policy, seconds: float, batch: int,
               compile_nets: bool = False) -> float:
    # pad_batches is coupled to compile_nets for the same reason
    # search_engine.py and match.py couple them: CUDA graphs need a static
    # shape, and without padding the row count varies, the graph recaptures
    # every batch, and the measurement is of the recapture.
    kw = {"compile_nets": True, "pad_batches": True} if compile_nets else {}
    mcts = mcts_cls(value, policy=policy, simulations=10**9, batch=batch,
                    reuse=False, **kw)
    board = chess.Board(POSITION)
    mcts.search(board.copy(), deadline=time.perf_counter() + 3.0)   # warm
    if compile_nets:
        # torch.compile in reduce-overhead mode traces, then CAPTURES a CUDA
        # graph, and neither is steady state. Measured 2026-08-15 with a single
        # 5s warm search: three identical one-shape-per-process runs read
        # 5,402 / 12,560 / 6,874 nps, a 2.3x spread, because the timed window
        # sometimes still contained capture. Warm until two consecutive
        # measurements agree within 5% instead of guessing a duration.
        prev = None
        for _ in range(8):
            mcts.reset()
            t = time.perf_counter()
            mcts.search(board.copy(), deadline=t + 3.0)
            cur = mcts.evaluations / (time.perf_counter() - t)
            if prev is not None and abs(cur - prev) / max(cur, prev) < 0.05:
                break
            prev = cur
    best = 0.0
    for _ in range(3):
        mcts.reset()
        start = time.perf_counter()
        mcts.search(board.copy(), deadline=start + seconds)
        best = max(best, mcts.evaluations / (time.perf_counter() - start))
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="136M",
                    help="preset name or shape spec: d384, d512L8h16")
    ap.add_argument("--baseline", default="9M",
                    help="preset name or shape spec")
    ap.add_argument("--decompose", action="store_true",
                    help="Arm C. Instead of comparing two widths, run the "
                         "BASELINE shape three ways -- both nets, value only, "
                         "policy only -- to split the per-node cost into the "
                         "tree, the value forward and the policy forward. This "
                         "is the term the historical fit buried in its "
                         "intercept: bench_search loaded the policy net once "
                         "outside the loop and varied only the value net, so "
                         "the measured 3.06x was (tree+p256+v1024)/"
                         "(tree+p256+v256) and the policy pass a fused trunk "
                         "would delete never appeared as a variable.")
    ap.add_argument("--fused", action="store_true",
                    help="model the shared-trunk two-head net: ONE forward of "
                         "the given shape with a bins+1968 wide head, against "
                         "today's two separate forwards.")
    ap.add_argument("--compile", action="store_true",
                    help="measure under torch.compile, the DEPLOYED config "
                         "since 2026-08-15. It cuts per-launch overhead, which "
                         "is exactly the term the fusion case rests on, so "
                         "every cost number taken without it describes a "
                         "configuration that no longer ships. Requires "
                         "--single.")
    ap.add_argument("--single", default=None, metavar="SPEC",
                    help="measure ONE shape and print its nps instead of "
                         "comparing two, leaving the arithmetic to the caller. "
                         "Required with --compile.")
    ap.add_argument("--bins", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--core", default=None, choices=("rust", "python"),
                    help="pin the search core. Default: whatever CHESSGPU_CORE "
                         "resolves to, i.e. the core that actually plays.")
    args = ap.parse_args()

    # Pin by env rather than by passing the class around, so `select_mcts_class`
    # stays the single place that decides and a typo still fails loudly there.
    if args.core:
        os.environ["CHESSGPU_CORE"] = args.core
    mcts_cls, core_name = select_mcts_class()
    print(f"  core: {core_name}  (batch {args.batch}, {args.seconds}s per arm)")

    from sumofish.tokenizer import NUM_ACTIONS

    def real_policy():
        return load_policy(str(ROOT / "runs/policy.pt"), device=args.device)[0]

    def arm(spec: str, *, skip_policy=False, skip_value=False, fused=False):
        """One measured configuration. Returns (nps, params, label)."""
        out = args.bins + NUM_ACTIONS if fused else args.bins
        cfg = shape(spec, out)
        model = ChessTransformer(cfg)
        # A fused trunk emits bins + NUM_ACTIONS; the value path wants bins.
        value = ValuePolicy(_FusedHead(model, args.bins) if fused else model,
                            HLGauss(bins=args.bins), device=args.device)
        if skip_value:
            # Keep the ValuePolicy wrapper (the search calls into it) but make
            # its forward free, so what is left is tree + policy forward.
            value.model = _FreeNet(args.bins, args.device).to(args.device).eval()
        pol = real_policy()
        if skip_policy or fused:
            pol.model = _FreeNet(NUM_ACTIONS, args.device).to(args.device).eval()
        nps = search_nps(mcts_cls, value, pol, args.seconds, args.batch,
                         compile_nets=args.compile)
        n = 0 if skip_value else model.num_parameters()
        del model, value, pol
        torch.cuda.empty_cache()
        return nps, n

    if args.compile and not args.single:
        raise SystemExit(
            "--compile requires --single: one shape per process.\n"
            "\n"
            "Measured 2026-08-15. Two arms in ONE process, BOTH d=256, read\n"
            "13,087 and 7,077 nps: the first captures a CUDA graph, the second\n"
            "gets a fresh model object that never recaptures, so it is\n"
            "effectively uncompiled. The same-shape control reads 1.85x where\n"
            "it must read 1.00x, and on a re-run the fast and slow arms SWAP,\n"
            "so it is not even a fixed bias. torch._dynamo.reset() between arms\n"
            "makes it worse (4,991 nps), because the recompilation then lands\n"
            "inside the timed window.\n"
            "\n"
            "There is no in-process fix. Run one arm per process and compare\n"
            "across them; scripts/run_cost_under_compile.sh does that."
        )

    if args.single:
        nps, params = arm(args.single, fused=args.fused)
        tag = (" fused two-head" if args.fused else "") + (" +compile" if args.compile else "")
        print(f"  {args.single:>10}  {nps:7.0f} nps in the search loop "
              f"({params:,} params){tag}")
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(
                {"mode": "single", "spec": args.single, "nps": round(nps),
                 "params": params, "fused": bool(args.fused),
                 "compile": bool(args.compile), "core": core_name,
                 "batch": args.batch, "seconds": args.seconds,
                 "position": POSITION}, indent=2))
            print(f"  wrote {args.out}")
        return

    if args.decompose:
        # Three arms at ONE shape. The point is the differences, not the levels.
        both, n_both = arm(args.baseline)
        v_only, _ = arm(args.baseline, skip_policy=True)
        p_only, _ = arm(args.baseline, skip_value=True)
        us = lambda nps: 1e6 / nps                      # noqa: E731
        t_both, t_v, t_p = us(both), us(v_only), us(p_only)
        # tree = what survives when BOTH forwards are removed, inferred:
        #   t_both = tree + v + p ;  t_v = tree + v ;  t_p = tree + p
        tree = t_v + t_p - t_both
        print(f"\n  both nets   {both:7.0f} nps   {t_both:7.2f} us/node")
        print(f"  value only  {v_only:7.0f} nps   {t_v:7.2f} us/node")
        print(f"  policy only {p_only:7.0f} nps   {t_p:7.2f} us/node")
        print(f"\n  inferred:  tree {tree:6.2f}   value fwd {t_both - t_p:6.2f}"
              f"   policy fwd {t_both - t_v:6.2f}  (us/node)")
        share = (t_both - t_v) / t_both
        print(f"\n  the policy forward is {share:.1%} of per-node cost.")
        print("  KILL CONDITION (plan 1.1): under 25% means the policy forward "
              "is not serial,\n  a fused trunk saves less than half what the "
              "arithmetic assumes, and the\n  cost-neutral width drops toward "
              "d=320-384.")
        out_obj = {"mode": "decompose", "shape": args.baseline,
                   "nps": {"both": round(both), "value_only": round(v_only),
                           "policy_only": round(p_only)},
                   "us_per_node": {"both": round(t_both, 3),
                                   "value_only": round(t_v, 3),
                                   "policy_only": round(t_p, 3),
                                   "tree_inferred": round(tree, 3),
                                   "value_fwd": round(t_both - t_p, 3),
                                   "policy_fwd": round(t_both - t_v, 3)},
                   "policy_share": round(share, 4),
                   "core": core_name, "batch": args.batch, "position": POSITION}
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out_obj, indent=2))
            print(f"\n  wrote {args.out}")
        return

    results = {}
    for name in (args.baseline, args.preset):
        fused = args.fused and name == args.preset
        nps, n = arm(name, fused=fused)
        results[name] = nps
        tag = " fused two-head" if fused else ""
        print(f"  {name:>10}  {nps:7.0f} nps in the search loop "
              f"({n:,} params){tag}")

    # Cost ratio as the SEARCH feels it: how many times fewer nodes per second.
    ratio = results[args.baseline] / results[args.preset]
    out = {"baseline": args.baseline, "preset": args.preset,
           "nps": {k: round(v) for k, v in results.items()},
           "ratio": round(ratio, 3),
           "doublings_lost": round(__import__("math").log2(ratio), 3),
           # `core` is not decoration: a ratio is only comparable to another
           # ratio measured on the same tree. The historical 2.17 carries no
           # such field, which is precisely why it outlived the port.
           "core": core_name,
           "batch": args.batch, "position": POSITION}
    print(f"\n  {args.preset} costs {ratio:.2f}x per node as the search feels it, "
          f"= {out['doublings_lost']:.2f} doublings of simulations at equal time")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
