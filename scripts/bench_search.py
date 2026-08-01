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


def search_nps(mcts_cls, value: ValuePolicy, policy, seconds: float, batch: int) -> float:
    mcts = mcts_cls(value, policy=policy, simulations=10**9, batch=batch, reuse=False)
    board = chess.Board(POSITION)
    mcts.search(board.copy(), deadline=time.perf_counter() + 3.0)   # warm
    best = 0.0
    for _ in range(3):
        mcts.reset()
        start = time.perf_counter()
        mcts.search(board.copy(), deadline=start + seconds)
        best = max(best, mcts.evaluations / (time.perf_counter() - start))
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="136M", choices=list(PRESETS))
    ap.add_argument("--baseline", default="9M", choices=list(PRESETS))
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

    policy, _ = load_policy(str(ROOT / "runs/policy.pt"), device=args.device)

    results = {}
    for name in (args.baseline, args.preset):
        cfg = ModelConfig(**{**PRESETS[name].__dict__, "output_size": args.bins})
        model = ChessTransformer(cfg)
        value = ValuePolicy(model, HLGauss(bins=args.bins), device=args.device)
        results[name] = search_nps(mcts_cls, value, policy, args.seconds, args.batch)
        print(f"  {name:>5}  {results[name]:7.0f} nps in the search loop "
              f"({model.num_parameters():,} params)")
        del model, value
        torch.cuda.empty_cache()

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
