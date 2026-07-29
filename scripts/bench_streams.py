#!/usr/bin/env python
"""Can the two forward passes overlap? Run when the GPU is FREE.

The profile says the network is 89% of the search once the tree is in Rust, and
that it is launch-bound below ~128 rows. Two independent models are called per
batch -- policy for priors, value for the leaf -- so the obvious question is
whether they can overlap instead of running back to back.

This matters because of what it competes with. Halving the launches per node is
also what a **shared-trunk two-head net** buys, and that needs a ~40 GPU-hour
training run plus a new architecture. If separate CUDA streams get most of the
same win for an afternoon of plumbing, that is worth knowing BEFORE committing
the 40 hours.

Four configurations, same input, same models:

  sequential eager      what the engine does today
  sequential compiled   reduce-overhead, i.e. CUDA graphs
  two streams eager     the cheap idea being tested
  two streams compiled  both, if they compose

DO NOT run this while a --time match is in flight. A timed match measures how
much search fits in a wall-clock window, so anything else on the card corrupts
it -- and not symmetrically, since the arms have different launch profiles.

    scripts/bench_streams.py --rows 64
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def timeit(fn, reps=40, warm=8):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=64)
    args = ap.parse_args()

    from identity_engine import load_nets

    policy, value, _, _ = load_nets()
    dev, dt = "cuda:0", value.dtype
    x = torch.randint(0, 31, (args.rows, 77), device=dev, dtype=torch.long)

    pm, vm = policy.model, value.model
    pc = torch.compile(pm, mode="reduce-overhead")
    vc = torch.compile(vm, mode="reduce-overhead")
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()

    def seq(a, b):
        def f():
            with torch.inference_mode(), torch.autocast("cuda", dtype=dt):
                a(x).float(); b(x).float()
        return f

    def par(a, b):
        def f():
            with torch.inference_mode():
                cur = torch.cuda.current_stream()
                s1.wait_stream(cur); s2.wait_stream(cur)
                with torch.cuda.stream(s1):
                    with torch.autocast("cuda", dtype=dt):
                        a(x).float()
                with torch.cuda.stream(s2):
                    with torch.autocast("cuda", dtype=dt):
                        b(x).float()
                cur.wait_stream(s1); cur.wait_stream(s2)
        return f

    print(f"rows={args.rows}\n")
    results = {}
    for label, fn in [
        ("sequential eager", seq(pm, vm)),
        ("sequential compiled", seq(pc, vc)),
        ("two streams eager", par(pm, vm)),
        ("two streams compiled", par(pc, vc)),
    ]:
        try:
            el = timeit(fn)
            results[label] = el
            print(f"  {label:24s} {el*1000:6.2f} ms/call")
        except Exception as exc:
            print(f"  {label:24s} FAILED: {type(exc).__name__}: {exc}")

    base = results.get("sequential eager")
    if base:
        print()
        for k, v in results.items():
            if k != "sequential eager":
                print(f"  {k:24s} {base/v:5.2f}x vs sequential eager")
        print()
        print("  A shared-trunk net would also halve the launches, for ~40 GPU-hours")
        print("  of training. If streams get close to 2x here, that is the cheap")
        print("  substitute; if they do not, the 40 hours is the honest price.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
