#!/usr/bin/env python
"""Fit the per-node cost model the width decision rests on, from measured arms.

    scripts/cost_model.py runs/lab/sweep-2026-08-14
    scripts/cost_model.py runs/lab/sweep-2026-08-14 --armc runs/lab/armc-rep1.json

Zero GPU. Reads what `bench_search.py` wrote and does the arithmetic in one
place, so the number that decides d=384 against d=512 is derived rather than
asserted.

**The model, and why it has exactly this shape.**

    per_pass(d) = a + b * d^2

`a` is the LAUNCH term: kernel launches, the Python boundary, tokenisation,
everything that costs the same whatever the matrices are. `b * d^2` is the
COMPUTE term, because a transformer block's parameter count and its FLOPs both
go as d^2 at fixed depth. Today's engine pays TWO passes per node plus the
tree:

    today(d)  = tree + 2 * per_pass(d)
    fused(d)  = tree + 1 * per_pass(d) + (a small head)

So the prize for fusing is one whole `a`, and `a` is spent back by every
increment of `b * d^2`. **That is the entire capacity argument**: fuse and the
saved launch pays for width, up to whatever width exhausts it. Which is why the
split between `a` and `b` is the number that matters, and why a bench that
never varied the policy pass could not see it -- the historical
`t = 40.3 + 0.346*d` fit buried the whole second pass in its intercept and its
own two points imply a NEGATIVE tree time.

**What this cannot do.** It prices the cost side only. Whether a wider net plays
better per node is a separate question that only a match answers, and
PHILOSOPHY forbids converting held-out loss into Elo to shortcut it. A cost win
is a licence to run the acceptance match, not a result.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(d: Path) -> dict:
    """Every bench json under a directory, keyed by filename stem."""
    out = {}
    for f in sorted(d.glob("*.json")):
        try:
            out[f.stem] = json.loads(f.read_text())
        except Exception:                                   # noqa: BLE001
            continue
    return out


def dim_of(spec: str) -> int | None:
    """The width a shape spec names, or the preset's if it is a preset."""
    if spec.startswith("d") and spec[1:].split("L")[0].split("h")[0].isdigit():
        return int(spec[1:].split("L")[0].split("h")[0])
    return {"9M": 256, "136M": 1024, "tiny": 64}.get(spec)


def fit_quadratic(points: list[tuple[int, float]]) -> tuple[float, float, float]:
    """Least squares of `y = a + b*x^2` over (d, per_pass) pairs.

    Linear in the parameters once x^2 is the regressor, so this is an ordinary
    two-parameter fit and needs no solver. Returns (a, b, r2).
    """
    xs = [d * d for d, _ in points]
    ys = [y for _, y in points]
    if len(points) < 2:
        return float("nan"), float("nan"), float("nan")
    mx, my = st.fmean(xs), st.fmean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return float("nan"), float("nan"), float("nan")
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, (1 - ss_res / ss_tot) if ss_tot else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sweep", help="directory of bench_search --out files")
    ap.add_argument("--armc", default=None,
                    help="an Arm C decompose json, for the measured tree time "
                         "and the measured per-pass at the baseline width")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    sweep = load(Path(a.sweep))
    if not sweep:
        raise SystemExit(f"no bench json under {a.sweep}")

    # ---- the measured anchor, from Arm C ---------------------------------
    tree = per_pass_256 = None
    if a.armc:
        c = json.loads(Path(a.armc).read_text())
        u = c["us_per_node"]
        tree = u["tree_inferred"]
        per_pass_256 = (u["value_fwd"] + u["policy_fwd"]) / 2
        print(f"Arm C anchor ({c['shape']}, batch {c['batch']}):")
        print(f"  tree            {tree:7.2f} us/node")
        print(f"  per pass        {per_pass_256:7.2f} us/node  "
              f"(value {u['value_fwd']:.2f}, policy {u['policy_fwd']:.2f}, "
              f"{abs(u['value_fwd'] - u['policy_fwd']) / per_pass_256:.1%} apart)")
        print(f"  today, 2 passes {u['both']:7.2f} us/node")
        print(f"  policy share    {c['policy_share']:7.1%}  "
              f"(kill condition is under 25%)\n")

    # ---- the width family -------------------------------------------------
    # Each unfused arm gives today(d) = tree + 2*per_pass(d), so per_pass(d)
    # follows once the tree time is known. That is what Arm C is for: without
    # it the two unknowns cannot be separated, which is the historical failure.
    rows = []
    base_nps = None
    for k, v in sweep.items():
        if v.get("mode") == "decompose" or "nps" not in v:
            continue
        if base_nps is None and v.get("baseline") in v["nps"]:
            base_nps = v["nps"][v["baseline"]]
        d = dim_of(v.get("preset", ""))
        if d is None:
            continue
        us = 1e6 / v["nps"][v["preset"]]
        rows.append({"key": k, "preset": v["preset"], "d": d, "us": us,
                     "fused": k.startswith("fused"),
                     "batch": v.get("batch"), "ratio": v.get("ratio")})

    unfused = [r for r in rows if not r["fused"] and r["batch"] in (64, None)
               and "L" not in r["preset"][1:]]
    pts = []
    if tree is not None:
        pts = sorted({(r["d"], (r["us"] - tree) / 2) for r in unfused})
    if len(pts) >= 2:
        A, B, r2 = fit_quadratic(pts)
        print("per-pass fit over "
              f"{len(pts)} widths:  per_pass(d) = {A:.2f} + {B:.3e}*d^2   "
              f"R^2 = {r2:.4f}")
        print(f"  launch term at d=256: {A / (A + B * 256 ** 2):.0%} of the pass. "
              "That fraction IS the fusion prize.")
        if per_pass_256 is not None:
            pred = A + B * 256 ** 2
            print(f"  fit vs Arm C at d=256: {pred:.2f} predicted, "
                  f"{per_pass_256:.2f} measured, {pred - per_pass_256:+.2f} us\n")

        today = tree + 2 * (A + B * 256 ** 2)
        print(f"{'shape':>8} {'params':>9} {'today':>8} {'fused':>8} "
              f"{'vs today':>9} {'doublings':>10}")
        for d in (256, 320, 384, 448, 512, 576, 640):
            pp = A + B * d * d
            fused_us = tree + pp
            ratio = today / fused_us
            print(f"{'d' + str(d):>8} {12 * d * d * 8 / 1e6:>8.1f}M "
                  f"{tree + 2 * pp:>8.1f} {fused_us:>8.1f} "
                  f"{ratio:>8.2f}x {math.log2(ratio):>+10.2f}")
        print("\n  'vs today' is the fused net of that width against today's two "
              "separate\n  d=256 passes. Above 1.00x it is FASTER while being "
              "BIGGER, which is the\n  whole thesis. 'doublings' converts that "
              "into sims-doublings at a fixed clock.")
    else:
        print("not enough unfused width arms yet to fit the model "
              f"(have {len(pts)}, need 2+). Run the sweep.")

    # ---- the fused family, measured rather than modelled -------------------
    fused = sorted((r for r in rows if r["fused"]), key=lambda r: r["d"])
    if fused:
        print(f"\nMEASURED fused arms (these beat the model where they disagree):")
        print(f"{'shape':>8} {'us/node':>9} {'vs today':>9} {'doublings':>10}")
        for r in fused:
            today_us = 1e6 / base_nps if base_nps else None
            if today_us:
                ratio = today_us / r["us"]
                print(f"{r['preset']:>8} {r['us']:>9.1f} {ratio:>8.2f}x "
                      f"{math.log2(ratio):>+10.2f}")
        print("  Slightly CONSERVATIVE: the fused arm still runs the policy "
              "wrapper's own\n  tokenisation before hitting the free stub, "
              "which a real fused net pays once.")

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(
            {"tree": tree, "per_pass_baseline": per_pass_256,
             "fit": {"a": A, "b": B, "r2": r2} if len(pts) >= 2 else None,
             "rows": rows}, indent=2, default=str))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
