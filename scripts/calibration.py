#!/usr/bin/env python
"""Is the value head CORRECT, or merely CONFIDENT?

    scripts/calibration.py                        # every match with curves
    scripts/calibration.py --match stockfish-anchor-700nodes
    scripts/calibration.py --json runs/lab/calibration.json

Zero GPU. Every archived game already stores `curve`, the per-ply win
probability in White's frame, and `result`. That is everything needed, and it
had been sitting unused in ~60,000 games.

**Read the RESIDUAL column, never the BIAS column.** That is the whole lesson
of this file and it cost a Council finding.

Raw bias is `mean(predicted) - mean(realised)` over an engine's plies. It looks
like a calibration statistic and it is not one: an evaluation function scores
the POSITION, while the realised result also contains who was holding the
pieces afterwards. So the weaker side of any pairing is "overconfident" and the
stronger side is "underconfident" by construction, whatever either one's
calibration actually is. Measured over 138 engine-rows in 69 archived matches:

    bias = -0.4876 * (score - 0.5) + 0.0017      R^2 = 0.90

Ninety percent of the variance is the match result. The residual is what is
left, its archive-wide spread is 0.026, and the difference between Stockfish
rows and SumoFish rows in it is **+0.0004 +-0.0089**, i.e. nothing.

**Retracted 2026-08-14: "SumoFish is overconfident by +0.021 at parity and
+0.126 when outclassed."** Both numbers are the regression line.
`stockfish-anchor-1600nodes` reads bias +0.0963 where the score alone predicts
+0.1110, a residual of **-0.0147**: against a stronger opponent SumoFish is
marginally *better* calibrated than the pairing effect accounts for, which is
the opposite of the claim. The "Stockfish is underconfident in the same games"
control does not rescue it either, because the two sides of one match are
near-exact mirror images of each other by the same construction:
`ruler-1400-vs-2800` is Stockfish against Stockfish, one evaluation function,
and reads +0.1619 / -0.1602. Full working in LAB-NOTES.md, 2026-08-14.

The +-0.0089 above is the null for a difference of GROUP MEANS over ~65 rows
each. A single engine-row is judged against the spread of the residuals
themselves, roughly +-0.05, which is printed per run. Do not read a row against
the group null.

**What survives, and it is worth having.** The residual is a commissioned
instrument: it reads zero across the whole archive, so a candidate whose
residual leaves the printed row band has done something the match result does
not explain. Use it that way, as a screen with a null attached, and never quote
a raw bias again. It is a coarse screen: at +-0.05 per row it will only catch a
gross confidence pathology, which is exactly what it is for.

**What it still cannot do, so do not ask it to.** It cannot gate a promotion on
its own. A candidate that wins its acceptance match gets a negative bias for
winning, and the residual is a difference of two noisy means over positions the
two arms did not share. The non-circular version of this measurement is
calibration against the HELD-OUT set, where the label does not depend on who
was playing; that lives in `scripts/eval_heldout.py`.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}

# Two different nulls, and confusing them is the easy mistake here.
#
# GROUP null: comparing the MEAN residual of one family of engines against
# another's. Stockfish rows vs SumoFish rows over the archive differ by
# +0.0004 +-0.0089, which is what refutes the "SumoFish is overconfident"
# claim. It averages 60-70 rows and is correspondingly tight.
#
# ROW null: judging ONE engine-row. That is the spread of the residuals
# themselves, ~+-0.05, computed per run from the fit rather than hardcoded so
# it tracks whatever population is in front of it. Quoting the group null on a
# single row stars almost every row in the archive and means nothing.
GROUP_NULL_95 = 0.0089


def per_engine(d: Path) -> dict | None:
    """Split every ply's prediction by which engine was to move."""
    log = d / "games.jsonl"
    if not log.exists():
        return None
    cfg = {}
    if (d / "config.json").exists():
        try:
            cfg = json.loads((d / "config.json").read_text())
        except Exception:
            cfg = {}
    stats = {"a": {"n": 0, "sum_p": 0.0, "sum_y": 0.0, "brier": 0.0, "bins": {}},
             "b": {"n": 0, "sum_p": 0.0, "sum_y": 0.0, "brier": 0.0, "bins": {}}}
    games = 0
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        try:
            g = json.loads(line)
        except Exception:
            continue
        curve, res = g.get("curve"), g.get("result")
        if not curve or res not in RESULT:
            continue
        games += 1
        y_white = RESULT[res]
        # `a_white` says which arm had White. Ply i after the book is played by
        # White when i is even, so the mover alternates from there.
        a_white = bool(g.get("a_white"))
        for i, p_white in enumerate(curve):
            mover_is_white = (i % 2 == 0)
            side = "a" if (mover_is_white == a_white) else "b"
            # Convert to the MOVER's frame: the value head predicts the win
            # probability of the side to move, and `curve` is stored in White's.
            p = p_white if mover_is_white else 1.0 - p_white
            y = y_white if mover_is_white else 1.0 - y_white
            s = stats[side]
            s["n"] += 1
            s["sum_p"] += p
            s["sum_y"] += y
            s["brier"] += (p - y) ** 2
            b = min(9, int(p * 10))
            bp, by, bn = s["bins"].get(b, (0.0, 0.0, 0))
            s["bins"][b] = (bp + p, by + y, bn + 1)
    if not games:
        return None

    out = {"match": d.name, "games": games,
           "labels": {"a": cfg.get("a", {}).get("label", "A"),
                      "b": cfg.get("b", {}).get("label", "B")}}
    for side in ("a", "b"):
        s = stats[side]
        if not s["n"]:
            continue
        n = s["n"]
        mean_p, mean_y = s["sum_p"] / n, s["sum_y"] / n
        # Expected calibration error: mean |predicted - realised| per decile,
        # weighted by occupancy. Bias is the signed version and is the one that
        # says "overconfident" rather than "noisy".
        ece = sum(abs(bp / bn - by / bn) * bn for bp, by, bn in s["bins"].values()) / n
        out[side] = {"plies": n, "brier": round(s["brier"] / n, 4),
                     "mean_predicted": round(mean_p, 4),
                     "mean_realised": round(mean_y, 4),
                     "bias": round(mean_p - mean_y, 4), "ece": round(ece, 4)}
    return out


def fit_pairing_effect(rows: list[dict]) -> tuple[float, float, float, float]:
    """Least squares `bias = slope * (score - 0.5) + intercept`.

    Fitted over whatever rows are in front of it rather than hardcoded, so the
    correction stays honest as the archive grows. Returns (slope, intercept,
    R^2, row_null_95). This is the confounder, not a result: it exists to be
    subtracted.

    Note the two rows of a match are near-mirror images (one evaluation of one
    game, read from both ends), so a match contributes roughly one independent
    residual, not two. Do not treat the row count as a sample size.
    """
    pts = [(r[s]["mean_realised"] - 0.5, r[s]["bias"])
           for r in rows for s in ("a", "b") if s in r]
    if len(pts) < 3:
        # Fall back to the measured archive-wide line rather than fitting two
        # points, which would drive every residual to exactly zero and hide
        # the very thing the column exists to show.
        return -0.4876, 0.00165, float("nan"), 0.0518
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    mx, my = st.fmean(xs), st.fmean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return -0.4876, 0.00165, float("nan"), 0.0518
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    intercept = my - slope * mx
    resid = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum(r * r for r in resid)
    r2 = (1 - ss_res / ss_tot) if ss_tot else float("nan")
    return slope, intercept, r2, 1.96 * st.pstdev(resid)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--match", default=None, help="one match directory name")
    ap.add_argument("--min-games", type=int, default=100)
    ap.add_argument("--json", default=None)
    ap.add_argument("--fit-from", default=None, metavar="GLOB_SUBSTR",
                    help="fit the pairing line on matches whose name contains "
                         "this, then apply it to everything. Use when scoring "
                         "one new match against the historical line.")
    a = ap.parse_args()

    root = ROOT / "runs/matches"
    dirs = [root / a.match] if a.match else sorted(p for p in root.iterdir() if p.is_dir())
    rows = [r for r in (per_engine(d) for d in dirs) if r and r["games"] >= a.min_games]

    # The line must be fitted on a population, so a single --match run scores
    # itself against every other archived match rather than against itself.
    if a.match or a.fit_from:
        pool = [r for r in (per_engine(p) for p in sorted(root.iterdir()) if p.is_dir())
                if r and r["games"] >= a.min_games
                and (a.fit_from is None or a.fit_from in r["match"])]
    else:
        pool = rows
    slope, intercept, r2, row_null = fit_pairing_effect(pool)

    for r in rows:
        for side in ("a", "b"):
            if side not in r:
                continue
            s = r[side]
            s["predicted_bias"] = round(slope * (s["mean_realised"] - 0.5) + intercept, 4)
            s["residual"] = round(s["bias"] - s["predicted_bias"], 4)
    rows.sort(key=lambda r: -abs(r.get("a", {}).get("residual", 0)))

    print(f"pairing effect fitted on {len(pool)} engine-rows: "
          f"bias = {slope:+.4f}*(score-0.5) {intercept:+.5f}   R^2 = {r2:.3f}")
    print(f"RESIDUAL is the column to read. Row null +-{row_null:.4f} (95%, the "
          f"spread of the fit); group-mean null +-{GROUP_NULL_95:.4f}.\n")
    print(f"{'match':<34} {'engine':<22} {'plies':>7} {'Brier':>7} "
          f"{'pred':>7} {'real':>7} {'bias':>8} {'RESID':>8} {'ECE':>6}")
    for r in rows:
        for side in ("a", "b"):
            if side not in r:
                continue
            s = r[side]
            flag = " *" if abs(s["residual"]) > row_null else ""
            print(f"{r['match'][:33]:<34} {r['labels'][side][:21]:<22} "
                  f"{s['plies']:>7,} {s['brier']:>7.4f} {s['mean_predicted']:>7.4f} "
                  f"{s['mean_realised']:>7.4f} {s['bias']:>+8.4f} "
                  f"{s['residual']:>+8.4f}{flag} {s['ece']:>6.4f}")
    print("\nbias > 0 means the engine predicted more wins than it got. It is NOT a")
    print("calibration statistic: 90% of it is the match result, because an eval")
    print("scores the position and the result also contains who played it out.")
    print("RESIDUAL is bias with that pairing effect subtracted. A '*' marks a row")
    print("outside the archive-wide null; those are the only rows worth reading.")

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(
            {"fit": {"slope": slope, "intercept": intercept, "r2": r2,
                     "rows_fitted": len(pool), "row_null_95": row_null,
                     "group_null_95": GROUP_NULL_95},
             "matches": rows}, indent=2))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
