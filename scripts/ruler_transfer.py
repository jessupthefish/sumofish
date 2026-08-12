#!/usr/bin/env python
"""Does the Stockfish-vs-Stockfish ruler transfer to SumoFish? Measured: NO.

    scripts/ruler_transfer.py            # print the check
    scripts/ruler_transfer.py --json     # machine-readable

`sim_ladder.py` puts every rung on an absolute scale by adding a WALK ALONG THE
RULER: `absolute = rung_elo + (ruler(N) - ruler(700))`, where the ruler is
measured Stockfish against Stockfish. That step assumes Elo is transitive
across the two populations -- that a node budget worth X Elo to Stockfish is
worth the same X Elo to SumoFish.

This script tests the assumption, which nothing did until 2026-08-11. It needs
two anchors: the same SumoFish configuration measured directly against
Stockfish at two different node budgets. The ruler predicts their difference.

The answer, on the 700/1600 pair at 400 sims:

    ruler, Stockfish vs Stockfish   280.8 +-16.4 Elo
    the same span through SumoFish  192.8 +-18.0 Elo

which is a transfer factor of 0.69 and a difference of 88.0 +-24.4, z = 7.1.
The ladder's absolutes chain onto a scale SumoFish does not experience.

The mechanism is not settled, but the two populations are not comparable
matches in the first place: SF-vs-SF rungs draw 7-13% of games and end 74-87%
by arbiter adjudication, while SumoFish-vs-SF anchors draw 28-38% and adjudicate
30-35%. Elo inferred from a score is draw-rate dependent, so a decisive
population and a drawish one are not measuring on the same scale even when
both are correct about who is stronger.

What this does NOT touch: the anchors themselves. `SumoFish@400 is +30.5 +-12
on Stockfish@700 nodes` is a direct measurement of a fixed external opponent
with no chain in it, and it stands. What it does touch is every number that
walks the ruler, which is `scale_D`, `scale_bar`, and four of the five ladder
absolutes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import sim_ladder as SL  # noqa: E402

M = ROOT / "runs" / "matches"
KEYS = ("sims", "c_puct_init", "fpu", "vloss_fix", "core", "value", "policy")


def anchors() -> list[dict]:
    """Every SumoFish-vs-Stockfish-at-fixed-nodes anchor on disk."""
    out = []
    for d in sorted(M.glob("stockfish-anchor-*nodes")):
        cfg_p, st_p = d / "config.json", d / "status.json"
        if not (cfg_p.exists() and st_p.exists()):
            continue
        cfg, st = json.loads(cfg_p.read_text()), json.loads(st_p.read_text())
        a, b = cfg.get("a", {}), cfg.get("b", {})
        if a.get("stockfish_nodes") is not None or b.get("stockfish_nodes") is None:
            continue                      # A must be SumoFish, B must be Stockfish
        out.append({
            "name": d.name, "nodes": int(b["stockfish_nodes"]),
            "elo": st["elo"], "err": st["err"], "games": st["games"],
            "draws": st["d"] / st["games"],
            # The engine under test has to be the SAME engine on both sides of a
            # comparison, or the difference is two effects and not one. This is
            # the 2026-08-11 "+30.5, was +44.4" mistake, which compared a v5 warm
            # run with a v6 fixed one and read the sum as a harness effect.
            "arm": tuple((k, str(a.get(k))) for k in KEYS),
            "code": cfg.get("code"),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    nodes_map, coefs, errs, prov = SL.ruler_curve()
    if not nodes_map:
        print("no ruler on disk; nothing to check")
        return 1

    found = anchors()
    groups: dict[tuple, list[dict]] = {}
    for a in found:
        groups.setdefault(a["arm"], []).append(a)

    report = {"ruler_provenance": prov, "pairs": []}
    for arm, arms in groups.items():
        arms.sort(key=lambda a: a["nodes"])
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                lo, hi = arms[i], arms[j]
                r_lo, r_hi = SL.ruler_at(nodes_map, lo["nodes"], coefs), \
                    SL.ruler_at(nodes_map, hi["nodes"], coefs)
                if r_lo is None or r_hi is None:
                    continue              # outside the measured ruler; refuse
                gap_ruler = r_hi[0] - r_lo[0]
                err_ruler = SL._combine(SL._sub(r_hi[1], r_lo[1]), errs)
                gap_seen = lo["elo"] - hi["elo"]
                err_seen = math.hypot(lo["err"], hi["err"])
                diff = gap_ruler - gap_seen
                differr = math.hypot(err_ruler, err_seen)
                report["pairs"].append({
                    "arm": dict(arm), "lo_nodes": lo["nodes"], "hi_nodes": hi["nodes"],
                    "doublings": math.log2(hi["nodes"] / lo["nodes"]),
                    "gap_ruler": round(gap_ruler, 1), "gap_ruler_err": round(err_ruler, 1),
                    "gap_through_sumofish": round(gap_seen, 1),
                    "gap_through_sumofish_err": round(err_seen, 1),
                    "difference": round(diff, 1), "difference_err": round(differr, 1),
                    "z": round(diff / (differr / 1.96), 1),
                    "transfer_factor": round(gap_seen / gap_ruler, 3),
                    "draws_lo": round(lo["draws"], 3), "draws_hi": round(hi["draws"], 3),
                    "n_lo": lo["games"], "n_hi": hi["games"],
                })

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"ruler: {prov}\n")
    if not report["pairs"]:
        print("Fewer than two anchors share a configuration, so the ruler's\n"
              "transfer to SumoFish is UNTESTED. Every ladder absolute rests on\n"
              "it. Run a second anchor at a different node budget.")
        return 1

    for p in report["pairs"]:
        print(f"SumoFish@{p['arm']['sims']} sims, "
              f"c_puct_init={p['arm']['c_puct_init']} fpu={p['arm']['fpu']}")
        print(f"  SF@{p['lo_nodes']}n -> SF@{p['hi_nodes']}n "
              f"({p['doublings']:.3f} doublings of Stockfish nodes)")
        print(f"    Stockfish vs Stockfish (the ruler) : "
              f"{p['gap_ruler']:7.1f} +-{p['gap_ruler_err']:.1f}")
        print(f"    the same span through SumoFish     : "
              f"{p['gap_through_sumofish']:7.1f} +-{p['gap_through_sumofish_err']:.1f}"
              f"   (n={p['n_lo']}, {p['n_hi']})")
        print(f"    difference                         : "
              f"{p['difference']:7.1f} +-{p['difference_err']:.1f}   z = {p['z']}")
        print(f"    transfer factor                    : {p['transfer_factor']:.2f}")
        print(f"    draw rates                         : "
              f"{p['draws_lo']:.1%} and {p['draws_hi']:.1%} vs the ruler's 7-13%")
        verdict = ("TRANSFERS (consistent with 1.0)" if abs(p["z"]) < 2
                   else "DOES NOT TRANSFER -- every ladder absolute inherits this")
        print(f"    {verdict}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
