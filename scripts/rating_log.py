#!/usr/bin/env python
"""Record SumoFish's lichess rating over time, and what was deployed when.

The point of this file: the fun is tweaking something and watching the number
move. That only works if the number is recorded alongside *what changed*, so a
jump can be attributed rather than guessed at.

Every sample stores the rating for each time control plus a fingerprint of what
was actually running -- which checkpoints, which engine. Then `--report` shows
the rating history annotated with the deployments, so "did that help?" is a
question with an answer.

    scripts/rating_log.py             # take a sample (run from a timer)
    scripts/rating_log.py --report    # rating history + what changed when
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "rating.jsonl"
USER = "SumoFish"
CONTROLS = ("bullet", "blitz", "rapid", "classical")


def fingerprint() -> dict:
    """What is deployed right now: engine name and checkpoint identities."""
    out: dict[str, str | None] = {}
    cfg = ROOT / "config" / "lichess-bot.yml"
    if cfg.exists():
        m = re.search(r'^\s*name:\s*"(.*?)"', cfg.read_text(), re.M)
        out["engine"] = m.group(1) if m else None
    for tag in ("policy", "value", "current"):
        p = ROOT / "runs" / f"{tag}.pt"
        if p.exists():
            st = p.stat()
            # Size + mtime is a cheap identity; hashing 136MB every 15 minutes
            # would be silly and this changes only on promotion.
            out[tag] = hashlib.sha1(f"{st.st_size}:{int(st.st_mtime)}".encode()).hexdigest()[:8]
    return out


def sample() -> dict | None:
    try:
        with urllib.request.urlopen(f"https://lichess.org/api/user/{USER}", timeout=15) as r:
            data = json.load(r)
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        print(f"[rating] lichess unreachable ({type(e).__name__}); skipping")
        return None

    perfs = data.get("perfs", {})
    counts = data.get("count", {})
    rec = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ts": int(time.time()),
        "games": counts.get("all", 0),
        "rated": counts.get("rated", 0),
        "win": counts.get("win", 0),
        "loss": counts.get("loss", 0),
        "draw": counts.get("draw", 0),
        "ratings": {
            c: {
                "rating": perfs[c].get("rating"),
                "games": perfs[c].get("games", 0),
                "prov": bool(perfs[c].get("prov", False)),
                "rd": perfs[c].get("rd"),
            }
            for c in CONTROLS
            if c in perfs and perfs[c].get("games")
        },
        "deployed": fingerprint(),
    }
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def report() -> None:
    if not LOG.exists():
        print("no samples yet")
        return
    rows = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    if not rows:
        print("no samples yet")
        return

    print(f"{'when':<17} {'games':>6} {'W-D-L':>12}  " + "  ".join(f"{c[:5]:>7}" for c in CONTROLS))
    print("-" * 78)
    prev_deploy = None
    for r in rows:
        if r["deployed"] != prev_deploy:
            bits = " ".join(f"{k}={v}" for k, v in sorted(r["deployed"].items()) if v)
            print(f"  >>> deployed: {bits}")
            prev_deploy = r["deployed"]
        when = r["at"][5:16].replace("T", " ")
        wdl = f"{r['win']}-{r['draw']}-{r['loss']}"
        cells = []
        for c in CONTROLS:
            d = r["ratings"].get(c)
            cells.append(f"{d['rating']}{'?' if d['prov'] else ' '}".rjust(7) if d else " " * 7)
        print(f"{when:<17} {r['games']:>6} {wdl:>12}  " + "  ".join(cells))

    first, last = rows[0], rows[-1]
    print()
    for c in CONTROLS:
        a, b = first["ratings"].get(c), last["ratings"].get(c)
        if a and b:
            print(f"  {c:<10} {a['rating']} -> {b['rating']}  ({b['rating']-a['rating']:+d}) "
                  f"over {b['games']-a['games']} games"
                  + ("  [still provisional]" if b["prov"] else ""))
    print("\n  '?' means provisional -- lichess is still unsure and moves the number in big steps.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report()
        return
    rec = sample()
    if rec:
        bits = ", ".join(
            f"{c} {d['rating']}{'?' if d['prov'] else ''}" for c, d in rec["ratings"].items()
        )
        print(f"[rating] {rec['rated']} rated of {rec['games']} games | {bits or 'no rated games yet'}")


if __name__ == "__main__":
    main()
