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


def _token() -> str | None:
    """The bot's token, by name only. Never printed, never logged."""
    tok = os.environ.get("LICHESS_BOT_TOKEN")
    if tok:
        return tok
    env = Path.home() / ".config/chess-gpu/bot.env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("LICHESS_BOT_TOKEN="):
                return line.split("=", 1)[1].strip() or None
    return None


def _fetch(path: str, token: str | None):
    headers = {"User-Agent": "sumofish-rating (local, read-only)",
               "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"https://lichess.org{path}", headers=headers)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def sample() -> dict | None:
    """Our own profile, from the endpoint that will actually answer.

    `/api/user/{name}` is public, and **429s from this machine even at one
    request every fifteen minutes**: lichess-bot is talking to lichess from the
    same address and the public per-IP budget is shared and small. This sampler
    used it and had been failing silently since 14:31 -- "lichess unreachable
    (HTTPError); skipping", four times an hour, for seven and a half hours --
    which meant that the one record of whether a deployment helped had no
    samples on either side of it. The dashboard already learned this and moved
    to `/api/account`, which is authenticated, returns the identical shape for
    our own account, and is budgeted separately. The public endpoint stays as a
    fallback so this still works with no token at all.
    """
    token = _token()
    attempts = [("/api/account", token)] if token else []
    attempts.append((f"/api/user/{USER}", None))
    data = None
    for path, tok in attempts:
        try:
            data = _fetch(path, tok)
            break
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            code = getattr(e, "code", None)
            print(f"[rating] {path} unavailable ({type(e).__name__}"
                  f"{f' {code}' if code else ''})")
    if data is None:
        print("[rating] no sample taken")
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


def _score_stats(win: int, draw: int, loss: int) -> tuple[int, float, float] | None:
    """Points-per-game and its standard error for a set of games.

    Score is 1 / 0.5 / 0 for win / draw / loss, so the per-game mean is the
    thing lichess is really estimating, cleaned of Glicko lag. The SE is the
    honest one for a mean of {0, 0.5, 1} outcomes -- draws cut the variance, so
    a drawish engine needs fewer games to pin its level than a 0/1 one would.
    Returned so a rating jump can be checked against its own noise before it is
    believed, which is the whole point of writing any of this down.
    """
    n = win + draw + loss
    if n == 0:
        return None
    mean = (win + 0.5 * draw) / n
    second = (win + 0.25 * draw) / n  # E[score^2]; loss contributes 0
    var = max(second - mean * mean, 0.0)
    se = (var / n) ** 0.5
    return n, mean, se


def _windows(rows: list[dict]) -> list[dict]:
    """Group consecutive samples that ran the same deployment.

    A window is one uninterrupted stretch of a single fingerprint. The games,
    wins, draws and losses that accrued *inside* it are the deployment's own
    record -- everything before the window belongs to whatever ran before.
    """
    out: list[dict] = []
    for r in rows:
        if not out or out[-1]["deployed"] != r["deployed"]:
            out.append({"deployed": r["deployed"], "first": r, "last": r})
        else:
            out[-1]["last"] = r
    return out


def deploys() -> None:
    """Per-deployment performance, so a checkpoint swap can be judged.

    `--report` shows the rating history; this answers the attribution question
    the history only hints at. For each deployment it reports the games played
    *while it was live* and the score rate on them, with a standard error, then
    compares each deployment to the one before on the same footing.
    """
    if not LOG.exists():
        print("no samples yet")
        return
    rows = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    if not rows:
        print("no samples yet")
        return

    wins = _windows(rows)
    prev = None
    for w in wins:
        a, b = w["first"], w["last"]
        bits = " ".join(f"{k}={v}" for k, v in sorted(w["deployed"].items()) if v)
        span_start = a["at"][5:16].replace("T", " ")
        span_end = b["at"][5:16].replace("T", " ")
        print(f"\n>>> {bits or '(nothing recorded)'}")
        print(f"    live {span_start} -> {span_end}  ({len(_between(rows, a, b))} samples)")

        dg = b["games"] - a["games"]
        dw = b["win"] - a["win"]
        dd = b["draw"] - a["draw"]
        dl = b["loss"] - a["loss"]
        stats = _score_stats(dw, dd, dl)
        if stats is None:
            print("    no games played in this window yet")
        else:
            n, mean, se = stats
            print(f"    {n} games in-window: {dw}-{dd}-{dl} (W-D-L)  "
                  f"score {mean*100:.1f}% +/- {se*100:.1f}%")
            if prev is not None and prev[0] is not None:
                pn, pmean, pse = prev[0]
                diff = mean - pmean
                pooled = (se * se + pse * pse) ** 0.5
                sig = abs(diff) / pooled if pooled > 0 else 0.0
                verdict = ("clear" if sig >= 2 else
                           "suggestive" if sig >= 1 else
                           "inside the noise")
                arrow = "up" if diff > 0 else "down" if diff < 0 else "flat"
                print(f"    vs previous deployment: {diff*100:+.1f}% ({arrow}), "
                      f"{sig:.1f} sigma -- {verdict}")

        for c in CONTROLS:
            ra, rb = a["ratings"].get(c), b["ratings"].get(c)
            if ra and rb:
                prov = "?" if rb["prov"] else ""
                print(f"    {c:<10} {ra['rating']} -> {rb['rating']}{prov} "
                      f"({rb['rating']-ra['rating']:+d})")
        prev = (stats, w)

    print("\n  Score rate is points-per-game (win 1, draw 1/2) on the games each")
    print("  deployment actually played -- it moves immediately, where the rating")
    print("  lags. A change under ~2 sigma is not yet distinguishable from noise.")


def _between(rows: list[dict], a: dict, b: dict) -> list[dict]:
    """Samples from a to b inclusive, by timestamp -- just for the count."""
    return [r for r in rows if a["ts"] <= r["ts"] <= b["ts"]]


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
    ap.add_argument("--deploys", action="store_true",
                    help="per-deployment score rate, to judge a checkpoint swap")
    args = ap.parse_args()
    if args.report:
        report()
        return
    if args.deploys:
        deploys()
        return
    rec = sample()
    if rec:
        bits = ", ".join(
            f"{c} {d['rating']}{'?' if d['prov'] else ''}" for c, d in rec["ratings"].items()
        )
        print(f"[rating] {rec['rated']} rated of {rec['games']} games | {bits or 'no rated games yet'}")


if __name__ == "__main__":
    main()
