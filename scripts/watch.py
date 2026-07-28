#!/usr/bin/env python
"""Live dashboard. Leave it running in a spare terminal.

    scripts/watch.py

Shows, refreshing every few seconds: the rating per time control, the current
game with a board, and how training is progressing. Everything you would
otherwise be checking by hand in three different places.

Read-only. Uses the bot token for the "what am I playing" endpoint only, and
never prints it.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USER = "SumoFish"
REFRESH = 5

PIECES = {
    "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
    "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
}
DIM, BOLD, GREEN, RED, YELLOW, RESET = "\033[2m", "\033[1m", "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def api(path: str, token: str | None = None) -> dict | list | None:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        req = urllib.request.Request(f"https://lichess.org{path}", headers=headers)
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.load(r)
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None


def board_lines(fen: str, flip: bool) -> list[str]:
    rows = fen.split()[0].split("/")
    if flip:
        rows = [r[::-1] for r in reversed(rows)]
    out = []
    for i, row in enumerate(rows):
        rank = (i + 1) if flip else (8 - i)
        cells = []
        for ch in row:
            if ch.isdigit():
                cells.extend("·" * int(ch))
            else:
                cells.append(PIECES.get(ch, ch))
        out.append(f" {rank} " + " ".join(cells))
    files = "hgfedcba" if flip else "abcdefgh"
    out.append("   " + " ".join(files))
    return out


def training_status() -> str:
    active = ROOT / "runs" / "active.json"
    try:
        log = Path(json.loads(active.read_text())["log"])
        rows = [json.loads(l) for l in log.read_text().splitlines() if '"loss"' in l]
        evals = [json.loads(l) for l in log.read_text().splitlines() if "puzzle_acc" in l]
    except (OSError, ValueError, KeyError):
        return f"{DIM}no training run active{RESET}"
    if not rows:
        return f"{DIM}training warming up{RESET}"
    r = rows[-1]
    pz = f"  puzzles {evals[-1]['puzzle_acc']:.3f}" if evals else ""
    pct = r["step"] / 300_000 * 100
    return (f"step {r['step']:,}/300,000 ({pct:.0f}%)  loss {r['loss']:.4f}  "
            f"{r['samples_per_s']:,}/s{pz}")


def main() -> None:
    token = os.environ.get("LICHESS_BOT_TOKEN")
    if not token:
        env = Path.home() / ".config/chess-gpu/bot.env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("LICHESS_BOT_TOKEN="):
                    token = line.split("=", 1)[1].strip() or None
    if not token:
        sys.exit("LICHESS_BOT_TOKEN not set (see ~/.config/chess-gpu/bot.env)")

    try:
        while True:
            width = shutil.get_terminal_size((80, 24)).columns
            profile = api(f"/api/user/{USER}") or {}
            playing = (api("/api/account/playing", token) or {}).get("nowPlaying", [])

            out = ["\033[H\033[J"]
            out.append(f"{BOLD}SumoFish{RESET}  {DIM}{datetime.now():%H:%M:%S}   "
                       f"lichess.org/@/{USER}/tv{RESET}")
            out.append("─" * min(width, 72))

            c = profile.get("count", {})
            perfs = profile.get("perfs", {})
            bits = []
            for tc in ("bullet", "blitz", "rapid", "classical"):
                p = perfs.get(tc)
                if p and p.get("games"):
                    bits.append(f"{tc[:3]} {BOLD}{p['rating']}{RESET}"
                                f"{'?' if p.get('prov') else ''}")
            out.append(f"  {'  '.join(bits) if bits else DIM + 'unrated' + RESET}")
            out.append(f"  {c.get('all',0)} games  "
                       f"{GREEN}{c.get('win',0)}W{RESET} {c.get('draw',0)}D "
                       f"{RED}{c.get('loss',0)}L{RESET}   rated: {c.get('rated',0)}")
            out.append("")

            if not playing:
                out.append(f"  {DIM}no game right now (challenges every minute){RESET}")
            for g in playing[:2]:
                opp = g["opponent"]
                turn = f"{YELLOW}our move{RESET}" if g.get("isMyTurn") else f"{DIM}waiting{RESET}"
                out.append(f"  vs {BOLD}{opp.get('username')}{RESET} ({opp.get('rating')})  "
                           f"{g.get('speed')} {'rated' if g.get('rated') else 'casual'}  {turn}")
                out.append(f"  {DIM}lichess.org/{g['gameId']}{RESET}")
                if g.get("fen"):
                    out += board_lines(g["fen"], flip=(g.get("color") == "black"))
                out.append("")

            out.append("─" * min(width, 72))
            out.append(f"  {training_status()}")
            out.append(f"\n  {DIM}ctrl-c to quit{RESET}")
            print("\n".join(out), flush=True)
            time.sleep(REFRESH)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
