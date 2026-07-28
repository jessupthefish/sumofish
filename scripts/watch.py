#!/usr/bin/env python
"""Live SumoFish dashboard.

    sumofish            (or: .venv/bin/python scripts/watch.py)

Rating, the game in progress on a real board, and training, refreshing in
place. Gruvbox palette to match the desktop.

Read-only. The bot token is used for the "what am I playing" endpoint and is
never printed.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import chess
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parent.parent
USER = "SumoFish"
REFRESH = 3.0

# Gruvbox dark.
BG        = "#282828"
FG        = "#ebdbb2"
DIM       = "#928374"
LIGHT_SQ  = "#a89984"
DARK_SQ   = "#665c54"
MOVE_SQ   = "#b8bb26"   # last move
CHECK_SQ  = "#fb4934"
W_PIECE   = "#fbf1c7"
B_PIECE   = "#1d2021"
ACCENT    = "#fabd2f"
GOOD      = "#b8bb26"
BAD       = "#fb4934"
INFO      = "#83a598"

GLYPH = {
    "K": "♚", "Q": "♛", "R": "♜", "B": "♝", "N": "♞", "P": "♟",
}
SPARK = "▁▂▃▄▅▆▇█"
VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
          chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def api(path: str, token: str | None = None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        req = urllib.request.Request(f"https://lichess.org{path}", headers=headers)
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.load(r)
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None


def sparkline(values: list[float], width: int = 24) -> str:
    if len(values) < 2:
        return ""
    vals = values[-width:]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return SPARK[len(SPARK) // 2] * len(vals)
    return "".join(SPARK[int((v - lo) / (hi - lo) * (len(SPARK) - 1))] for v in vals)


def clock(seconds: int | None) -> str:
    if seconds is None:
        return "--:--"
    m, s = divmod(max(0, int(seconds)), 60)
    return f"{m}:{s:02d}"


def render_board(fen: str, flip: bool, last_move: str | None) -> Text:
    """A real board: alternating square colours, last move highlighted."""
    board = chess.Board(fen)
    hi = set()
    if last_move and len(last_move) >= 4:
        try:
            hi = {chess.parse_square(last_move[:2]), chess.parse_square(last_move[2:4])}
        except ValueError:
            pass
    checked = (
        board.king(board.turn) if board.is_check() else None
    )

    ranks = range(8) if flip else range(7, -1, -1)
    files_order = range(7, -1, -1) if flip else range(8)

    out = Text()
    for r in ranks:
        out.append(f" {r+1} ", style=f"{DIM} on {BG}")
        for f in files_order:
            sq = chess.square(f, r)
            piece = board.piece_at(sq)
            if sq == checked:
                bg = CHECK_SQ
            elif sq in hi:
                bg = MOVE_SQ
            else:
                bg = LIGHT_SQ if (f + r) % 2 else DARK_SQ
            if piece:
                fgc = W_PIECE if piece.color == chess.WHITE else B_PIECE
                glyph = GLYPH[piece.symbol().upper()]
                out.append(f" {glyph} ", style=f"bold {fgc} on {bg}")
            else:
                out.append("   ", style=f"on {bg}")
        out.append("\n")
    out.append("   ", style=f"on {BG}")
    for f in files_order:
        out.append(f" {'abcdefgh'[f]} ", style=f"{DIM} on {BG}")
    return out


def material(fen: str) -> int:
    b = chess.Board(fen)
    return sum(
        (1 if p.color == chess.WHITE else -1) * VALUES[p.piece_type]
        for p in b.piece_map().values()
    )


def rating_history() -> dict[str, list[float]]:
    log = ROOT / "logs" / "rating.jsonl"
    hist: dict[str, list[float]] = {}
    if not log.exists():
        return hist
    for line in log.read_text().splitlines()[-200:]:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        for tc, d in rec.get("ratings", {}).items():
            if d.get("rating"):
                hist.setdefault(tc, []).append(d["rating"])
    return hist


def header_panel(profile: dict) -> Panel:
    perfs = profile.get("perfs", {})
    c = profile.get("count", {})
    hist = rating_history()

    t = Table.grid(padding=(0, 2))
    t.add_column(justify="left")
    t.add_column(justify="left")
    t.add_column(justify="left")

    any_rated = False
    for tc in ("bullet", "blitz", "rapid", "classical"):
        p = perfs.get(tc)
        if not p or not p.get("games"):
            continue
        any_rated = True
        prov = "?" if p.get("prov") else ""
        spark = sparkline(hist.get(tc, []))
        trend = ""
        h = hist.get(tc, [])
        if len(h) >= 2:
            d = h[-1] - h[0]
            trend = f"[{GOOD if d >= 0 else BAD}]{d:+.0f}[/]"
        t.add_row(
            Text(f"{tc:<9}", style=DIM),
            Text(f"{p['rating']}{prov}", style=f"bold {ACCENT}"),
            Text.from_markup(f"[{INFO}]{spark}[/] {trend}  [{DIM}]{p['games']} games  RD {p.get('rd','?')}[/]"),
        )
    if not any_rated:
        t.add_row(Text("no rated games yet", style=DIM), Text(""), Text(""))

    record = Text.from_markup(
        f"[{GOOD}]{c.get('win',0)}W[/] [{DIM}]{c.get('draw',0)}D[/] [{BAD}]{c.get('loss',0)}L[/]"
        f"   [{DIM}]{c.get('all',0)} total, {c.get('rated',0)} rated[/]"
    )
    return Panel(
        Group(t, Text(""), record),
        title=f"[bold {FG}]SumoFish[/]",
        subtitle=f"[{DIM}]lichess.org/@/{USER}/tv[/]",
        border_style=DIM,
        padding=(0, 1),
    )


def game_panel(games: list[dict]) -> Panel:
    if not games:
        return Panel(
            Align.center(Text("no game in progress\nchallenging every minute", style=DIM),
                         vertical="middle"),
            title=f"[bold {FG}]board[/]", border_style=DIM, height=13,
        )
    g = games[0]
    opp = g["opponent"]
    flip = g.get("color") == "black"
    board_txt = render_board(g["fen"], flip, g.get("lastMove"))

    mat = material(g["fen"])
    ours = mat if g.get("color") == "white" else -mat
    mat_txt = (f"[{GOOD}]+{ours}[/]" if ours > 0 else
               f"[{BAD}]{ours}[/]" if ours < 0 else f"[{DIM}]level[/]")

    turn = f"[{ACCENT}]● our move[/]" if g.get("isMyTurn") else f"[{DIM}]○ waiting[/]"
    info = Text.from_markup(
        f"[{DIM}]vs[/] [bold]{opp.get('username')}[/] [{DIM}]({opp.get('rating')})[/]   "
        f"{'[bold]rated[/]' if g.get('rated') else f'[{DIM}]casual[/]'} {g.get('speed')}   "
        f"{turn}   [{DIM}]material[/] {mat_txt}"
    )
    clock_txt = Text.from_markup(
        f"[{DIM}]clock[/] [bold {ACCENT}]{clock(g.get('secondsLeft'))}[/]   "
        f"[{DIM}]lichess.org/{g['gameId']}[/]"
    )
    extra = ""
    if len(games) > 1:
        o2 = games[1]["opponent"]
        extra = f"\n[{DIM}]also playing {o2.get('username')} ({o2.get('rating')})[/]"

    return Panel(
        Group(info, Text(""), board_txt, Text(""), clock_txt,
              Text.from_markup(extra) if extra else Text("")),
        title=f"[bold {FG}]live game[/]", border_style=ACCENT, padding=(0, 1),
    )


def training_panel() -> Panel:
    try:
        info = json.loads((ROOT / "runs" / "active.json").read_text())
        log = Path(info["log"])
        lines = log.read_text().splitlines()
    except (OSError, ValueError, KeyError):
        return Panel(Text("no training run active", style=DIM),
                     title=f"[bold {FG}]training[/]", border_style=DIM)

    loss_rows = [json.loads(l) for l in lines if '"loss"' in l]
    evals = [json.loads(l) for l in lines if "puzzle_acc" in l]
    if not loss_rows:
        return Panel(Text("warming up (torch.compile)", style=DIM),
                     title=f"[bold {FG}]training[/]", border_style=DIM)

    r = loss_rows[-1]
    total = 300_000
    frac = min(1.0, r["step"] / total)
    bar = ProgressBar(total=100, completed=frac * 100, width=40,
                      style=DIM, complete_style=GOOD)

    losses = [x["loss"] for x in loss_rows[-60:]]
    accs = [x["puzzle_acc"] for x in evals]
    eta = (total - r["step"]) / max(1, r["samples_per_s"] / 1024) / 3600

    line1 = Text.from_markup(
        f"[{DIM}]{info.get('run','?')}[/]   [bold]{r['step']:,}[/][{DIM}]/{total:,}[/]  "
        f"[{DIM}]({frac*100:.0f}%)[/]   [{DIM}]eta[/] [bold]{eta:.1f}h[/]"
    )
    line2 = Text.from_markup(
        f"[{DIM}]loss[/] [bold]{r['loss']:.4f}[/] [{INFO}]{sparkline(losses)}[/]   "
        f"[{DIM}]{r['samples_per_s']:,}/s[/]"
    )
    line3 = Text.from_markup(
        f"[{DIM}]puzzles[/] [bold {ACCENT}]{accs[-1]:.3f}[/] [{INFO}]{sparkline(accs)}[/]"
        if accs else f"[{DIM}]no puzzle eval yet[/]"
    )
    return Panel(Group(line1, line2, line3, Text(""), bar),
                 title=f"[bold {FG}]training[/]", border_style=DIM, padding=(0, 1))


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

    console = Console()
    layout = Layout()
    layout.split_column(
        Layout(name="head", size=8),
        Layout(name="game", size=15),
        Layout(name="train", size=9),
    )

    profile, games, last_poll = {}, [], 0.0
    try:
        with Live(layout, console=console, refresh_per_second=4, screen=True):
            while True:
                now = time.time()
                if now - last_poll >= REFRESH:
                    profile = api(f"/api/user/{USER}") or profile
                    resp = api("/api/account/playing", token)
                    if resp is not None:
                        games = resp.get("nowPlaying", [])
                    last_poll = now
                layout["head"].update(header_panel(profile))
                layout["game"].update(game_panel(games))
                layout["train"].update(training_panel())
                time.sleep(0.25)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
