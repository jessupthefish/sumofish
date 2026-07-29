#!/usr/bin/env python
"""What happened while you were asleep.

    sumofish-games                 the last 25 games, newest first
    sumofish-games --since 12h     everything since then
    sumofish-games --all           the whole history
    sumofish-games --losses        only the ones it lost, for reading afterwards

lichess-bot already writes every finished game to `logs/games/` with the result,
the opponent, both ratings, the time control, the opening and how it ended
(`pgn_directory` in the config). Nothing here collects anything; it reads that
file and answers the question you actually have in the morning, which is not
"what is the rating" but "did it win, against whom, and how did it lose".

Deliberately a separate view rather than a panel in `sumofish`. The dashboard is
for a game in progress and is a fixed-height layout with a board in it; this is
a scrollable list of history, and the two want different shapes. It also means
this can be read over SSH from a phone without a terminal that does sixel.

The bot's own name is inferred as whichever player appears in the most games,
so nothing has to be configured and it keeps working if the account is renamed.
`--me NAME` overrides it.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import chess.pgn
from rich.console import Console
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parent.parent
GAMES = ROOT / "logs" / "games"
RATING = ROOT / "logs" / "rating.jsonl"

WIN, LOSS, DRAW = "green", "red", "yellow"


def parse_since(text: str) -> datetime | None:
    """'12h', '3d', '90m', or an ISO date."""
    if not text:
        return None
    m = re.fullmatch(r"(\d+)([hdm])", text.strip().lower())
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = {"h": timedelta(hours=n), "d": timedelta(days=n),
                 "m": timedelta(minutes=n)}[unit]
        return datetime.now(timezone.utc) - delta
    try:
        return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
    except ValueError:
        raise SystemExit(f"could not read --since {text!r}; try 12h, 3d, or 2026-07-28")


def load_games() -> list[dict]:
    """Every finished game lichess-bot has written, flattened to what matters."""
    out: list[dict] = []
    for path in sorted(GAMES.glob("*.pgn")):
        with path.open() as fh:
            while (game := chess.pgn.read_game(fh)) is not None:
                h = game.headers
                when = None
                if h.get("UTCDate", "?") != "?" and h.get("UTCTime", "?") != "?":
                    try:
                        when = datetime.strptime(
                            f"{h['UTCDate']} {h['UTCTime']}", "%Y.%m.%d %H:%M:%S"
                        ).replace(tzinfo=timezone.utc)
                    except ValueError:
                        pass
                # Ply count without replaying: the mainline is already parsed.
                plies = sum(1 for _ in game.mainline_moves())
                out.append({
                    "when": when, "white": h.get("White", "?"), "black": h.get("Black", "?"),
                    "white_elo": h.get("WhiteElo", "?"), "black_elo": h.get("BlackElo", "?"),
                    "result": h.get("Result", "*"), "event": h.get("Event", ""),
                    "tc": h.get("TimeControl", "?"), "opening": h.get("Opening", "?"),
                    "eco": h.get("ECO", ""), "termination": h.get("Termination", "?"),
                    "site": h.get("Site", ""), "plies": plies,
                })
    out.sort(key=lambda g: g["when"] or datetime.min.replace(tzinfo=timezone.utc))
    return out


def infer_me(games: list[dict]) -> str:
    counts: collections.Counter = collections.Counter()
    for g in games:
        counts[g["white"]] += 1
        counts[g["black"]] += 1
    return counts.most_common(1)[0][0] if counts else "SumoFish"


def outcome(game: dict, me: str) -> tuple[str, float]:
    """('win'|'loss'|'draw'|'unfinished', score from our point of view)."""
    we_are_white = game["white"] == me
    if game["result"] == "1/2-1/2":
        return "draw", 0.5
    if game["result"] == "1-0":
        return ("win", 1.0) if we_are_white else ("loss", 0.0)
    if game["result"] == "0-1":
        return ("loss", 0.0) if we_are_white else ("win", 1.0)
    return "unfinished", 0.5


def rating_change(since: datetime | None) -> str:
    """What the rating did over the window, from the sampler's own log."""
    if not RATING.exists():
        return ""
    rows = []
    for line in RATING.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not rows:
        return ""
    window = [r for r in rows
              if since is None or datetime.fromtimestamp(r["ts"], timezone.utc) >= since]
    if len(window) < 2:
        window = rows[-2:]
    first, last = window[0].get("ratings", {}), window[-1].get("ratings", {})
    parts = []
    for kind in ("bullet", "blitz", "rapid", "classical"):
        if kind in last:
            now = last[kind]["rating"]
            was = first.get(kind, {}).get("rating")
            arrow = f" ({now - was:+d})" if was is not None and was != now else ""
            prov = "?" if last[kind].get("prov") else ""
            parts.append(f"{kind} {now}{prov}{arrow}")
    return "   ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-n", "--limit", type=int, default=25)
    ap.add_argument("--since", default=None, help="12h, 3d, 90m, or 2026-07-28")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--losses", action="store_true", help="only games it lost")
    ap.add_argument("--me", default=None, help="override the inferred bot name")
    ap.add_argument("--links", action="store_true", help="show the lichess game ids")
    args = ap.parse_args()

    console = Console()
    games = load_games()
    if not games:
        console.print(f"[dim]no games in {GAMES}[/dim]")
        return

    me = args.me or infer_me(games)
    since = parse_since(args.since) if args.since else None
    if since:
        games = [g for g in games if g["when"] and g["when"] >= since]

    scored = [(g, *outcome(g, me)) for g in games]
    if args.losses:
        scored = [s for s in scored if s[1] == "loss"]

    shown = scored if args.all else scored[-args.limit:]
    shown = list(reversed(shown))                       # newest first

    table = Table(box=None, pad_edge=False, expand=False)
    for col, just in (("when", "left"), ("", "center"), ("opponent", "left"),
                      ("as", "center"), ("control", "right"), ("opening", "left"),
                      ("ended", "left"), ("mv", "right")):
        table.add_column(col, justify=just, no_wrap=True)
    if args.links:
        table.add_column("link", justify="left", no_wrap=True)

    for game, verdict, _score in shown:
        we_are_white = game["white"] == me
        them = game["black"] if we_are_white else game["white"]
        their_elo = game["black_elo"] if we_are_white else game["white_elo"]
        mark = {"win": Text("W", style=WIN), "loss": Text("L", style=LOSS),
                "draw": Text("D", style=DRAW)}.get(verdict, Text("*", style="dim"))
        when = game["when"].astimezone().strftime("%a %H:%M") if game["when"] else "?"
        # "rated blitz game" -> "rated". The control is its own column.
        kind = "rated" if "rated" in game["event"].lower() else "casual"
        table.add_row(
            Text(when, style="dim"), mark,
            f"{them} {their_elo}" if their_elo != "?" else them,
            "W" if we_are_white else "B",
            game["tc"],
            Text(game["opening"][:30], style="dim"),
            Text(f"{game['termination'].lower()} / {kind}", style="dim"),
            str(game["plies"] // 2),
            *([Text(game["site"].replace("https://lichess.org/", ""), style="dim")]
              if args.links else []),
        )

    w = sum(1 for _, v, _ in scored if v == "win")
    d = sum(1 for _, v, _ in scored if v == "draw")
    lo = sum(1 for _, v, _ in scored if v == "loss")
    n = w + d + lo
    label = f"since {args.since}" if since else "all time"

    console.print(table)
    console.print()
    if n:
        pct = (w + 0.5 * d) / n * 100
        console.print(
            Text.assemble(
                (f"  {n} games {label}   ", "bold"),
                (f"{w}W", WIN), " ", (f"{d}D", DRAW), " ", (f"{lo}L", LOSS),
                (f"   scoring {pct:.0f}%", "bold"),
            )
        )
    # Break it down: an aggregate hides that it is fine at one control and
    # losing at another, which is the whole reason the time control changed.
    by_tc: collections.Counter = collections.Counter()
    tally: dict[str, list[int]] = {}
    for game, verdict, _ in scored:
        key = game["tc"]
        by_tc[key] += 1
        row = tally.setdefault(key, [0, 0, 0])
        row[{"win": 0, "draw": 1, "loss": 2}.get(verdict, 1)] += 1
    if len(tally) > 1:
        console.print()
        for key, (kw, kd, kl) in sorted(tally.items(), key=lambda kv: -by_tc[kv[0]]):
            total = kw + kd + kl
            console.print(Text.assemble(
                (f"    {key:>10}  ", "dim"),
                (f"{kw}W", WIN), " ", (f"{kd}D", DRAW), " ", (f"{kl}L", LOSS),
                (f"   {(kw + 0.5 * kd) / total * 100:.0f}%", "dim"),
            ))
    ratings = rating_change(since)
    if ratings:
        console.print(f"\n  [dim]rating[/dim]  {ratings}")


if __name__ == "__main__":
    main()
