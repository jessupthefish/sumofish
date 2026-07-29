"""Turning state into panels. Pure functions: state in, renderable out.

Nothing here does I/O, sleeps, or blocks. That separation is what lets the
render loop run at a steady frame rate while a lichess request is timing out
somewhere else, and it is why a source failing degrades one panel instead of
freezing the screen.

Layout order is a claim about what matters. The board and what the engine is
thinking about it sit together at the top, because that is the thing being
watched. Ratings, training and machine health are context and sit below. The
event tape is last, because it answers a question asked after the fact.
"""

from __future__ import annotations

import time

import chess
from rich.align import Align
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import board as boardmod
from .sources import GATE
from .theme import (
    ACCENT, BAD, BG, COOL, DIM, FAINT, FG, GAUGE_MAX, GOOD, INFO, PLUM, WARM,
)
from .widgets import bar, curve_chart, evalbar, ladder, sparkline, track_tag

# Cells across for the evaluation gauge. The player lines reserve the same
# width for their percentages, so the two line up as one column.
EVAL_WIDTH = 3

VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
          chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
FIGURES = {chess.PAWN: "♟", chess.KNIGHT: "♞", chess.BISHOP: "♝",
           chess.ROOK: "♜", chess.QUEEN: "♛"}


def _panel(body, title: str, border: str = FAINT, subtitle: str | None = None):
    return Panel(body, title=f"[{FG}]{title}[/]", title_align="left",
                 subtitle=subtitle, subtitle_align="right",
                 border_style=border, padding=(0, 1), style=f"on {BG}")


def clockstr(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    seconds = max(0.0, seconds)
    m, s = divmod(int(seconds), 60)
    if m >= 60:
        return f"{m // 60}:{m % 60:02d}:{s:02d}"
    # Under ten seconds the tenths are the only thing that matters.
    return f"{m}:{s:02d}.{int((seconds % 1) * 10)}" if seconds < 10 else f"{m}:{s:02d}"


def live_clocks(game: dict) -> tuple[float | None, float | None]:
    """Both clocks, ticking.

    lichess sends the clocks only when a move is played, so a display that
    just reprints them stutters between moves. Counting down locally fixes
    that, and resyncing to the server number on every message is what stops it
    drifting into disagreeing with lichess's own page.
    """
    wc, bc = game.get("wc"), game.get("bc")
    if wc is None or bc is None:
        return wc, bc
    elapsed = time.time() - game.get("clock_at", time.time())
    board = game.get("board")
    if board is not None and board.move_stack:
        if board.turn == chess.WHITE:
            wc = wc - elapsed
        else:
            bc = bc - elapsed
    return wc, bc


# ---- header ---------------------------------------------------------------

def header(state, user: str, width: int):
    prof = state.get("profile", {})
    f = state.field("profile")
    perfs = prof.get("perfs", {})
    counts = prof.get("count", {})
    hist = _rating_history(state)

    grid = Table.grid(padding=(0, 2), expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="left")

    left = Text(no_wrap=True)
    left.append(" SUMOFISH ", style=f"bold {BG} on {ACCENT}")
    left.append("  ")
    shown = 0
    for tc in ("bullet", "blitz", "rapid", "classical"):
        p = perfs.get(tc)
        if not p or not p.get("games"):
            continue
        shown += 1
        left.append(f"{tc[:3]} ", style=f"{DIM} on {BG}")
        left.append(f"{p['rating']}", style=f"bold {ACCENT} on {BG}")
        left.append("?" if p.get("prov") else " ", style=f"{DIM} on {BG}")
        series = hist.get(tc, [])
        if len(series) >= 2:
            spark, lo, hi = sparkline(series, width=18, style=INFO)
            left.append_text(spark)
            delta = series[-1] - series[0]
            left.append(f" {delta:+.0f}",
                        style=f"{GOOD if delta >= 0 else BAD} on {BG}")
        left.append(f"  rd{p.get('rd','?')}  {p.get('games')}g   ",
                    style=f"{DIM} on {BG}")
    if not shown:
        left.append("no rated games yet", style=f"{DIM} on {BG}")

    right = Text(no_wrap=True, justify="right")
    right.append(f"{counts.get('win', 0)}W ", style=f"{GOOD} on {BG}")
    right.append(f"{counts.get('draw', 0)}D ", style=f"{DIM} on {BG}")
    right.append(f"{counts.get('loss', 0)}L", style=f"{BAD} on {BG}")
    right.append(f"   {counts.get('all', 0)} games  ", style=f"{DIM} on {BG}")
    right.append_text(track_tag(f, "lichess"))
    # A heartbeat that is independent of the data it sits next to: it proves
    # the render loop is alive even when nothing has changed, which is the
    # difference between "quiet" and "dead".
    right.append(f"  {'◆' if int(time.time() * 2) % 2 else '◇'}",
                 style=f"{FAINT} on {BG}")

    grid.add_row(left, right)
    return _panel(grid, "", border=FAINT, subtitle=f"[{FAINT}]lichess.org/@/{user}/tv[/]")


def _rating_history(state) -> dict[str, list[float]]:
    hist: dict[str, list[float]] = {}
    for rec in state.get("rating_log", []) or []:
        for tc, d in (rec.get("ratings") or {}).items():
            if d.get("rating"):
                hist.setdefault(tc, []).append(d["rating"])
    return hist


# ---- the board ------------------------------------------------------------

def board_panel(state, user: str, width: int, height: int, scale: str = "pixel2",
                image_rows: int = 0):
    """The board, and the two player lines that frame it.

    `image_rows` is non-zero when the board is being drawn as a sixel image by
    the caller. In that case this reserves exactly that many blank rows and
    draws nothing into them: the image lives underneath, and the region has to
    render identically every frame or `rich` will repaint over the picture.
    """
    game = state.get("game")
    playing = state.get("playing", []) or []

    if not game:
        return _panel(
            Align.center(_idle_text(state), vertical="middle"),
            "board", border=FAINT,
        )

    board: chess.Board = game["board"]
    meta = game.get("meta", {})
    players = meta.get("players", {})
    we = _our_colour(players, user, playing)
    flip = we == "black"

    wc, bc = live_clocks(game)
    top_clock = bc if not flip else wc
    bottom_clock = wc if not flip else bc
    top_name, bottom_name = _names(players, flip)

    eng = state.get("engine") or {}
    wp_white = eng.get("wp_white") if _engine_matches(eng, board) else None
    # The gauge belongs to whoever is at the bottom of the board, which is us.
    wp_bottom = None if wp_white is None else (1 - wp_white if flip else wp_white)
    wp_top = None if wp_bottom is None else 1 - wp_bottom

    if image_rows:
        # The image region has to be *unstyled*, not merely blank.
        #
        # `rich` skips writing cells that carry no style, which is what lets a
        # picture underneath survive a redraw. The moment those cells have a
        # background -- from a Panel, or from a console-wide style -- rich
        # writes them on every frame and the image is erased between one frame
        # and the next. Measured both ways: with a Panel the board never
        # appears at all, bare it survives indefinitely.
        #
        # So no Panel here, and the only styled thing on these rows is the
        # eval bar in the first two columns, which sits to the left of the
        # image and is therefore safe to repaint.
        head = Text(no_wrap=True)
        head.append("board ", style=f"{FG} on {BG}")
        head.append(f"lichess.org/{game['id']}", style=f"{FAINT} on {BG}")
        rows = [head]
        rows.append(_player_line(top_name, top_clock, board.turn == (chess.BLACK if not flip else chess.WHITE), _captured(board, chess.WHITE if flip else chess.BLACK), wp_top))
        ebar = evalbar(wp_bottom, image_rows, width=EVAL_WIDTH)
        for i in range(image_rows):
            row = Text(no_wrap=True)
            row.append_text(ebar[i])
            rows.append(row)
        rows.append(_player_line(bottom_name, bottom_clock, board.turn == (chess.WHITE if not flip else chess.BLACK), _captured(board, chess.BLACK if flip else chess.WHITE), wp_bottom))
        return Group(*rows)

    rows = []
    rows.append(_player_line(top_name, top_clock, board.turn == (chess.BLACK if not flip else chess.WHITE), _captured(board, chess.WHITE if flip else chess.BLACK), wp_top))
    if True:
        art = boardmod.render(board, flip=flip, last=game.get("last"), scale=scale)
        _bw, bh = boardmod.board_size(scale)
        ebar = evalbar(wp_bottom, bh - 1, width=EVAL_WIDTH)
        for i, line in enumerate(art.split("\n")):
            row = Text(no_wrap=True)
            row.append_text(ebar[i] if i < len(ebar) else Text("  ", style=f"on {BG}"))
            row.append(" ", style=f"on {BG}")
            row.append_text(line)
            rows.append(row)
    rows.append(_player_line(bottom_name, bottom_clock, board.turn == (chess.WHITE if not flip else chess.BLACK), _captured(board, chess.BLACK if flip else chess.WHITE), wp_bottom))

    # Same shape as the image path above: a label line rather than a panel, so
    # the two renderers do not look like two different programs.
    head = Text(no_wrap=True)
    head.append("board ", style=f"{FG} on {BG}")
    head.append(f"lichess.org/{game['id']}", style=f"{FAINT} on {BG}")
    return Group(head, *rows)


def _engine_matches(eng: dict, board: chess.Board) -> bool:
    """Only trust the engine's evaluation if it is about this position.

    The telemetry carries the FEN it was computed for. Without this check a
    stale eval from two plies ago renders as though it described the board on
    screen, which is precisely the confidently-wrong-number failure.
    """
    fen = eng.get("fen")
    if not fen:
        return False
    return fen.split(" ")[0] == board.fen().split(" ")[0] or \
        eng.get("ply") in (board.ply(), board.ply() - 1)


def _our_colour(players: dict, user: str, playing: list) -> str:
    name = (players.get("white", {}).get("user", {}) or {}).get("name", "")
    if name and name.lower() == user.lower():
        return "white"
    if playing:
        return playing[0].get("color", "white")
    return "black"


def _names(players: dict, flip: bool) -> tuple[str, str]:
    def label(side):
        u = (players.get(side, {}).get("user") or {})
        rating = players.get(side, {}).get("rating")
        title = u.get("title")
        name = u.get("name", "?")
        return f"{title + ' ' if title else ''}{name}" + (f" ({rating})" if rating else "")
    return (label("white"), label("black")) if flip else (label("black"), label("white"))


def _player_line(name: str, clock: float | None, to_move: bool, taken: Text,
                 prob: float | None = None) -> Text:
    """One player: their share of the evaluation, name, captures, clock.

    The percentage sits in the same columns as the gauge, directly above and
    below it, so the bar is labelled at both ends rather than being a coloured
    column the reader has to infer the meaning of.
    """
    line = Text(no_wrap=True)
    if prob is None:
        line.append(" " * (EVAL_WIDTH + 1), style=f"on {BG}")
    else:
        line.append(f"{prob * 100:>{EVAL_WIDTH}.0f}", style=f"{FG} on {BG}")
        line.append(" ", style=f"{DIM} on {BG}")
    line.append("▸ " if to_move else "  ", style=f"{ACCENT} on {BG}")
    line.append(f"{name}", style=f"{'bold ' + FG if to_move else DIM} on {BG}")
    line.append("  ")
    line.append_text(taken)
    line.append("   ")
    style = BAD if (clock is not None and clock < 10) else (ACCENT if to_move else DIM)
    line.append(clockstr(clock), style=f"bold {style} on {BG}")
    return line


def _captured(board: chess.Board, colour) -> Text:
    """What this side has taken, and by how much they are up.

    Derived from what is missing off the starting set rather than from a move
    history, so it is correct even if the dashboard attached mid-game.
    """
    start = {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2,
             chess.ROOK: 2, chess.QUEEN: 1}
    other = not colour
    out = Text(no_wrap=True)
    score = 0
    for piece_type, n in start.items():
        left = len(board.pieces(piece_type, other))
        taken = n - left
        if taken > 0:
            out.append(FIGURES[piece_type] * taken, style=f"{FAINT} on {BG}")
            score += taken * VALUES[piece_type]
    mine = sum(VALUES[p] * len(board.pieces(p, colour)) for p in VALUES)
    theirs = sum(VALUES[p] * len(board.pieces(p, other)) for p in VALUES)
    if mine > theirs:
        out.append(f" +{mine - theirs}", style=f"{GOOD} on {BG}")
    return out


def _idle_text(state) -> Text:
    t = Text(justify="center")
    t.append("no game in progress\n", style=f"{DIM} on {BG}")
    fin = state.get("finished")
    if fin:
        colour = {"win": GOOD, "loss": BAD, "draw": DIM}[fin["result"]]
        t.append(f"\nlast: {fin['result'].upper()}", style=f"bold {colour} on {BG}")
        t.append(f" vs {fin['opponent']}", style=f"{FG} on {BG}")
        if fin.get("opp_rating"):
            t.append(f" ({fin['opp_rating']})", style=f"{DIM} on {BG}")
        t.append(f"\n{fin.get('speed','')} {'rated' if fin.get('rated') else 'casual'}"
                 f" · {fin.get('status','')}", style=f"{DIM} on {BG}")
        if fin.get("delta") is not None:
            t.append(f"  {fin['delta']:+d}",
                     style=f"{GOOD if fin['delta'] >= 0 else BAD} on {BG}")
    return t


# ---- the mind -------------------------------------------------------------

def mind_panel(state, width: int, height: int):
    """What the search is doing, while it is doing it.

    This is the panel the whole exercise was for. Everything in it comes from
    the engine's own telemetry; the dashboard never evaluates anything itself,
    which is both a correctness property (it cannot disagree with the engine)
    and a resource one (it cannot steal the GPU the engine is searching on).
    """
    eng = state.get("engine")
    f = state.field("engine")
    inner = width - 4

    if not eng:
        body = Text("waiting for the engine to move\n", style=f"{DIM} on {BG}")
        body.append("logs/engine.jsonl", style=f"{FAINT} on {BG}")
        return _panel(body, "mind", border=FAINT)

    thinking = eng.get("ev") == "think"
    rows = []

    # Annunciator: state as position and colour, readable without parsing text.
    line = Text(no_wrap=True)
    line.append(" SEARCHING " if thinking else "   IDLE    ",
                style=f"bold {BG} on {ACCENT if thinking else FAINT}")
    line.append("  ")
    wp = eng.get("wp", 0.5)
    line.append(f"{wp * 100:5.1f}%", style=f"bold {FG} on {BG}")
    line.append(" to move", style=f"{DIM} on {BG}")
    if eng.get("mate"):
        line.append("   MATE IN LINE ", style=f"bold {BG} on {BAD}")
    line.append("   ")
    line.append_text(track_tag(f))
    rows.append(line)

    # The time budget as a draining gauge. This is a decision being made
    # against a deadline, and a bar communicates "nearly out of time" in a way
    # that two numbers side by side do not.
    used, budget = eng.get("elapsed", 0.0), max(eng.get("budget", 1e-9), 1e-9)
    frac = min(1.0, used / budget)
    gauge = Text(no_wrap=True)
    gauge.append("time  ", style=f"{DIM} on {BG}")
    gauge.append_text(bar(1.0 - frac, min(GAUGE_MAX, max(8, inner - 34)),
                          BAD if frac > 0.9 else COOL))
    gauge.append(f" {used:5.2f}s of {budget:.2f}s", style=f"{DIM} on {BG}")
    rows.append(gauge)

    stats = Text(no_wrap=True)
    stats.append("nodes ", style=f"{DIM} on {BG}")
    stats.append(f"{eng.get('nodes', 0):,}", style=f"{FG} on {BG}")
    stats.append("   nps ", style=f"{DIM} on {BG}")
    stats.append(f"{eng.get('nps', 0):,}", style=f"{FG} on {BG}")
    stats.append("   sims ", style=f"{DIM} on {BG}")
    stats.append(f"{eng.get('sims', 0):,}", style=f"{FG} on {BG}")
    stats.append("   ply ", style=f"{DIM} on {BG}")
    stats.append(f"{eng.get('ply', 0)}", style=f"{FG} on {BG}")
    rows.append(stats)
    rows.append(Text("", style=f"on {BG}"))

    top = eng.get("top") or []
    # Three header rows, a blank, then a blank and the PV underneath, inside a
    # panel that costs two rows of chrome. Everything else is ladder.
    room = height - 8
    rows.extend(ladder(top[:max(1, room)], min(GAUGE_MAX, max(6, inner - 34)), ACCENT))

    pv = eng.get("pv") or []
    if pv:
        rows.append(Text("", style=f"on {BG}"))
        line = Text(no_wrap=True, overflow="ellipsis")
        line.append("pv  ", style=f"{DIM} on {BG}")
        for i, san in enumerate(pv):
            line.append(san + " ", style=f"{(PLUM if i == 0 else FG)} on {BG}")
        rows.append(line)

    return _panel(Group(*rows), "mind",
                  border=ACCENT if thinking else FAINT,
                  subtitle=f"[{FAINT}]visits · q · prior[/]")


# ---- the move list --------------------------------------------------------

def moves_panel(state, width: int, height: int):
    """The game so far, with our own evaluation attached to our own moves.

    The curve at the top is the engine's P(White wins) per ply, held in
    White's frame. Only our moves have a point, because the opponent's
    evaluation is theirs and we never see it, so the trace is what SumoFish
    thought at each of its own turns.
    """
    game = state.get("game")
    curve = state.curve_series()
    rows = height - 4 - (1 if curve else 0)
    if not game or not game.get("moves"):
        body = [Text("no moves yet", style=f"{DIM} on {BG}")]
        return _panel(Group(*body), "moves")

    moves = game["moves"]
    body = []
    if len(curve) >= 2:
        spark, lo, hi = sparkline(curve, width=min(GAUGE_MAX, max(8, width - 20)), style=PLUM)
        line = Text(no_wrap=True)
        line.append("eval ", style=f"{DIM} on {BG}")
        line.append_text(spark)
        line.append(f" {lo:.2f}-{hi:.2f}", style=f"{FAINT} on {BG}")
        body.append(line)

    pairs = []
    for i in range(0, len(moves), 2):
        w = moves[i]
        b = moves[i + 1] if i + 1 < len(moves) else None
        pairs.append((i // 2 + 1, w, b))

    with_eval = state.curve
    for n, w, b in pairs[-max(1, rows):]:
        line = Text(no_wrap=True)
        line.append(f"{n:>3}. ", style=f"{FAINT} on {BG}")
        line.append(f"{w['san']:<8}", style=f"{FG} on {BG}")
        line.append(f"{b['san']:<8}" if b else " " * 8, style=f"{FG} on {BG}")
        # Whichever of the pair we actually played is the one we have a number
        # for; annotate it and leave the other blank rather than guessing.
        for mv in (w, b):
            if mv and mv["ply"] in with_eval:
                line.append(f"{with_eval[mv['ply']]:.2f}", style=f"{PLUM} on {BG}")
                break
        body.append(line)
    return _panel(Group(*body), "moves",
                  subtitle=f"[{FAINT}]{len(moves)} ply[/]")


# ---- training, machine, tape ---------------------------------------------

def curve_panel(state, width: int, height: int):
    """The whole game's evaluation, on a fixed 0..1 scale.

    Fixed rather than auto-scaled on purpose. Renormalising to the game's own
    range makes a dead-level draw look as dramatic as a collapse, which is the
    standard way an evaluation chart lies. The drawn midline is 0.50, and the
    vertical distance from it means the same thing in every game.
    """
    series = state.curve_series()
    inner_h = max(3, height - 4)
    if len(series) < 2:
        return _panel(Text("not enough moves yet", style=f"{DIM} on {BG}"),
                      "eval")
    rows = curve_chart(series, max(8, width - 10), inner_h)
    # Axis anchors in text. A trace without a scale is decoration.
    labelled = []
    for i, row in enumerate(rows):
        line = Text(no_wrap=True)
        if i == 0:
            line.append("1.0 ", style=f"{FAINT} on {BG}")
        elif i == inner_h // 2:
            line.append("0.5 ", style=f"{FAINT} on {BG}")
        elif i == inner_h - 1:
            line.append("0.0 ", style=f"{FAINT} on {BG}")
        else:
            line.append("    ", style=f"on {BG}")
        line.append_text(row)
        labelled.append(line)
    last = series[-1]
    return _panel(Group(*labelled), "eval",
                  subtitle=f"[{FAINT}]P(white) now {last:.3f} · {len(series)} points[/]")


def train_panel(state, width: int):
    data = state.get("train")
    f = state.field("train")
    if not data or not data.get("loss"):
        return _panel(Text("no training run active", style=f"{DIM} on {BG}"), "training")

    last = data["loss"][-1]
    total = 300_000
    frac = min(1.0, last["step"] / total)
    inner = width - 4

    line1 = Text(no_wrap=True)
    line1.append(f"{data.get('run', '?')}  ", style=f"{COOL} on {BG}")
    line1.append(f"{last['step']:,}", style=f"bold {FG} on {BG}")
    line1.append(f"/{total:,}  ", style=f"{DIM} on {BG}")
    eta = (total - last["step"]) / max(1.0, last.get("samples_per_s", 1) / 1024) / 3600
    line1.append(f"eta {eta:.1f}h  ", style=f"{DIM} on {BG}")
    line1.append_text(track_tag(f))

    line2 = Text(no_wrap=True)
    line2.append_text(bar(frac, min(GAUGE_MAX, max(8, inner - 8)), COOL))
    line2.append(f" {frac * 100:4.1f}%", style=f"{DIM} on {BG}")

    losses = [r["loss"] for r in data["loss"]]
    spark, lo, hi = sparkline(losses, width=min(GAUGE_MAX, max(8, inner - 30)), style=INFO)
    line3 = Text(no_wrap=True)
    line3.append("loss ", style=f"{DIM} on {BG}")
    line3.append(f"{last['loss']:.4f} ", style=f"{FG} on {BG}")
    line3.append_text(spark)
    line3.append(f" {lo:.3f}-{hi:.3f}", style=f"{FAINT} on {BG}")

    accs = [r["puzzle_acc"] for r in data.get("evals", [])]
    line4 = Text(no_wrap=True)
    if accs:
        aspark, alo, ahi = sparkline(accs, width=min(GAUGE_MAX, max(8, inner - 30)), style=WARM)
        line4.append("puzz ", style=f"{DIM} on {BG}")
        line4.append(f"{accs[-1]:.3f} ", style=f"{ACCENT} on {BG}")
        line4.append_text(aspark)
        line4.append(f" {alo:.3f}-{ahi:.3f}", style=f"{FAINT} on {BG}")
    else:
        line4.append("no puzzle eval yet", style=f"{FAINT} on {BG}")

    return _panel(Group(line1, line2, line3, line4), "training")


def machine_panel(state, width: int):
    """The shared plant. Training and the bot draw on the same GPU, and
    "is one starving the other" should be answerable by looking."""
    gpu = state.get("gpu")
    units = state.get("units") or {}
    f = state.field("gpu")
    inner = width - 4
    rows = []

    if gpu:
        line = Text(no_wrap=True)
        line.append("gpu  ", style=f"{DIM} on {BG}")
        line.append_text(bar(gpu["util"] / 100.0, min(GAUGE_MAX, max(6, inner - 40)), COOL))
        line.append(f" {gpu['util']:3.0f}%", style=f"{FG} on {BG}")
        line.append(f"  {gpu['temp']:.0f}°C",
                    style=f"{BAD if gpu['temp'] > 80 else DIM} on {BG}")
        line.append(f"  {gpu['power']:.0f}W", style=f"{DIM} on {BG}")
        rows.append(line)
        line = Text(no_wrap=True)
        line.append("vram ", style=f"{DIM} on {BG}")
        line.append_text(bar(gpu["used"] / max(gpu["total"], 1), min(GAUGE_MAX, max(6, inner - 40)), PLUM))
        line.append(f" {gpu['used'] / 1024:.1f}/{gpu['total'] / 1024:.0f}G",
                    style=f"{DIM} on {BG}")
        line.append("  ")
        line.append_text(track_tag(f))
        rows.append(line)
    else:
        rows.append(Text("nvidia-smi unavailable", style=f"{DIM} on {BG}"))

    line = Text(no_wrap=True)
    for unit, info in units.items():
        ok = info.get("active") == "active" and info.get("sub") == "running"
        line.append("●", style=f"{GOOD if ok else BAD} on {BG}")
        line.append(f" {unit.replace('chess-gpu-', '')} ", style=f"{DIM} on {BG}")
        if info.get("restarts", "0") not in ("0", "?"):
            line.append(f"({info['restarts']} restarts) ", style=f"{WARM} on {BG}")
    # How hard this viewer is leaning on lichess, since it shares an address
    # with the bot and a throttled bot stops playing. Answerable by looking.
    rate = GATE.per_minute()
    line.append(f"  api {rate}/min", style=f"{DIM} on {BG}")
    if GATE.throttled:
        line.append(f"  {GATE.throttled} throttled", style=f"{BAD} on {BG}")
    rows.append(line)
    return _panel(Group(*rows), "machine")


def tape_panel(state, width: int, height: int):
    kinds = {"move": ACCENT, "game": INFO, "result": GOOD,
             "warn": BAD, "engine": PLUM}
    rows = []
    for ts, kind, text in state.tape.tail(max(1, height - 4)):
        line = Text(no_wrap=True, overflow="ellipsis")
        line.append(time.strftime("%H:%M:%S ", time.localtime(ts)),
                    style=f"{FAINT} on {BG}")
        line.append(f"{kind:<7}", style=f"{kinds.get(kind, DIM)} on {BG}")
        line.append(text, style=f"{FG} on {BG}")
        rows.append(line)
    if not rows:
        rows = [Text("nothing yet", style=f"{FAINT} on {BG}")]
    return _panel(Group(*rows), "tape")
