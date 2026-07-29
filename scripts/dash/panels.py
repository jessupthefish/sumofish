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

import math
import time
from pathlib import Path

import chess
from rich.align import Align
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import board as boardmod
from .sources import GATE
from .theme import (
    ACCENT, BAD, BG, COOL, DIM, EVAL_WHITE, FAINT, FG, GAUGE_MAX, GOOD, INFO,
    PLUM, WARM,
)
from .widgets import (bar, curve_chart, evalbar, ladder, sparkline,
                      track_tag)

# Columns for the evaluation gauge, which lives in the evaluation panel beside
# the history it is the latest point of. Three rather than two: at two it is
# taller than it is wide by a factor of twelve and reads as a rule, not a bar.
GAUGE_W = 3

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

def active_controls() -> set[str]:
    """The time controls the bot actually accepts, from its own config.

    Used to decide which rating gets the full treatment. The bot plays 15+10
    and nothing else, so bullet and blitz can no longer move: they are history
    worth keeping on screen and not worth spending a sparkline, a deviation and
    a game count on.

    The except is narrow on purpose. It was `except Exception`, and it silently
    swallowed a NameError -- `Path` was not imported in this module -- so the
    function returned "everything" and looked like it was working. A broad
    except around a config read turns a bug into wrong behaviour with no
    signal, which is the whole failure it was meant to prevent.
    """
    try:
        import yaml
        cfg = yaml.safe_load((ROOT / "config" / "lichess-bot.yml").read_text())
        return set(cfg.get("challenge", {}).get("time_controls") or []) or _ALL
    except (OSError, ImportError, ValueError, AttributeError):
        return _ALL


ROOT = Path(__file__).resolve().parent.parent.parent
_ALL = {"bullet", "blitz", "rapid", "classical"}


def header(state, user: str, width: int):
    """Who we are, how we are rated, how we are doing, and is the feed alive.

    Everything here was previously an unlabelled abbreviation -- "bul 2513?
    rd199 4g -456" -- which is dense but unreadable unless you already know
    what it says. The values are still compact, because the row is one line
    and there are four time controls, but the notation is now explained once
    in the subtitle rather than left to be guessed at.
    """
    prof = state.get("profile", {})
    f = state.field("profile")
    perfs = prof.get("perfs", {})
    counts = prof.get("count", {})
    hist = _rating_history(state)

    grid = Table.grid(padding=(0, 2), expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="right")

    left = Text(no_wrap=True)
    left.append(" SUMOFISH ", style=f"bold {BG} on {ACCENT}")
    shown = 0
    live = active_controls()
    # The control we play gets the detail; the ones we no longer play get the
    # number and nothing else. A deviation, a game count and a sparkline on a
    # rating that cannot move again is width spent on history.
    ordering = sorted(("bullet", "blitz", "rapid", "classical"),
                      key=lambda tc: tc not in live)
    for tc in ordering:
        p = perfs.get(tc)
        if not p or not p.get("games"):
            continue
        shown += 1
        left.append("   ")
        if tc not in live:
            # History, and the first thing to give up its space. The wide
            # layout starts at 100 columns and at that size the record on the
            # right plus the live rating already fill the row -- keeping these
            # would push the blitz number off the end, which is worse than not
            # showing it. The verbose "34 won / 8 drawn" on the right is a
            # deliberate readability choice by whoever wrote it, so it is not
            # the thing to compress.
            if width < 120:
                shown -= 1
                continue
            left.append(f"{tc} ", style=f"{FAINT} on {BG}")
            left.append(f"{p['rating']}", style=f"{DIM} on {BG}")
            continue
        left.append(f"{tc} ", style=f"{DIM} on {BG}")
        left.append(f"{p['rating']}", style=f"bold {ACCENT} on {BG}")
        if p.get("prov"):
            left.append("?", style=f"{WARM} on {BG}")
        left.append(f" \u00b1{p.get('rd','?')}", style=f"{FAINT} on {BG}")
        # "32g" rather than "32 games": the wide layout starts at 100 columns
        # and the long form clipped the blitz rating off the end there.
        left.append(f"  {p.get('games')}g", style=f"{DIM} on {BG}")
        series = hist.get(tc, [])
        if len(series) >= 2:
            left.append("  ")
            spark, _lo, _hi = sparkline(series, width=14, style=INFO)
            left.append_text(spark)
            delta = series[-1] - series[0]
            left.append(f" {delta:+.0f}",
                        style=f"{GOOD if delta >= 0 else BAD} on {BG}")
    if not shown:
        left.append("   no rated games at the control we play",
                    style=f"{DIM} on {BG}")

    right = Text(no_wrap=True)
    game = state.get("game")
    if game:
        # Up here rather than over the board: it is a thing you go and look at
        # once, not a thing you watch, and beside the picture it was clutter.
        right.append(f"lichess.org/{game['id']}   ", style=f"{FAINT} on {BG}")
    right.append(f"{counts.get('win', 0)} won", style=f"{GOOD} on {BG}")
    right.append(" \u00b7 ", style=f"{FAINT} on {BG}")
    right.append(f"{counts.get('draw', 0)} drawn", style=f"{DIM} on {BG}")
    right.append(" \u00b7 ", style=f"{FAINT} on {BG}")
    right.append(f"{counts.get('loss', 0)} lost", style=f"{BAD} on {BG}")
    right.append(f"   of {counts.get('all', 0)}   ", style=f"{DIM} on {BG}")
    right.append_text(track_tag(f, "lichess"))
    # A heartbeat independent of the data beside it: it proves the render loop
    # is alive even when nothing has changed, which is the difference between
    # "quiet" and "dead".
    right.append(f"  {'\u25c6' if int(time.time() * 2) % 2 else '\u25c7'}",
                 style=f"{FAINT} on {BG}")

    grid.add_row(left, right)
    return _panel(grid, "", border=FAINT)


def _rating_history(state) -> dict[str, list[float]]:
    hist: dict[str, list[float]] = {}
    for rec in state.get("rating_log", []) or []:
        for tc, d in (rec.get("ratings") or {}).items():
            if d.get("rating"):
                hist.setdefault(tc, []).append(d["rating"])
    return hist


# ---- the board ------------------------------------------------------------

def board_panel(state, user: str, width: int, height: int, scale: str = "pixel2",
                image_rows: int = 0, image_cols: int = 0, indent: int = 0):
    """The board, and the two player blocks that frame it.

    `image_rows` is non-zero when the board is being drawn as a sixel image by
    the caller. In that case this reserves exactly that many blank rows and
    draws nothing into them: the image lives underneath, and the region has to
    render identically every frame or `rich` will repaint over the picture.

    `indent` and `image_cols` are where the picture starts and how wide it is,
    so the player blocks can be set to the board's own edges rather than to the
    column's. They are not the same thing -- the column is the picture plus a
    gutter either side -- and a name that starts left of the board's edge while
    a clock ends right of it is what made the whole thing look off-centre.
    """
    game = state.get("game")
    playing = state.get("playing", []) or []

    if not game:
        return _panel(
            Align.center(_idle_text(state), vertical="middle"),
            "board", border=FAINT,
        )

    # The stream is the authoritative record but it runs about nine seconds
    # behind (measured; see dash/fusion.py). The engine's own view of the
    # position is local and instant, so it wins whenever it is ahead -- which
    # is nearly always, and is why the mind panel used to be showing a
    # position the board had not reached.
    stream_board: chess.Board = game["board"]
    eb = state.get("engine_board")
    if eb and eb["ply"] >= stream_board.ply():
        board, last_move = eb["board"], eb["last"]
    else:
        board, last_move = stream_board, game.get("last")
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

    span = image_cols or width
    top_block = _player_block(
        top_name, top_clock, board.turn == (chess.BLACK if not flip else chess.WHITE),
        _captured(board, chess.WHITE if flip else chess.BLACK), wp_top,
        span, indent, above=True)
    bottom_block = _player_block(
        bottom_name, bottom_clock, board.turn == (chess.WHITE if not flip else chess.BLACK),
        _captured(board, chess.BLACK if flip else chess.WHITE), wp_bottom,
        span, indent, above=False)

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
        rows = list(top_block)
        # Nothing styled on these rows at all. The picture is underneath and
        # rich rewrites a whole line when any cell in it changes, so a styled
        # gauge here erases the board a row at a time as it animates.
        # Picture only. Nothing styled goes in this column beside or below
        # the image any more: two separate corruption bugs came from trying,
        # and the gauge reads better in the search panel next to the number it
        # is a picture of.
        for _ in range(image_rows):
            rows.append(Text(""))
        rows.extend(bottom_block)
        return Group(*rows)

    rows = list(top_block)
    art = boardmod.render(board, flip=flip, last=last_move, scale=scale)
    for line in art.split("\n"):
        rows.append(line)
    rows.extend(bottom_block)

    # Same shape as the image path above: a label line rather than a panel, so
    # the two renderers do not look like two different programs.
    return Group(*rows)


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


def ours(state, wp_white: float | None) -> float | None:
    """Convert a White-framed probability to our side's.

    The single place this conversion happens. It used to be done at each call
    site and one of them was missed, so with SumoFish playing Black and being
    mated the move list read 0.97 while the chart read 0.03 -- both correct in
    their own frame, and the panel that happened to be in White's frame said
    we were winning. One function, so the frames cannot drift apart again.
    """
    if wp_white is None:
        return None
    return 1.0 - wp_white if _we_are_black(state) else wp_white


def _we_are_black(state) -> bool:
    """Which way round the evaluation should read, for this game."""
    game = state.get("game")
    if not game:
        return False
    players = game.get("meta", {}).get("players", {})
    return _our_colour(players, "SumoFish", state.get("playing", []) or []) == "black"


def _our_colour(players: dict, user: str, playing: list) -> str:
    name = (players.get("white", {}).get("user", {}) or {}).get("name", "")
    if name and name.lower() == user.lower():
        return "white"
    if playing:
        return playing[0].get("color", "white")
    return "black"


def _names(players: dict, flip: bool):
    """(title, name, rating) per side, nearest player last.

    Kept as three fields rather than one string because the rating wants its
    own column: buried in parentheses after a variable-length name it lands in
    a different place on every line, which is exactly why it was hard to find.
    """
    def parts(side):
        u = (players.get(side, {}).get("user") or {})
        return (u.get("title") or "", u.get("name", "?"),
                players.get(side, {}).get("rating"))
    return (parts("white"), parts("black")) if flip else (parts("black"), parts("white"))


# Columns for a player line. Fixed, so both players' figures land in the same
# place and can be compared by looking rather than by reading.
NAME_W, RATING_W, CLOCK_W = 20, 5, 8

# Where the second row of a player block starts: past the to-move marker, so
# the material hangs under the name rather than under the marker column.
MARK_W = 2


def _player_block(who, clock: float | None, to_move: bool, taken: Text,
                  prob: float | None, width: int, indent: int,
                  above: bool) -> list[Text]:
    """One player, over two rows, set to the board's own width.

    The material used to share the name's row, wedged between the rating and
    the clock, where twelve captured pieces became an unreadable smear of
    identical figurines with no room to separate them. It gets its own row
    here, and the row always exists even when nothing has been taken, because a
    block that changes height would move the board picture under it.

    The material row is the one nearest the board on both sides -- names
    outermost, captures innermost -- so the two rows read outward from the
    position rather than in the same direction on both halves of the screen.
    """
    name = _name_line(who, clock, to_move, prob, width, indent)
    mat = _material_line(taken, width, indent)
    return [name, mat] if above else [mat, name]


def _name_line(who, clock: float | None, to_move: bool,
               prob: float | None, width: int, indent: int) -> Text:
    """Name, title, rating on the left; clock and win estimate on the right.

    Aligned on purpose. The rating used to sit in parentheses after the name,
    so with a four-character name above a twelve-character one the two numbers
    were nowhere near each other and comparing them meant hunting.

    Both names are drawn bold, not just the one to move: dimming the waiting
    player costs half the readability of the line to say something the marker
    and the live clock already say twice.
    """
    title, name, rating = who
    line = Text(no_wrap=True)
    if indent:
        line.append(" " * indent, style=f"on {BG}")
    line.append("\u25b8 " if to_move else "  ", style=f"{ACCENT} on {BG}")
    line.append(f"{name[:NAME_W]:<{NAME_W}}", style=f"bold {FG} on {BG}")
    line.append(f"{title:>3} ", style=f"{PLUM if title else DIM} on {BG}")
    line.append(f"{rating if rating else '--':>{RATING_W}}",
                style=f"bold {ACCENT} on {BG}")

    tail = Text(no_wrap=True)
    style = BAD if (clock is not None and clock < 10) else (ACCENT if to_move else DIM)
    tail.append(f"{clockstr(clock):>{CLOCK_W}}", style=f"bold {style} on {BG}")
    if prob is None:
        tail.append("      ", style=f"on {BG}")
    else:
        tail.append(f"{prob * 100:>5.0f}%", style=f"{FG} on {BG}")

    pad = indent + width - line.cell_len - tail.cell_len
    if pad > 0:
        line.append(" " * pad, style=f"on {BG}")
    line.append_text(tail)
    return line


def _material_line(taken: Text, width: int, indent: int) -> Text:
    line = Text(no_wrap=True)
    if indent + MARK_W:
        line.append(" " * (indent + MARK_W), style=f"on {BG}")
    line.append_text(taken)
    pad = indent + width - line.cell_len
    if pad > 0:
        line.append(" " * pad, style=f"on {BG}")
    return line


def _captured(board: chess.Board, colour) -> Text:
    """What this side has taken, and by how much they are up.

    Grouped and counted rather than drawn one figurine per piece. Six pawns
    printed as six pawns is fourteen characters of identical eight-pixel
    silhouettes with nothing between them, which is a texture and not a number;
    `\u265f6` is two, one of which is a digit. The count is dropped at one, because
    `\u265b1` reads as a worse `\u265b`.

    Derived from what is missing off the starting set rather than from a move
    history, so it is correct even if the dashboard attached mid-game.
    """
    start = {chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 2,
             chess.KNIGHT: 2, chess.PAWN: 8}
    other = not colour
    out = Text(no_wrap=True)
    for piece_type, n in start.items():
        taken = n - len(board.pieces(piece_type, other))
        if taken > 0:
            out.append(FIGURES[piece_type], style=f"{DIM} on {BG}")
            out.append(f"{taken}" if taken > 1 else " ",
                       style=f"bold {FG} on {BG}")
            out.append(" ", style=f"on {BG}")
    mine = sum(VALUES[p] * len(board.pieces(p, colour)) for p in VALUES)
    theirs = sum(VALUES[p] * len(board.pieces(p, other)) for p in VALUES)
    if mine > theirs:
        out.append(f" +{mine - theirs}", style=f"bold {GOOD} on {BG}")
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
        return _panel(body, "engine search", border=FAINT)

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

    return _panel(Group(*rows), "engine search",
                  border=ACCENT if thinking else FAINT,
                  )


# ---- the move list --------------------------------------------------------

def moves_panel(state, width: int, height: int):
    """The game so far, with our own evaluation attached to our own moves.

    The curve at the top is the engine's P(White wins) per ply, held in
    White's frame. Only our moves have a point, because the opponent's
    evaluation is theirs and we never see it, so the trace is what SumoFish
    thought at each of its own turns.
    """
    game = state.get("game")
    curve = [ours(state, v) for v in state.curve_series()]
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

    with_eval = {ply: ours(state, v) for ply, v in state.curve_items()}
    # Stockfish's verdict on each move, ours and theirs. Only the bad ones are
    # marked: annotating every accurate move would be a column of noise, and
    # the eye is looking for where a game went wrong.
    grades = state.get("grades") or {}
    MARKS = {"inaccuracy": ("?!", WARM), "mistake": ("?", BAD),
             "blunder": ("??", BAD)}

    def annotate(line, mv):
        mark, style = MARKS.get(
            (grades.get(mv["ply"]) or {}).get("grade", ""), ("  ", FAINT))
        line.append(f"{mark:<2}", style=f"bold {style} on {BG}")

    for n, w, b in pairs[-max(1, rows):]:
        line = Text(no_wrap=True)
        line.append(f"{n:>3}. ", style=f"{FAINT} on {BG}")
        line.append(f"{w['san']:<7}", style=f"{FG} on {BG}")
        annotate(line, w)
        line.append(f"{b['san']:<7}" if b else " " * 7, style=f"{FG} on {BG}")
        if b:
            annotate(line, b)
        else:
            line.append("  ", style=f"{FAINT} on {BG}")
        # Whichever of the pair we actually played is the one we have a number
        # for; annotate it and leave the other blank rather than guessing.
        for mv in (w, b):
            if mv and mv["ply"] in with_eval:
                line.append(f"{with_eval[mv['ply']]:.2f}", style=f"{PLUM} on {BG}")
                break
        body.append(line)
    return _panel(Group(*body), "moves",
                  )


# ---- training, machine, tape ---------------------------------------------

def advantage(wp: float | None) -> str | None:
    """A win probability on the conventional pawn scale, ours positive.

    SumoFish has no centipawn scale. It predicts P(win) and nothing else, so
    this is a second way of writing the same number rather than a second
    opinion about the position -- the mapping is monotone, which is what makes
    it safe: `+1.20` and a gauge two thirds full can never be telling different
    stories, because one is a function of the other.

    The mapping is the Elo logistic inverted, `cp = 400 * log10(p / (1 - p))`,
    which is the one `chessgpu.engines.search_engine.centipawns` already uses
    for `score cp` over UCI, so the number here and the number a lichess
    analysis board shows for our moves are the same number. Copied rather than
    imported on purpose: that module pulls in torch, and the viewer not
    importing torch is the property that makes it impossible for the dashboard
    to take GPU time from the engine.

    Signed from SumoFish's point of view, like everything else in this panel:
    `+` is us better, whichever colour we are. That is deliberately not the
    White-relative sign a chess GUI uses, and it is why the number is drawn in
    the gauge's own ink rather than in black and white.
    """
    if wp is None:
        return None
    p = min(max(wp, 1e-4), 1.0 - 1e-4)
    return f"{400.0 * math.log10(p / (1.0 - p)) / 100.0:+.2f}"


def curve_panel(state, width: int, height: int):
    """Now and so far: a gauge of the current evaluation, then its history.

    Both on one axis, and both from *our* side's point of view. They used to
    disagree -- the gauge measured the side at the bottom of the board while
    the chart plotted White -- which is invisible until we play Black and then
    they point opposite ways. Side by side that would be indefensible, so the
    whole panel is now "chance SumoFish wins" and the gauge is simply the
    right-hand end of the curve, drawn tall enough to read.

    The scale is fixed 0..1 and never fitted to the game's own range.
    Renormalising makes a dead-level draw look like a collapse, which is the
    usual way a chart of this kind lies.
    """
    # One row goes to the readout, which is the number the gauge is a picture
    # of. It reads the *target* rather than the eased value: a bar may slide,
    # a figure may not show a value the engine never produced.
    inner_h = max(3, height - 5)
    series = [ours(state, v) for v in state.curve_series()]

    now = state.eval_smooth.target
    head = Text(no_wrap=True)
    adv = advantage(now)
    if adv is None:
        # The gauge is empty in this case too, so the panel says "no evaluation
        # for the position on the board" in both places at once.
        head.append("     --", style=f"{DIM} on {BG}")
    else:
        head.append(f"{adv:>7}", style=f"{EVAL_WHITE} on {BG}")
        head.append(f"   wp {now:.2f}", style=f"{FAINT} on {BG}")

    gauge = evalbar(state.eval_smooth.value, inner_h, width=GAUGE_W)
    chart_w = max(8, width - GAUGE_W - 10)
    rows = curve_chart(series, chart_w, inner_h) if len(series) >= 2 else []

    out = [head]
    for i in range(inner_h):
        line = Text(no_wrap=True)
        if i == 0:
            line.append("1.0 ", style=f"{FAINT} on {BG}")
        elif i == inner_h // 2:
            line.append("0.5 ", style=f"{FAINT} on {BG}")
        elif i == inner_h - 1:
            line.append("0.0 ", style=f"{FAINT} on {BG}")
        else:
            line.append("    ", style=f"on {BG}")
        line.append_text(gauge[i])
        line.append("  ", style=f"on {BG}")
        if i < len(rows):
            line.append_text(rows[i])
        out.append(line)

    return _panel(Group(*out), "evaluation")


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

    return _panel(Group(line1, line2, line3, line4), "training",
                  )


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
    return _panel(Group(*rows), "machine",
                  )


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
    return _panel(Group(*rows), "event log")


def results_panel(state, width: int, height: int):
    """The last few finished games, won and lost.

    The header has carried a lifetime tally for a while -- "19 won, 4 drawn,
    63 lost of 86" -- and that number cannot answer the question you actually
    have after leaving it running overnight, which is *which* games. A record
    of 19-63 reads the same whether the bot lost eight straight to one opponent
    at one time control or drifted down against everybody, and those call for
    completely different responses.

    Source is `logs/games/*.pgn`, which lichess-bot writes itself on every
    finished game (`pgn_directory`), so this costs no API calls and survives a
    restart. Parsing is done by the source thread, never here: `dash/sources.py`
    owns cadence, panels are a pure function of state.

    The same data in a wider, scrollable, filterable form is `sumofish-games`,
    which is the one to open when a game here looks worth reading.
    """
    rows = state.get("results") or []
    f = state.field("results")
    if not rows:
        return _panel(Text("no finished games yet", style=f"{DIM} on {BG}"),
                      "recent games", subtitle=track_tag(f, ""))

    # Width is allocated, not guessed. The first version hardcoded an 11-column
    # slice for the termination and "time forfeit" is twelve, so every flagged
    # game read "time forfei" -- the exact class of bug a fixed slice invites.
    # Everything but the opponent is fixed-width and known, so measure those,
    # give the termination what it needs, and let the opponent take the rest.
    # Count EVERY column including the separators, or the row is one over and
    # rich clips the last character -- which is how "time forfeit" became
    # "time forfei" twice: once from an 11-char slice, and again from an
    # off-by-one here that dropped the same letter for a different reason.
    # The assertion below is what stops there being a third time.
    HOW = 12                      # "time forfeit", the longest lichess sends
    fixed = 7 + 2 + 1 + 8 + HOW   # " HH:MM " + "D " + gap + "  180+0 " + how
    who_w = max(8, width - 4 - fixed)

    body = Text(no_wrap=True)
    # Two rows of chrome: the panel border top and bottom.
    for game in rows[: max(0, height - 2)]:
        mark, style = {"win": ("W", GOOD), "loss": ("L", BAD),
                       "draw": ("D", DIM)}.get(game["verdict"], ("·", FAINT))
        body.append(f" {game['when']} ", style=f"{FAINT} on {BG}")
        body.append(f"{mark} ", style=f"bold {style} on {BG}")
        body.append(f"{game['opponent'][:who_w]:<{who_w}} ", style=f"{DIM} on {BG}")
        body.append(f"{game['tc']:>7} ", style=f"{FAINT} on {BG}")
        body.append(f"{game['how'][:HOW]}\n", style=f"{FAINT} on {BG}")

    won = sum(1 for g in rows if g["verdict"] == "win")
    lost = sum(1 for g in rows if g["verdict"] == "loss")
    drew = sum(1 for g in rows if g["verdict"] == "draw")
    sub = Text(no_wrap=True)
    sub.append(f"last {won + drew + lost}: ", style=f"{FAINT} on {BG}")
    sub.append(f"{won}W", style=f"{GOOD} on {BG}")
    sub.append(" ", style=f"{FAINT} on {BG}")
    sub.append(f"{drew}D", style=f"{DIM} on {BG}")
    sub.append(" ", style=f"{FAINT} on {BG}")
    sub.append(f"{lost}L", style=f"{BAD} on {BG}")
    return _panel(body, "recent games", subtitle=sub)
