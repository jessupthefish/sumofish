"""Drawing the board as a real image, when the terminal can take one.

Half-blocks give a board whose resolution is the cell grid: one sprite pixel
per cell, so ~8 screen pixels per pixel, and pieces that look like Atari art
however carefully they are drawn. Sixel removes that ceiling entirely. The
terminal accepts a bitmap at screen resolution, so the board becomes the
actual cburnett artwork at whatever size the panel is, smooth, identical to
the one on lichess because it is rendered from the same SVGs.

This was ruled out early on the assumption that Konsole's sixel support is off
by default and unreliable. That assumption was never tested and it was wrong:
Konsole 26.04 answers the primary device attributes query with `4` in the
list, which is the standard way a terminal says "I do sixel", and it renders
without any configuration at all.

## Everything here is optional and everything here can fail

Sixel needs `rsvg-convert` and ImageMagick at run time, which the rest of the
dashboard does not. So this module reports what it can do and the caller falls
back to the text renderer when the answer is "nothing". A missing binary, a
terminal that does not answer, output that is not a tty: all of them mean the
board is drawn as text, and none of them mean an error on screen.

## How it survives a redraw

`rich`'s `Live` diffs frames and only writes cells that changed. If the board
region renders as the same blank space every frame, it is written once and
never touched again, so the image underneath stays. The image therefore only
has to be re-emitted when the position changes, which is about once a second
rather than eight times.
"""

from __future__ import annotations

import os
import re
import select
import shutil
import subprocess
import sys
import termios
import tty
from dataclasses import dataclass

import chess
import chess.svg

# The board panel keeps a small margin so the image never lands on the panel
# border, which sixel would happily paint straight over.
MARGIN_CELLS = 1


@dataclass
class Caps:
    """What this terminal will actually let us do."""

    cell_w: float
    cell_h: float

    def px(self, cols: int, rows: int) -> tuple[int, int]:
        return int(cols * self.cell_w), int(rows * self.cell_h)


def _ask(seq: str, terminator: str, timeout: float = 0.4) -> str:
    """Send a query and read the reply, with the tty in raw mode.

    Two things here are load-bearing and both were got wrong first time.

    Raw mode, because otherwise the reply is line buffered and never arrives,
    and it is echoed into the display on the way past.

    And `os.read` on the file descriptor rather than `sys.stdin.read`. Python's
    text layer does its own buffering: it pulls the whole reply off the fd,
    hands back one character, and keeps the rest. `select` then reports nothing
    pending, the loop exits with a single escape byte, and the remainder
    surfaces in the *next* query's answer -- which reads exactly like a
    terminal that does not support the query at all.
    """
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        sys.stdout.write(seq)
        sys.stdout.flush()
        out = b""
        while select.select([fd], [], [], timeout)[0]:
            chunk = os.read(fd, 64)
            if not chunk:
                break
            out += chunk
            if terminator.encode() in chunk:
                break
        return out.decode("ascii", "replace")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _drain() -> None:
    """Swallow anything the terminal is still sending.

    A reply can arrive with a trailing byte after its terminator, and once the
    tty leaves raw mode that byte is echoed -- it turned up as a stray `t` in
    the bottom-left corner of the screen. Read whatever is left and discard it.
    """
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while select.select([fd], [], [], 0.05)[0]:
            if not os.read(fd, 256):
                break
    except Exception:                                    # noqa: BLE001
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def probe(cols: int, rows: int) -> Caps | None:
    """Ask the terminal whether it does sixel, and how big a cell is.

    Returns None for any reason at all, because every reason has the same
    consequence: draw the board as text instead.
    """
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    if not (shutil.which("rsvg-convert") and shutil.which("magick")):
        return None
    try:
        # Primary device attributes. A `4` among the parameters is the
        # standard claim of sixel support.
        da = _ask("\x1b[c", "c")
        if ";4" not in da and ";4;" not in da:
            return None
        # Text area in pixels: the reply is ESC [ 4 ; height ; width t.
        geom = _ask("\x1b[14t", "t")
        m = re.search(r"\[4;(\d+);(\d+)t", geom)
        if not m:
            return None
        h_px, w_px = int(m.group(1)), int(m.group(2))
        if not (cols and rows and w_px and h_px):
            return None
        _drain()
        return Caps(w_px / cols, h_px / rows)
    except Exception:                                    # noqa: BLE001
        return None


# lichess's own board colours. The point of this renderer is that the board is
# the one from the website, so it does not get restyled into the desktop
# palette; the panels around it still are.
COLOURS = {
    "square light": "#f0d9b5",
    "square dark": "#b58863",
    "square light lastmove": "#cdd26a",
    "square dark lastmove": "#aaa23a",
    "margin": "#2b2724",
    "coord": "#e5e0d5",
}

_CACHE: dict = {}


def render(board: chess.Board, flip: bool, last, size_px: int) -> bytes | None:
    """The position as a sixel payload, or None if the toolchain failed."""
    key = (board.fen(), flip, last.uci() if last else None, size_px)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    try:
        svg = chess.svg.board(
            board,
            orientation=chess.BLACK if flip else chess.WHITE,
            lastmove=last,
            check=board.king(board.turn) if board.is_check() else None,
            size=size_px,
            coordinates=True,
            colors=COLOURS,
        )
        png = subprocess.run(
            ["rsvg-convert", "-w", str(size_px), "-h", str(size_px), "-f", "png"],
            input=svg.encode(), capture_output=True, timeout=10, check=True,
        ).stdout
        data = subprocess.run(
            ["magick", "png:-", "-depth", "8", "sixel:-"],
            input=png, capture_output=True, timeout=10, check=True,
        ).stdout
    except Exception as exc:                             # noqa: BLE001
        _log(f"render FAILED {type(exc).__name__}: {exc}")
        return None
    if len(_CACHE) > 4:
        _CACHE.clear()
    _CACHE[key] = data
    return data


DEBUG = os.environ.get("CHESSGPU_SIXEL_DEBUG")


def _log(msg: str) -> None:
    if not DEBUG:
        return
    try:
        with open(DEBUG, "a") as fh:
            fh.write(msg + "\n")
    except OSError:
        pass


def emit(row: int, col: int, data: bytes) -> None:
    """Place the image with its top-left corner at a 1-based cell position.

    The cursor is parked back at the origin afterwards. Sixel leaves it
    wherever the image ended, and `rich` assumes it owns cursor state.
    """
    try:
        # Flush the *text* layer first. `rich` writes through
        # `sys.stdout` (a buffered TextIOWrapper) and this writes to
        # `sys.stdout.buffer` underneath it. Without this, rich's pending
        # frame is still sitting in the wrapper and gets flushed after the
        # image, painting blank cells straight over the picture -- which looks
        # exactly like sixel not working at all.
        sys.stdout.flush()
        buf = sys.stdout.buffer
        buf.write(f"\x1b[{row};{col}H".encode())
        buf.write(data)
        buf.write(b"\x1b[H")
        buf.flush()
        _log(f"emit row={row} col={col} bytes={len(data)}")
    except Exception as exc:                             # noqa: BLE001
        _log(f"emit FAILED {type(exc).__name__}: {exc}")
