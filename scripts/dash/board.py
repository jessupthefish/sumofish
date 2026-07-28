"""Drawing a chess board into text cells.

The pieces are **cburnett**, the set lichess draws by default, rasterised from
the SVGs `python-chess` already ships and composited per pixel with their
antialiasing intact. So the shapes on this board are the same shapes as on the
website, at whatever resolution the terminal can afford. See `make_sprites.py`
for how the artwork gets here and `ink.py` for how a two-tone sprite becomes
two cell colours.

Five scales, chosen by how much room the terminal actually has. A square of
`w` by `h` cells is a `w` by `h*2` pixel bitmap, because `▀` stacks two pixels
in one cell:

    pixel3  24x12 cells, a 24x24 sprite. Wants ~200 columns and ~100 rows, so
            it needs a smaller font than the default before it comes up.
    pixel2  16x8 cells, a 16x16 sprite. The good one, and what a maximised
            terminal on a 1440p screen gets.
    pixel1  12x6 cells, a 12x12 sprite. The smallest size at which cburnett is
            still legible; below it the king and queen collapse into two blobs,
            so the smaller scales draw a font glyph instead of a worse bitmap.
    glyph3  8x4 cells, a unicode chess figure centred in a large square.
    glyph2  4x2 cells, the same at ordinary size.
    glyph1  2x1 cells. The compact fallback, still square, because two cells
            wide by one tall is almost exactly 1:1.

Everything coordinate-dependent goes through `_squares()`, which yields squares
in draw order for the current orientation. Highlights, the check marker and the
coordinate labels all read from the same iterator, so flipping the board for
Black cannot leave an overlay behind on the wrong file. That has to be
structural: it is the classic way a flipped board goes subtly wrong.
"""

from __future__ import annotations

import chess
from rich.text import Text

from .ink import blend
from .sprites import SPRITES
from .theme import (
    BOARD_DARK, BOARD_LIGHT, CHECK_DARK, CHECK_LIGHT, CHECK_RING, DIM,
    LAST_DARK, LAST_LIGHT, PIECE_B, PIECE_B_EDGE, PIECE_W, PIECE_W_EDGE,
)

FILES = "abcdefgh"


def _ring(px: tuple[int, int], width: int, height: int) -> bool:
    x, y = px
    return x == 0 or x == width - 1 or y == 0 or y == height - 1

# (cells wide, cells tall) per square, label gutter width, has a bitmap.
# For the bitmap scales the sprite is cells_wide x cells_tall*2 pixels, which
# has to be a size `make_sprites.py` generated: 8, 16 or 24.
SCALES = {
    "pixel3": (24, 12, 4, True),
    "pixel2": (16, 8, 4, True),
    "pixel1": (12, 6, 4, True),
    "glyph3": (8, 4, 3, False),
    "glyph2": (4, 2, 3, False),
    "glyph1": (2, 1, 2, False),
}
ORDER = ("pixel3", "pixel2", "pixel1", "glyph3", "glyph2", "glyph1")

FIGURES = {"K": "♚", "Q": "♛", "R": "♜", "B": "♝", "N": "♞", "P": "♟"}


def pick_scale(width: int, height: int) -> str:
    """The largest board that fits in the space the layout gave us."""
    for name in ORDER:
        w, h = board_size(name)
        if width >= w and height >= h:
            return name
    return "glyph1"


def board_size(scale: str) -> tuple[int, int]:
    """(columns, rows) a board at this scale occupies, labels included."""
    w, h, gutter, _zoom = SCALES[scale]
    return w * 8 + gutter, h * 8 + 1


def _squares(flip: bool):
    """Squares in draw order: left to right, top row first."""
    ranks = range(8) if flip else range(7, -1, -1)
    files = range(7, -1, -1) if flip else range(8)
    for r in ranks:
        yield r, [(chess.square(f, r), f) for f in files]


def highlight_squares(board: chess.Board, last: chess.Move | None) -> set[int]:
    """Which squares to mark as "this is what just changed".

    Castling is the case that catches people out: highlighting only the king's
    two squares leaves the rook appearing to have teleported for no reason, so
    the rook's origin and destination go in too.
    """
    if last is None:
        return set()
    marks = {last.from_square, last.to_square}
    piece = board.piece_at(last.to_square)
    if piece and piece.piece_type == chess.KING:
        delta = chess.square_file(last.to_square) - chess.square_file(last.from_square)
        rank = chess.square_rank(last.from_square)
        if delta == 2:      # short castling: rook h -> f
            marks |= {chess.square(7, rank), chess.square(5, rank)}
        elif delta == -2:   # long castling: rook a -> d
            marks |= {chess.square(0, rank), chess.square(3, rank)}
    return marks


def square_colours(sq: int, marked: bool, in_check: bool) -> str:
    light = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1
    if in_check:
        return CHECK_LIGHT if light else CHECK_DARK
    if marked:
        return LAST_LIGHT if light else LAST_DARK
    return BOARD_LIGHT if light else BOARD_DARK


# Rendering a 128x64 pixel board costs ~5ms, and the render loop runs at 8fps
# against a position that changes at most once a second. Keyed on everything
# that can change what is drawn, so a stale frame is not reachable.
_CACHE: dict = {}


def render(
    board: chess.Board,
    flip: bool = False,
    last: chess.Move | None = None,
    scale: str = "pixel2",
) -> Text:
    key = (board.fen(), flip, last.uci() if last else None, scale)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    if len(_CACHE) > 8:
        _CACHE.clear()
    out = _CACHE[key] = _render(board, flip, last, scale)
    return out


def _render(board, flip, last, scale) -> Text:
    marked = highlight_squares(board, last)
    checked = board.king(board.turn) if board.is_check() else None
    if SCALES[scale][3]:
        return _render_pixel(board, flip, marked, checked, scale)
    return _render_glyph(board, flip, marked, checked, scale)


def _render_pixel(board, flip, marked, checked, scale) -> Text:
    """One `▀` per cell: foreground is the upper pixel, background the lower.

    The sprite is cburnett, the set lichess draws by default, rasterised to
    exactly this pixel grid by `make_sprites.py`. Its antialiasing is kept and
    composited against whatever colour the square underneath happens to be,
    which is why a piece looks the same on a light square, a dark square, and a
    highlighted one instead of carrying a halo from whichever it was drawn on.
    """
    cw, ch, gutter = SCALES[scale][:3]
    px_h = ch * 2                       # pixel rows in one square, = sprite size
    out = Text(no_wrap=True)
    for rank, row in _squares(flip):
        # Each rank is `ch` cell-rows tall; each cell-row holds two pixel rows.
        for cell_row in range(ch):
            label = f"{rank + 1}".center(gutter) if cell_row == ch // 2 - 1 else " " * gutter
            out.append(label, style=DIM)
            for sq, _f in row:
                check = sq == checked
                bg = square_colours(sq, sq in marked, check)
                piece = board.piece_at(sq)
                if piece is None and not check:
                    out.append(" " * cw, style=f"on {bg}")
                    continue
                data = SPRITES.get((px_h, piece.symbol())) if piece else None
                if piece is not None and piece.color == chess.WHITE:
                    dark, light = PIECE_W_EDGE, PIECE_W
                else:
                    dark, light = PIECE_B, PIECE_B_EDGE
                for x in range(cw):
                    ty, by = cell_row * 2, cell_row * 2 + 1
                    if data is None:
                        fg_c = bg_c = bg
                    else:
                        ti, bi = (ty * cw + x) * 2, (by * cw + x) * 2
                        fg_c = blend(bg, dark, light, data[ti], data[ti + 1])
                        bg_c = blend(bg, dark, light, data[bi], data[bi + 1])
                    # A checked king covers almost all of its own square, so
                    # tinting the background hides the warning behind the
                    # piece that the warning is about. Draw a ring on the
                    # square's outer pixels instead, over everything.
                    if check:
                        if _ring((x, ty), cw, px_h):
                            fg_c = CHECK_RING
                        if _ring((x, by), cw, px_h):
                            bg_c = CHECK_RING
                    # Both halves the same colour: emit a plain space so the
                    # cell has one style instead of a redundant glyph.
                    if fg_c == bg_c:
                        out.append(" ", style=f"on {bg_c}")
                    else:
                        out.append("▀", style=f"{fg_c} on {bg_c}")
            out.append("\n")
    return _files_row(out, flip, cw, gutter)


def _render_glyph(board, flip, marked, checked, scale) -> Text:
    cw, ch, gutter = SCALES[scale][:3]
    out = Text(no_wrap=True)
    for rank, row in _squares(flip):
        for cell_row in range(ch):
            show = cell_row == (ch - 1) // 2
            label = f"{rank + 1}".center(gutter) if show else " " * gutter
            out.append(label, style=DIM)
            for sq, _f in row:
                bg = square_colours(sq, sq in marked, sq == checked)
                piece = board.piece_at(sq)
                if piece is None or not show:
                    out.append(" " * cw, style=f"on {bg}")
                    continue
                fg = PIECE_W if piece.color == chess.WHITE else PIECE_B
                glyph = FIGURES[piece.symbol().upper()]
                pad = cw - 1
                left = pad // 2
                out.append(" " * left, style=f"on {bg}")
                out.append(glyph, style=f"bold {fg} on {bg}")
                out.append(" " * (pad - left), style=f"on {bg}")
            out.append("\n")
    return _files_row(out, flip, cw, gutter)


def _files_row(out: Text, flip: bool, cw: int, gutter: int) -> Text:
    out.append(" " * gutter)
    order = range(7, -1, -1) if flip else range(8)
    for f in order:
        out.append(FILES[f].center(cw), style=DIM)
    return out
