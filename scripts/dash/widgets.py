"""Small drawing primitives.

Two rules run through all of them.

**Every run names its background.** `rich`'s `Live` diffs frames, so a block
character drawn with an implicit background leaves the previous frame's colour
behind wherever the new frame is shorter. That is what made the old sparklines
read as static.

**Every trend carries its endpoints as text.** A shaded trace tells you the
shape and nothing else. A trader's chart always prints the high and the low
beside it, because the shape without a scale is decoration. So `sparkline`
returns its range and the callers print it.
"""

from __future__ import annotations

from rich.text import Text

from .theme import BG, BLOCKS, COAST, DIM, LIVE, LOST, SPARK
from .state import COAST as T_COAST, LIVE as T_LIVE, LOST as T_LOST, ago

TRACK_COLOUR = {T_LIVE: LIVE, T_COAST: COAST, T_LOST: LOST}
TRACK_MARK = {T_LIVE: "●", T_COAST: "◐", T_LOST: "○"}


def track_tag(field, label: str | None = None) -> Text:
    """The provenance of one field: a state dot, and how old it is.

    This is the thing the old dashboard had no way to say. A number with no
    tag beside it is a number you cannot act on, because you cannot tell
    whether it is current.
    """
    t = field.track
    out = Text(no_wrap=True)
    out.append(TRACK_MARK[t], style=f"{TRACK_COLOUR[t]} on {BG}")
    if label:
        out.append(f" {label}", style=f"{DIM} on {BG}")
    if t != T_LIVE:
        out.append(f" {ago(field.age)}", style=f"{TRACK_COLOUR[t]} on {BG}")
    return out


def sparkline(values, width: int = 24, style: str = DIM) -> tuple[Text, float, float]:
    """A trace, plus the low and high it was drawn between."""
    vals = [v for v in values[-width:] if v is not None]
    out = Text(no_wrap=True)
    if len(vals) < 2:
        return out, 0.0, 0.0
    lo, hi = min(vals), max(vals)
    span = hi - lo
    for v in vals:
        idx = 0 if span < 1e-12 else int((v - lo) / span * (len(SPARK) - 1))
        out.append(SPARK[idx], style=f"{style} on {BG}")
    return out, lo, hi


def bar(fraction: float, width: int, fg: str, bg: str = BG) -> Text:
    """A horizontal gauge with eighth-of-a-cell resolution.

    Sub-cell resolution matters more than it looks: at whole-cell steps a
    twenty-column bar has twenty states, so a slow-moving quantity appears
    frozen and then jumps. Eighths give it a hundred and sixty.
    """
    fraction = max(0.0, min(1.0, fraction))
    total = fraction * width
    full = int(total)
    rest = int((total - full) * 8)
    out = Text(no_wrap=True)
    out.append("█" * full, style=f"{fg} on {bg}")
    if full < width:
        out.append(BLOCKS[rest], style=f"{fg} on {bg}")
        out.append(" " * (width - full - 1), style=f"on {bg}")
    return out


# Bottom-aligned partial blocks: one cell in nine states rather than two.
EIGHTHS = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
# Left-aligned partial blocks, for a gauge that fills sideways.
EIGHTHS_H = " \u258f\u258e\u258d\u258c\u258b\u258a\u2589\u2588"


def evalbar_h(value: float | None, width: int) -> Text:
    """A horizontal evaluation gauge, filling from the left for our side.

    Horizontal because it has to live on its own row. The board is a sixel
    image and `rich` diffs by line: any styled cell sharing a row with the
    picture means the whole line gets rewritten when it changes, and the
    unstyled padding written over the image erases it a row at a time. A
    gauge that animates does that on every frame, which is what turned the
    board to confetti between moves.

    Eighths again, so a one percent change moves the edge instead of landing
    on the same cell as the last one.
    """
    from .theme import EVAL_BLACK, EVAL_WHITE

    line = Text(no_wrap=True)
    if value is None:
        return line
    total = width * 8
    filled = int(round(max(0.0, min(1.0, value)) * total))
    for c in range(width):
        k = max(0, min(8, filled - c * 8))
        if k == 8:
            line.append(" ", style=f"on {EVAL_WHITE}")
        elif k == 0:
            line.append(" ", style=f"on {EVAL_BLACK}")
        else:
            line.append(EIGHTHS_H[k], style=f"{EVAL_WHITE} on {EVAL_BLACK}")
    return line


def evalbar(value: float | None, height: int, width: int = 2) -> list[Text]:
    """A vertical gauge for the side at the bottom of the board.

    Eighths, not halves. A seventy-row bar drawn in half-blocks has 140
    positions, which sounds like plenty until you notice that most of a game
    is spent between 0.45 and 0.60 -- twenty of them -- and every small change
    lands on the same one. Eighths give 560, so the boundary actually moves
    when the evaluation does, which is the whole point of watching it.

    The caller passes the *eased* value, so the boundary slides rather than
    jumping. See `state.Smooth`.
    """
    from .theme import EVAL_BLACK, EVAL_WHITE

    if value is None:
        # Unstyled, so there is nothing there at all rather than a dark column
        # down the side of the board.
        return [Text(" " * width) for _ in range(height)]

    total = height * 8
    filled = int(round(max(0.0, min(1.0, value)) * total))
    rows: list[Text] = []
    for r in range(height):
        # Rows are drawn top down; the gauge fills bottom up.
        base = (height - 1 - r) * 8
        k = max(0, min(8, filled - base))
        line = Text(no_wrap=True)
        for _ in range(width):
            if k == 0:
                line.append(" ", style=f"on {EVAL_BLACK}")
            elif k == 8:
                line.append(" ", style=f"on {EVAL_WHITE}")
            else:
                line.append(EIGHTHS[k], style=f"{EVAL_WHITE} on {EVAL_BLACK}")
        rows.append(line)
    return rows


# Braille gives one cell a 2x4 grid of dots, so a chart drawn in it has four
# times the vertical resolution of half-blocks and twice the horizontal. The
# cost is colour: a braille cell has one foreground, so a cell cannot hold two
# differently coloured dots.
BRAILLE_BASE = 0x2800
BRAILLE_DOTS = ((0x01, 0x02, 0x04, 0x40), (0x08, 0x10, 0x20, 0x80))


def _catmull_rom(points: list[float], samples: int) -> list[float]:
    """Resample a series through a smooth curve rather than straight segments.

    With twenty moves stretched across two hundred columns, linear
    interpolation is visibly a run of straight lines meeting at corners, and
    the corners read as features of the evaluation that are not there.
    Catmull-Rom passes through every real point -- it invents shape between
    them, never at them -- which is the honest kind of smoothing for this:
    no data point is moved.
    """
    n = len(points)
    if n < 2 or samples < 2:
        return points[:]
    out = []
    for i in range(samples):
        t = i * (n - 1) / (samples - 1)
        k = min(int(t), n - 2)
        u = t - k
        p0 = points[max(0, k - 1)]
        p1, p2 = points[k], points[k + 1]
        p3 = points[min(n - 1, k + 2)]
        out.append(0.5 * ((2 * p1)
                          + (-p0 + p2) * u
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * u * u * u))
    return out


def curve_chart(values, width: int, height: int, mid: float = 0.5) -> list[Text]:
    """The evaluation over a whole game, as a smooth line.

    Fixed 0..1 scale, never auto-scaled to the game's own range: renormalising
    makes a dead-level draw look as dramatic as a collapse, which is the
    standard way an evaluation chart lies. The drawn midline is 0.50 and
    vertical distance from it means the same thing in every game.
    """
    from .theme import EVAL_MID, GOOD as G, BAD as B

    src = [v for v in values if v is not None]
    px_w, px_h = width * 2, height * 4
    if len(src) < 2:
        return [Text("", no_wrap=True) for _ in range(height)]

    ys = _catmull_rom(src, px_w)
    cells: dict[tuple[int, int], int] = {}
    colours: dict[tuple[int, int], str] = {}

    def plot(x: int, y: int, colour: str) -> None:
        y = max(0, min(px_h - 1, y))
        cx, cy = x // 2, y // 4
        cells[(cx, cy)] = cells.get((cx, cy), 0) | BRAILLE_DOTS[x % 2][y % 4]
        # Last writer wins the cell's colour. The trace crosses level rarely,
        # so the only cells this can misreport are the crossing ones.
        colours[(cx, cy)] = colour

    prev = None
    for x, v in enumerate(ys):
        y = int(round((1.0 - max(0.0, min(1.0, v))) * (px_h - 1)))
        colour = G if v >= mid else B
        lo, hi = (y, y) if prev is None else (min(prev, y), max(prev, y))
        for yy in range(lo, hi + 1):        # join consecutive samples
            plot(x, yy, colour)
        prev = y

    mid_y = int(round((1.0 - mid) * (px_h - 1)))
    out: list[Text] = []
    for cy in range(height):
        line = Text(no_wrap=True)
        for cx in range(width):
            bits = cells.get((cx, cy), 0)
            if bits:
                line.append(chr(BRAILLE_BASE | bits),
                            style=f"{colours[(cx, cy)]} on {BG}")
            elif cy == mid_y // 4:
                # Level, drawn only where the trace is not: a braille cell has
                # one colour, so the two cannot share.
                line.append(chr(BRAILLE_BASE | BRAILLE_DOTS[0][mid_y % 4]
                                | BRAILLE_DOTS[1][mid_y % 4]),
                            style=f"{EVAL_MID} on {BG}")
            else:
                line.append(" ", style=f"on {BG}")
        out.append(line)
    return out


def ladder(rows, width: int, fg: str, dim: str = DIM) -> list[Text]:
    """A depth ladder: candidates sorted by the resource actually committed.

    Not by score. The search picks its move by visit count, so visit share is
    what the decision is made on, and a bar proportional to it shows how
    convinced the engine is rather than merely which move won. A single long
    bar is a forced move; five stubs of equal length mean it has no idea.
    """
    out: list[Text] = []
    if not rows:
        return out
    top = max((r[1] for r in rows), default=1) or 1
    for i, (san, visits, q, prior) in enumerate(rows):
        line = Text(no_wrap=True)
        chosen = i == 0
        line.append(f"{san:<7}", style=f"{'bold ' + fg if chosen else dim} on {BG}")
        line.append_text(bar(visits / top, width, fg if chosen else dim))
        line.append(f" {visits:>6}", style=f"{dim} on {BG}")
        line.append(f"  q {q:.3f}", style=f"{dim} on {BG}")
        line.append(f"  p {prior:.3f}", style=f"{dim} on {BG}")
        out.append(line)
    return out
