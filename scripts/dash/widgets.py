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


def curve_chart(values, width: int, height: int, mid: float = 0.5) -> list[Text]:
    """The evaluation over a whole game, as a line rather than a sparkline.

    A sparkline compresses to one row and answers "is it going up". Given real
    vertical room the useful question is different: how far from level, how
    suddenly, and around which moves. So this keeps a fixed 0..1 scale with the
    drawn midline, which means the height of a swing is comparable between
    games rather than being renormalised to whatever happened in this one.

    Half-blocks give two vertical samples per row, so a ten-row chart resolves
    to twenty levels rather than ten.
    """
    from .theme import EVAL_MID, GOOD as G, BAD as B

    src = [v for v in values if v is not None]
    depth = height * 2
    rows = [[None] * width for _ in range(depth)]
    mid_level = int(round((1.0 - mid) * (depth - 1)))

    def level_at(col: int) -> int:
        """Linearly interpolate the series across the full panel width.

        Drawing one column per data point would leave a ten-move game as a
        stub in the corner of a hundred-column panel. Stretching means the
        chart always reads as a chart, and the point count is printed in the
        subtitle so the resolution is never implied to be better than it is.
        """
        if len(src) == 1:
            v = src[0]
        else:
            pos = col * (len(src) - 1) / max(1, width - 1)
            i = min(int(pos), len(src) - 2)
            v = src[i] + (src[i + 1] - src[i]) * (pos - i)
        return int(round((1.0 - max(0.0, min(1.0, v))) * (depth - 1))), v

    if len(src) >= 2:
        prev = None
        for x in range(width):
            level, v = level_at(x)
            colour = G if v >= mid else B
            # Fill between the previous level and this one, so a sharp swing
            # draws as a connected line rather than two detached dots.
            lo, hi = (level, level) if prev is None else (min(prev, level), max(prev, level))
            for y in range(lo, hi + 1):
                rows[y][x] = colour
            prev = level

    out: list[Text] = []
    for r in range(height):
        line = Text(no_wrap=True)
        top_row, bot_row = rows[r * 2], rows[r * 2 + 1]
        for x in range(width):
            top = top_row[x] or (EVAL_MID if r * 2 == mid_level else None)
            bot = bot_row[x] or (EVAL_MID if r * 2 + 1 == mid_level else None)
            if top and bot:
                line.append("▀", style=f"{top} on {bot}")
            elif top:
                line.append("▀", style=f"{top} on {BG}")
            elif bot:
                line.append("▄", style=f"{bot} on {BG}")
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
