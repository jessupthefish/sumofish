"""Gruvbox, and the small number of decisions that come with it.

Two rules the rest of the dashboard depends on:

**Every styled run names its background explicitly.** `rich`'s `Live` redraws
by diffing frames, so a block character painted with an implicit background
leaves the previous frame's colour behind when the new frame is narrower. That
is the whole reason the old sparklines read as static rather than as a trend.

**Board squares and highlights come in light/dark pairs.** Tinting a
highlighted square with one flat colour destroys the checker pattern exactly
where the eye is being asked to look. Highlighting keeps the pattern and shifts
the hue instead.
"""

BG        = "#282828"   # gruvbox dark0
BG_SOFT   = "#32302f"
BG_HARD   = "#1d2021"
FG        = "#ebdbb2"
DIM       = "#928374"
FAINT     = "#665c54"

ACCENT    = "#fabd2f"   # yellow: ours, the thing to look at
GOOD      = "#b8bb26"   # green
BAD       = "#fb4934"   # red
INFO      = "#83a598"   # blue
COOL      = "#8ec07c"   # aqua
WARM      = "#fe8019"   # orange
PLUM      = "#d3869b"   # purple

# Board. Warm neutral pair, far enough apart to read at a glance and neither
# close in luminance to a piece fill.
BOARD_LIGHT = "#bdae93"
BOARD_DARK  = "#7c6f64"
LAST_LIGHT  = "#a9a03f"   # last move: the same pair shifted green
LAST_DARK   = "#79762f"
CHECK_LIGHT = "#cc5b47"   # king in check: shifted red
CHECK_DARK  = "#9d4436"
CHECK_RING  = "#fb4934"   # and a hard ring on top, see board._ring

# Pieces. Fill plus a contrasting outline, so a black piece stays legible on a
# dark square and a white piece on a light one.
# cburnett is two-tone: a white piece is a light fill with a dark outline, a
# black piece is a dark fill with light detail lines. These are the two inks
# each gets. Pushed further apart than gruvbox's own fg/bg pair because the
# outline is sub-pixel at this size and needs all the contrast it can get.
PIECE_W      = "#f9f5d7"   # gruvbox fg0_hard, near white
PIECE_W_EDGE = "#20211f"
PIECE_B      = "#16181a"
PIECE_B_EDGE = "#ebdbb2"

# Track quality, in the sense a plot display means it: is this number being
# fed, coasting on its last value, or gone. Used everywhere a field is drawn.
LIVE    = GOOD
COAST   = ACCENT
LOST    = BAD

# The eval bar. Chess convention: White's share at the bottom, Black's at the
# top, so "the bar is going up" means the same thing here as on lichess.
EVAL_WHITE = "#f9f5d7"   # further apart than fg/bg, so the boundary is crisp
EVAL_BLACK = "#1d2021"
EVAL_MID   = "#7c6f64"   # the level line: must read against both of the above
EVAL_EDGE  = "#504945"   # the bar's own outline, so it is a gauge not a smear

# Gauges get a cap. On a 2560px-wide terminal an "as wide as the panel" bar is
# 140 characters, which puts its own label so far from its fill that the eye
# cannot associate the two. A bar is a comparison, not a decoration.
GAUGE_MAX = 44

BLOCKS  = " ▏▎▍▌▋▊▉█"     # eighth-width bars, for horizontal gauges
SPARK   = "▁▂▃▄▅▆▇█"      # eighth-height bars, for sparklines
