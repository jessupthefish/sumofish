"""Turning a two-tone sprite plus a square colour into one cell's two colours.

The board draws every cell as `▀`: foreground is the upper pixel, background
the lower. So each pixel gets its own colour, and there is no palette limit to
work around. That is what lets the cburnett artwork keep its antialiasing
instead of being thresholded into a silhouette, and antialiasing is most of
what makes a sixteen-pixel piece read as a piece.

Two blends happen per pixel:

    ink    = lerp(dark_ink, light_ink, luminance)   which tone of the piece
    result = lerp(square,   ink,       alpha)       how much of it covers

Doing that twice per cell for a 128x64 board is 16k blends a frame, so the
results are memoised. The key space is small in practice: a rasterised piece
uses on the order of a hundred distinct (luminance, alpha) pairs, times a
handful of square colours, so the cache saturates within the first frame and
every frame after it is dictionary lookups.
"""

from __future__ import annotations

from functools import lru_cache


# Contrast stretch applied to sprite luminance before it picks an ink.
#
# cburnett draws a 1.5-unit outline in a 45-unit box. At a 16 pixel sprite that
# is half a pixel wide, so the rasteriser resolves it to a mid grey rather than
# a line, and a mid grey between a cream fill and a dark outline is a muddy tan
# very close to the light square. The piece goes soft exactly where its edge
# should be.
#
# Thickening the stroke before rasterising is the obvious fix and it is wrong:
# every path in the set carries the same stroke, so widening it closes the
# crown of the king and swallows the fill. Measured at stroke 2.2 and 3.1 the
# king loses its cross entirely.
#
# So the outline is recovered at display time instead. Everything below KNEE_LO
# is pure dark ink, everything above KNEE_HI pure light ink, and the band
# between is a linear ramp that keeps enough of the antialiasing for curves to
# stay smooth. Coverage (alpha) is untouched -- that is the silhouette, and
# hardening it would give the pieces jagged edges.
KNEE_LO, KNEE_HI = 90, 190


@lru_cache(maxsize=None)
def stretch(lum: int) -> float:
    if lum <= KNEE_LO:
        return 0.0
    if lum >= KNEE_HI:
        return 1.0
    return (lum - KNEE_LO) / (KNEE_HI - KNEE_LO)


@lru_cache(maxsize=None)
def rgb(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


@lru_cache(maxsize=None)
def blend(square: str, dark: str, light: str, lum: int, alpha: int) -> str:
    """The final hex colour of one sprite pixel drawn on one square colour."""
    if alpha == 0:
        return square
    sr, sg, sb = rgb(square)
    dr, dg, db = rgb(dark)
    lr, lg, lb = rgb(light)
    t = stretch(lum)
    ir = dr + (lr - dr) * t
    ig = dg + (lg - dg) * t
    ib = db + (lb - db) * t
    a = alpha / 255.0
    r = round(sr + (ir - sr) * a)
    g = round(sg + (ig - sg) * a)
    b = round(sb + (ib - sb) * a)
    return f"#{r:02x}{g:02x}{b:02x}"
