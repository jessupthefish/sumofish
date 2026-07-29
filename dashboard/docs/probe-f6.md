## F6 — resvg vs rsvg-convert, and the seam rule

### Fidelity at 1144px (same SVG into both rasterisers)

| case | max channel delta | pixels differing >2/255 | verdict |
|---|---|---|---|
| empty | 41 | 3450 (0.2636%) | INSPECT |
| start | 108 | 25630 (1.9584%) | INSPECT |
| midgame | 108 | 17063 (1.3038%) | INSPECT |
| check-flipped | 108 | 25646 (1.9596%) | INSPECT |

Worst differing fraction: 1.9596%. Threshold for R6 is 0.1%.

### The seam rule (empty board, interior scanline)

Counting pixels on a mid-board scanline that are neither square colour.
Python measured 8/line at 1152 and 0 at 1144 through rsvg.

| size | mod 26 | resvg strays | rsvg strays |
|---|---|---|---|
| 1144 | 0 | 0 | 0 |
| 1152 | 8 | 4 | 6 |
| 1170 | 0 | 0 | 0 |
| 1180 | 10 | 6 | 6 |
| 1196 | 0 | 0 | 0 |

`snap26(1152) = 1144`

### Where the differences are

AA gamma differs between any two rasterisers, so the raw count above is the wrong
question. These two are the right ones.

| case | structural (no 3x3 match) | box4 max delta | box4 differing >2 |
|---|---|---|---|
| empty | 1672 (0.1278%) | 10 | 411 (0.5025%) |
| start | 13690 (1.0460%) | 16 | 3312 (4.0491%) |
| midgame | 9337 (0.7134%) | 15 | 2351 (2.8742%) |
| check-flipped | 13703 (1.0470%) | 16 | 3323 (4.0625%) |

Dumped `midgame` at 390px to `docs/f6-{resvg,rsvg,diff}.png`. The diff is amplified 8x.
