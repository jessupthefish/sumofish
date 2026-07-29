## F1/F2/F3 — kitty graphics in this terminal

- `TERM=xterm-256color`
- `KONSOLE_VERSION=260403`
- `$TERM_PROGRAM=(unset)`

**DA1** (`ESC[c`) → `<ESC>[?62;1;4c`  
sixel claimed (a `4` in the list): **yes**

**F1a — capability query** `a=q,f=24` → `<ESC>_Gi=31;OK<ESC>\`  
verdict: **kitty graphics SUPPORTED**

**F2 — raw RGBA** `a=q,f=32` → `<ESC>_Gi=32;OK<ESC>\`  
verdict: **f=32 accepted; PNG encode is optional**

**F1b — PNG** `a=q,f=100` → `<ESC>_Gi=33;OK<ESC>\`  
verdict: **f=100 accepted**

**F1c — 4-chunk transmission** (64x64 RGB, 4096B chunks) → `<ESC>_Gi=34;OK<ESC>\`  
verdict: **chunking works**

**F1d — display**: 240x240 pattern placed at row 3 col 3 via `a=T,t=d,f=24,q=2`.
Expect a magenta border, a green diagonal and a flat grey field. Any banding,
speckle or colour shift would mean the transfer is being quantised.

**F3 — sibling text** written at column 40 for rows 4..14, i.e. on the rows the
image occupies. If the picture is intact, the region tolerates neighbouring text
and `CellDiffOption::Skip` only has to stop ratatui writing *inside* the rect.

**F1e — delete** `a=d,d=i,i=41` sent. Gone means placement ids can be reused
without `ESC[2J`. Still there means `ESC[2J` on geometry change stays the
eraser, exactly as today, and R3 has failed.

