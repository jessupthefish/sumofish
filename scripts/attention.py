#!/usr/bin/env python
"""What each attention head looks at, drawn on the board.

    scripts/attention.py                                  # value net, startpos
    scripts/attention.py --fen "<fen>" --out /tmp/a.html
    scripts/attention.py --checkpoint runs/policy.pt
    scripts/attention.py --check-only                     # just the self-test

Writes one standalone HTML file and prints its path. Open it in a browser:
pick a layer and head, click a square to make it the query, and the board
shades by how much that query attends to every other position.

## Why this is not a two-line hook

`MultiHeadAttention.forward` calls `F.scaled_dot_product_attention`, which is
fused. It never materialises the [T, T] probability matrix, so a forward hook
on the module gets you its input and its output and no attention weights at
all. There is no flag to make it hand them over.

So the weights are recomputed here from the same q and k the real forward pass
used: a hook captures `block.ln_attn`'s output, which IS the tensor `block.attn`
was called on, and this file then applies that block's own `q`/`k` projections
to it. Nothing is re-implemented except the softmax SDPA hid.

That leaves one way to be quietly wrong -- if this file's scale, mask or head
reshape disagreed with SDPA's, the maps would still look like attention maps.
So it is checked rather than trusted: `attention_maps` finishes by feeding its
own probabilities through `v` and `out` and comparing against the captured real
output of `block.attn`. A mismatch raises. That check is the reason to believe
any picture this script draws, and `--check-only` runs it on its own.

## Reading it honestly

Attention is not explanation. A head that attends to a square is not a head
that used it: information gets moved by the value projection and the residual
stream, and a large weight on a token whose value vector is near zero does
nothing at all. This shows where information could flow, not where it did.
`scripts/saliency.py` answers the causal version of the question by deleting
pieces and measuring what changes, and the two disagreeing is informative.

Two structural facts to keep in view while reading:

- **The readout is position 77**, the last digit of the fullmove counter,
  which for most positions is a padding '.' with no information in it. It is
  the only position whose output is read. Its row is the one that feeds the
  head, so it is the default view.
- **The mask is causal** (`cfg.causal`, and every shipped checkpoint has it
  True), so position i sees only 0..i. Square a8 is token 2 and h1 is token 65,
  which means a8 cannot see any other square and h1 sees all of them. The
  lower-right triangle being empty is the mask, not the model. `model.py` calls
  flipping this "the first experiment worth running".
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import viz  # noqa: E402
from sumofish.model import ChessTransformer  # noqa: E402


@torch.no_grad()
def attention_maps(
    model: ChessTransformer, tokens: torch.Tensor, tol: float = 2e-4
) -> tuple[np.ndarray, float]:
    """Attention probabilities [L, H, T, T], and the reconstruction error.

    `tol` is the gate on that error. It is not a formality: the whole file is
    a re-derivation of something the fused kernel would not return, and this is
    the only thing standing between a wrong reshape and a plausible picture.
    """
    cfg = model.cfg
    ln_out: list[torch.Tensor] = []
    attn_out: list[torch.Tensor] = []

    handles = []
    for block in model.blocks:
        handles.append(block.ln_attn.register_forward_hook(
            lambda m, i, o: ln_out.append(o.detach())))
        handles.append(block.attn.register_forward_hook(
            lambda m, i, o: attn_out.append(o.detach())))
    try:
        model(tokens)
    finally:
        for h in handles:
            h.remove()

    nh, hd = cfg.num_heads, cfg.head_dim
    maps = []
    worst = 0.0
    for block, x, real in zip(model.blocks, ln_out, attn_out):
        b, t, _ = x.shape
        shape = lambda w: w(x).view(b, t, nh, hd).transpose(1, 2)  # noqa: E731
        q, k, v = shape(block.attn.q), shape(block.attn.k), shape(block.attn.v)

        # SDPA's default scale, spelled out. `model.py` notes this matches
        # upstream; if that default ever changes the check below catches it.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        if cfg.causal:
            mask = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
            scores = scores.masked_fill(~mask, float("-inf"))
        p = scores.softmax(dim=-1)

        # The check. Same path the real module takes, from this file's own
        # probabilities: if the mask, the scale or the head reshape is wrong,
        # this diverges.
        y = (p @ v).transpose(1, 2).reshape(b, t, nh * hd)
        worst = max(worst, (block.attn.out(y) - real).abs().max().item())
        maps.append(p[0].float().cpu().numpy())

    err = worst
    if err > tol:
        raise RuntimeError(
            f"attention reconstruction disagrees with the model by {err:.2e} "
            f"(tolerance {tol:.0e}). The maps would be wrong; refusing to draw "
            f"them. Check the scale, the causal mask and the head reshape "
            f"against MultiHeadAttention.forward."
        )
    return np.stack(maps), err


def render(board: chess.Board, maps: np.ndarray, info: dict, err: float,
           fen: str) -> str:
    labels = viz.token_labels(board)
    payload, scales = viz.quantise_rows(maps)
    L, H, T, _ = maps.shape

    sq_index = {}
    for sq in chess.SQUARES:
        sq_index[chess.square_name(sq)] = viz.SQUARE_TO_MODEL_INDEX[sq]

    # Every position that is not one of the 64 squares, so nothing is hidden by
    # choosing a board-shaped picture.
    others = [i for i, l in enumerate(labels) if l["kind"] != "square"]

    data = viz.js_json({
        "L": L, "H": H, "T": T,
        "b64": payload,
        "scales": scales,
        "labels": labels,
        "others": others,
        "sqIndex": sq_index,
        "readout": viz.READOUT,
    })

    meta = (
        f"<b>{Path(info['path']).name}</b> &middot; {info['kind']} net &middot; "
        f"{info['params']/1e6:.1f}M params &middot; step {info['step']} &middot; "
        f"{'EMA' if info['ema'] else 'raw'} &middot; {L} layers x {H} heads "
        f"&middot; causal={info['causal']}<br>"
        f"<code>{fen}</code><br>"
        f"reconstruction error vs the real forward pass: <b>{err:.2e}</b>"
    )

    body = f"""
<h1>attention</h1>
<div class="meta">{meta}</div>
<div class="cols">
  <div class="panel">
    <h2>board</h2>
    <div class="boardwrap">{viz.board_svg(board)}<div class="overlay" id="ov"></div></div>
    <div class="row" style="margin-top:10px"><span class="lab">non-board</span>
      <div class="chips" id="chips"></div></div>
    <div class="note" id="note"></div>
  </div>
  <div class="panel">
    <h2>head</h2>
    <div class="row"><span class="lab">layer</span><span id="layers"></span></div>
    <div class="row"><span class="lab">head</span><span id="heads"></span></div>
    <div class="row"><span class="lab">query</span><span id="modes"></span></div>
    <h2 style="margin-top:14px">full {T}x{T} matrix</h2>
    <canvas id="mat" width="{T}" height="{T}"
            style="width:390px;height:390px"></canvas>
    <div class="note">Row = query, column = key. The empty upper-right triangle
      is the causal mask. Click a row to select that query.</div>
  </div>
</div>
<script>
const D = {data};
const raw = atob(D.b64), Q = new Uint8Array(raw.length);
for (let i = 0; i < raw.length; i++) Q[i] = raw.charCodeAt(i);

let layer = 0, head = 0, mode = "readout", query = D.readout;

// quantised back to a probability: per-row scale, see viz.quantise_rows
function at(l, h, i, j) {{
  const base = ((l * D.H + h) * D.T + i) * D.T;
  return Q[base + j] / 255 * D.scales[l][h][i];
}}
function row(l, h, i) {{
  const out = new Float64Array(D.T);
  for (let j = 0; j < D.T; j++) out[j] = at(l, h, i, j);
  return out;
}}
function received(l, h) {{      // column sums: attention each key collects
  const out = new Float64Array(D.T);
  for (let i = 0; i < D.T; i++)
    for (let j = 0; j < D.T; j++) out[j] += at(l, h, i, j);
  return out;
}}
function current() {{
  return mode === "received" ? received(layer, head) : row(layer, head, query);
}}
function paint(el, v, max) {{
  const a = max > 0 ? v / max : 0;
  el.style.background = `rgba(254,128,25,${{(0.92 * a).toFixed(3)}})`;
}}

const ov = document.getElementById("ov"), chips = document.getElementById("chips");
const cells = {{}};
for (let r = 7; r >= 0; r--) for (let f = 0; f < 8; f++) {{
  const name = "abcdefgh"[f] + (r + 1);
  const c = document.createElement("div");
  c.className = "cell";
  c.innerHTML = '<div class="paint"></div><div class="val"></div>';
  c.title = name;
  c.onclick = () => {{ mode = "row"; query = D.sqIndex[name]; draw(); }};
  ov.appendChild(c); cells[name] = c;
}}
const chipEls = {{}};
for (const i of D.others) {{
  const e = document.createElement("div");
  e.className = "chip";
  e.innerHTML = '<div class="paint"></div><span>' + D.labels[i].text + '</span>';
  e.onclick = () => {{ mode = "row"; query = i; draw(); }};
  chips.appendChild(e); chipEls[i] = e;
}}
function buttons(host, n, set, label) {{
  const el = document.getElementById(host);
  el.innerHTML = "";
  for (let i = 0; i < n; i++) {{
    const b = document.createElement("button");
    b.textContent = label ? label(i) : i;
    b.onclick = () => {{ set(i); draw(); }};
    b.dataset.i = i;
    el.appendChild(b);
  }}
}}
buttons("layers", D.L, i => layer = i);
buttons("heads", D.H, i => head = i);
const MODES = [["readout", "readout (pos " + D.readout + ")"],
               ["received", "received (column sum)"]];
{{
  const el = document.getElementById("modes");
  for (const [k, text] of MODES) {{
    const b = document.createElement("button");
    b.textContent = text; b.dataset.m = k;
    b.onclick = () => {{ mode = k; if (k === "readout") query = D.readout; draw(); }};
    el.appendChild(b);
  }}
}}

const ctx = document.getElementById("mat").getContext("2d");
document.getElementById("mat").onclick = e => {{
  const r = e.target.getBoundingClientRect();
  mode = "row";
  query = Math.min(D.T - 1, Math.floor((e.clientY - r.top) / r.height * D.T));
  draw();
}};
function drawMatrix() {{
  const img = ctx.createImageData(D.T, D.T);
  for (let i = 0; i < D.T; i++) for (let j = 0; j < D.T; j++) {{
    // per-row normalised, or row 0 (a single 1.0) would be the only bright pixel
    const base = ((layer * D.H + head) * D.T + i) * D.T;
    const a = D.scales[layer][head][i] > 0 ? Q[base + j] / 255 : 0;
    const o = (i * D.T + j) * 4;
    img.data[o] = 40 + 214 * a; img.data[o + 1] = 40 + 88 * a;
    img.data[o + 2] = 40 - 15 * a; img.data[o + 3] = 255;
  }}
  if (mode !== "received") for (let j = 0; j < D.T; j++) {{
    const o = (query * D.T + j) * 4;
    img.data[o + 2] = Math.min(255, img.data[o + 2] + 120);
  }}
  ctx.putImageData(img, 0, 0);
}}
function draw() {{
  const v = current();
  let max = 0;
  for (const x of v) max = Math.max(max, x);
  for (const name in cells) {{
    const i = D.sqIndex[name], c = cells[name];
    paint(c.querySelector(".paint"), v[i], max);
    c.querySelector(".val").textContent = v[i] >= 0.005 ? v[i].toFixed(2) : "";
    c.classList.toggle("sel", mode !== "received" && i === query);
  }}
  for (const i of D.others) {{
    paint(chipEls[i].querySelector(".paint"), v[i], max);
    chipEls[i].classList.toggle("sel", mode !== "received" && +i === query);
  }}
  for (const [host, cur] of [["layers", layer], ["heads", head]])
    for (const b of document.getElementById(host).children)
      b.classList.toggle("on", +b.dataset.i === cur);
  for (const b of document.getElementById("modes").children)
    b.classList.toggle("on", b.dataset.m === mode);

  let onBoard = 0;
  for (const name in cells) onBoard += v[D.sqIndex[name]];
  const total = v.reduce((a, b) => a + b, 0);
  const note = mode === "received"
    ? `layer ${{layer}} head ${{head}}: how much attention each position COLLECTS, `
      + `summed over all ${{D.T}} queries. Board squares take `
      + `${{(100 * onBoard / total).toFixed(1)}}% of it.`
    : `layer ${{layer}} head ${{head}}, query <b>${{D.labels[query].text}}</b> `
      + `(position ${{query}}). ${{(100 * onBoard / (total || 1)).toFixed(1)}}% of its `
      + `attention lands on board squares, the rest on the side-to-move, `
      + `castling, en-passant and clock tokens.`;
  document.getElementById("note").innerHTML = note;
  drawMatrix();
}}
draw();
</script>
"""
    return viz.html_page("sumofish attention", "", body)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default=str(ROOT / "runs" / "value.pt"),
                    help="default runs/value.pt; try runs/policy.pt")
    ap.add_argument("--fen", default=chess.STARTING_FEN)
    ap.add_argument("--out", default="/tmp/sumofish-attention.html")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tol", type=float, default=2e-4,
                    help="max allowed disagreement with the real forward pass")
    ap.add_argument("--check-only", action="store_true",
                    help="run the reconstruction check and exit, drawing nothing")
    args = ap.parse_args()

    board = chess.Board(args.fen)
    model, info = viz.load_model(args.checkpoint, device=args.device)
    tokens = viz.tokens_for(board, device=args.device)
    maps, err = attention_maps(model, tokens, tol=args.tol)

    print(f"{Path(args.checkpoint).name}: {info['layers']}L x {info['heads']}H, "
          f"{info['params']/1e6:.1f}M params, step {info['step']}")
    print(f"reconstruction error {err:.2e} (tolerance {args.tol:.0e})  OK")
    if args.check_only:
        return

    # Sanity that costs nothing and would catch a dead map: every row of every
    # head must sum to 1, because they are softmaxes.
    sums = maps.sum(axis=-1)
    assert abs(sums - 1.0).max() < 1e-4, f"rows do not sum to 1: {abs(sums-1).max()}"

    out = Path(args.out)
    out.write_text(render(board, maps, info, err, args.fen))
    print(f"{out}  ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
