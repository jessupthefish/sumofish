#!/usr/bin/env python
"""What the evaluation actually depends on: delete a piece and measure.

    scripts/saliency.py                                   # value net, startpos
    scripts/saliency.py --fen "<fen>" --out /tmp/s.html
    scripts/saliency.py --checkpoint runs/policy.pt
    scripts/saliency.py --print                           # no HTML, just numbers

Writes one standalone HTML file with two views of the same position: which
pieces the network's judgement rests on, and what it thinks the move is.

## Why this exists next to attention.py

`attention.py` shows where information *could* flow. This shows where it *did*,
by intervening: take the position, remove one piece, ask the network again, and
report the difference. That is a causal measurement of the deployed model, not
an inference about its internals, and it needs no hooks and no assumptions
about the architecture. If the two ever disagree -- a head that stares at a
square whose removal changes nothing -- the attention map is the one that is
lying, and that disagreement is worth more than either picture alone.

The metric depends on which net is loaded, decided by the checkpoint's own
output size rather than by a flag:

- **value net** (64 HL-Gauss bins): win probability for the side to move.
  A removal's score is the signed change in it. Positive means the position
  got BETTER for whoever is to move once that piece was taken off, so your
  own pieces should read negative and the opponent's positive. That sign
  flip is the first thing to check on a new checkpoint, and a net that gets
  it backwards on a queen is broken in a way no loss curve shows.
- **policy net** (1968 actions): the distribution over legal moves. A
  removal's score is the KL divergence from the original distribution in
  bits, which is unsigned by construction -- "how much did this rewrite the
  network's mind", not "which way".

## Three honest limits

1. **A one-piece-removed position is off-distribution.** Both nets were trained
   on real Stockfish-annotated games and have never seen a board with no white
   king. The removal is a real intervention on the model; it is not a real
   chess position. Read it as sensitivity, not as chess.
2. **Castling rights are left alone.** Remove a rook and the FEN still claims
   the right to castle with it. Clearing them would be a second simultaneous
   change and would stop this being a one-variable experiment, so the
   inconsistency is deliberate and it is the model's problem to have an
   opinion about.
3. **This is one position.** Nothing here has an interval. A saliency map is
   an instrument for deciding what to look at next, in exactly the sense
   PHILOSOPHY.md means: it is never a result and no proposal should cite it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import viz  # noqa: E402
from sumofish.hlgauss import HLGauss  # noqa: E402
from sumofish.model import ChessTransformer  # noqa: E402
from sumofish.rules import terminal_value  # noqa: E402
from sumofish.tokenizer import MOVE_TO_ACTION, tokenize_board  # noqa: E402


@torch.no_grad()
def _logits(model: ChessTransformer, boards: list[chess.Board],
            device: str) -> torch.Tensor:
    x = torch.from_numpy(np.stack([tokenize_board(b) for b in boards])).long().to(device)
    return model(x).float()


def win_probability(model, hl, boards, device) -> np.ndarray:
    """Win probability for the side to move in each board."""
    return hl.expectation(_logits(model, boards, device)).cpu().numpy()


def move_distribution(model, board, device) -> list[tuple[chess.Move, float]]:
    """The policy's probability over the LEGAL moves, best first.

    Masked to legal before the softmax, which is `policy.py`'s point: unmasked,
    the model proposes moves that do not exist and there is no search to catch
    it. Masking after the softmax would leave the probabilities normalised over
    the illegal ones too and quietly change every number here.
    """
    legal = list(board.legal_moves)
    if not legal:
        return []
    logits = _logits(model, [board], device)[0]
    idx = torch.tensor([MOVE_TO_ACTION[m.uci()] for m in legal], device=logits.device)
    p = logits[idx].softmax(dim=-1).cpu().numpy()
    return sorted(zip(legal, p.tolist()), key=lambda kv: -kv[1])


def move_values(model, hl, board, device) -> list[tuple[chess.Move, float]]:
    """OUR win probability after each legal move, best first.

    The sign convention `value_policy.py` warns about: the model answers for
    the side to move in the CHILD, who is the opponent, so ours is 1 - that.
    Terminal children are scored exactly rather than asked of a net that was
    never trained to notice a game had ended.
    """
    legal = list(board.legal_moves)
    if not legal:
        return []
    scores: dict[chess.Move, float] = {}
    ask: list[chess.Move] = []
    children: list[chess.Board] = []
    for mv in legal:
        board.push(mv)
        t = terminal_value(board)
        if t is not None:
            scores[mv] = 1.0 - t
        else:
            ask.append(mv)
            children.append(board.copy(stack=False))
        board.pop()
    if children:
        for mv, v in zip(ask, win_probability(model, hl, children, device), strict=True):
            scores[mv] = 1.0 - float(v)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def occlude(model, hl, board, device, kind: str) -> tuple[dict[int, float], dict]:
    """Remove each occupied square in turn; score how much the answer moved.

    One batched forward pass over ~32 boards, so this costs about what one
    position costs. The removals are made on copies; `board` is untouched.
    """
    occupied = [sq for sq in chess.SQUARES if board.piece_at(sq) is not None]
    variants = []
    for sq in occupied:
        b = board.copy(stack=False)
        b.remove_piece_at(sq)
        variants.append(b)

    if kind == "value":
        base = float(win_probability(model, hl, [board], device)[0])
        after = win_probability(model, hl, variants, device)
        scores = {sq: float(v) - base for sq, v in zip(occupied, after, strict=True)}
        summary = {"metric": "delta win probability for the side to move",
                   "base": base, "signed": True, "unit": ""}
    else:
        legal = list(board.legal_moves)
        idx = torch.tensor([MOVE_TO_ACTION[m.uci()] for m in legal])
        base_logits = _logits(model, [board], device)[0]
        p = base_logits[idx.to(base_logits.device)].softmax(dim=-1)
        all_logits = _logits(model, variants, device)
        q = all_logits[:, idx.to(all_logits.device)].softmax(dim=-1)
        # KL(base || occluded) in bits. Legal set is unchanged by construction
        # here only because the mask is taken from the ORIGINAL board; a removal
        # that changes what is legal is exactly the perturbation being measured.
        kl = (p[None] * ((p[None] + 1e-12).log2() - (q + 1e-12).log2())).sum(dim=-1)
        scores = {sq: float(v) for sq, v in zip(occupied, kl.cpu().numpy(), strict=True)}
        top = max(zip(legal, p.tolist()), key=lambda kv: kv[1])
        summary = {"metric": "KL(original || occluded) over legal moves",
                   "base": top[1], "base_move": top[0].uci(), "signed": False,
                   "unit": " bits"}
    return scores, summary


def render(board, scores, summary, moves, info, fen, kind) -> str:
    sq_scores = {chess.square_name(sq): v for sq, v in scores.items()}
    pieces = {chess.square_name(sq): board.piece_at(sq).symbol()
              for sq in chess.SQUARES if board.piece_at(sq)}
    move_rows = [{"uci": m.uci(), "san": board.san(m), "v": v,
                  "from": chess.square_name(m.from_square),
                  "to": chess.square_name(m.to_square)} for m, v in moves[:12]]
    best = {chess.square_name(m.to_square): v for m, v in reversed(moves)}

    data = viz.js_json({
        "scores": sq_scores, "pieces": pieces, "moves": move_rows,
        "best": best, "signed": summary["signed"], "unit": summary["unit"],
        "kind": kind, "base": summary["base"],
        "turn": "white" if board.turn else "black",
    })

    meta = (
        f"<b>{Path(info['path']).name}</b> &middot; {info['kind']} net &middot; "
        f"{info['params']/1e6:.1f}M params &middot; step {info['step']} &middot; "
        f"{'EMA' if info['ema'] else 'raw'}<br><code>{fen}</code><br>"
        f"{summary['metric']}"
        + (f" &middot; base <b>{summary['base']:.4f}</b>"
           f"{' for ' + summary['base_move'] if 'base_move' in summary else ''}")
    )
    rows = "".join(
        f"<tr><td>{i+1}</td><td><b>{r['san']}</b></td><td>{r['uci']}</td>"
        f"<td>{r['v']:.4f}</td></tr>" for i, r in enumerate(move_rows))

    body = f"""
<h1>saliency</h1>
<div class="meta">{meta}</div>
<div class="cols">
  <div class="panel">
    <h2>board</h2>
    <div class="boardwrap">{viz.board_svg(board)}
      <div class="overlay" id="ov"></div>
      <svg class="arrows" id="arrows" viewBox="0 0 360 360"></svg></div>
    <div class="row" style="margin-top:10px"><span class="lab">view</span>
      <span id="views"></span></div>
    <div class="note" id="note"></div>
  </div>
  <div class="panel">
    <h2>{'move values' if kind == 'value' else 'move probabilities'}</h2>
    <table><tr><th>#</th><th>san</th><th>uci</th>
      <th>{'win prob' if kind == 'value' else 'p'}</th></tr>{rows}</table>
  </div>
</div>
<script>
const D = {data};
let view = "saliency";

const ov = document.getElementById("ov"), cells = {{}};
for (let r = 7; r >= 0; r--) for (let f = 0; f < 8; f++) {{
  const name = "abcdefgh"[f] + (r + 1);
  const c = document.createElement("div");
  c.className = "cell";
  c.innerHTML = '<div class="paint"></div><div class="val"></div>';
  c.title = name;
  ov.appendChild(c); cells[name] = c;
}}
{{
  const el = document.getElementById("views");
  for (const [k, t] of [["saliency", "saliency"], ["moves", "moves"]]) {{
    const b = document.createElement("button");
    b.textContent = t; b.dataset.v = k;
    b.onclick = () => {{ view = k; draw(); }};
    el.appendChild(b);
  }}
}}
function xy(name) {{
  const f = name.charCodeAt(0) - 97, r = +name[1] - 1;
  return [45 * f + 22.5, 45 * (7 - r) + 22.5];
}}
function draw() {{
  const src = view === "saliency" ? D.scores : D.best;
  const vals = Object.values(src);
  // Saliency is anchored at zero, because zero means "deleting this changed
  // nothing" and that is a real reading. Move values are NOT: every legal move
  // from a balanced position scores 0.50-0.60, so normalising those against
  // zero paints the whole board one shade and hides the entire ranking. They
  // get the range instead, with a floor so the worst move stays visible.
  const max = Math.max(...vals.map(Math.abs));
  const lo = Math.min(...vals), hi = Math.max(...vals);
  for (const name in cells) {{
    const c = cells[name], v = src[name];
    const paint = c.querySelector(".paint"), lab = c.querySelector(".val");
    if (v === undefined) {{ paint.style.background = "none"; lab.textContent = ""; }}
    else {{
      const a = view === "saliency"
        ? (max > 0 ? Math.abs(v) / max : 0)
        : (hi > lo ? 0.15 + 0.85 * (v - lo) / (hi - lo) : 0.5);
      // diverging only where the metric has a sign; KL and win-prob-after-move
      // are one-sided and get one colour, so a reader cannot infer a direction
      // the number does not carry.
      const c1 = (D.signed && view === "saliency" && v < 0) ? "131,165,152" : "254,128,25";
      paint.style.background = `rgba(${{c1}},${{(0.9 * a).toFixed(3)}})`;
      lab.textContent = view === "saliency"
        ? (Math.abs(v) >= 0.005 ? v.toFixed(2) : "")
        : v.toFixed(2);
    }}
  }}
  const svg = document.getElementById("arrows");
  svg.innerHTML = view === "moves" ? D.moves.slice(0, 5).map((m, i) => {{
    const [x1, y1] = xy(m.from), [x2, y2] = xy(m.to);
    const w = 5 - i * 0.7, o = (1 - i * 0.16).toFixed(2);
    return `<line x1="${{x1}}" y1="${{y1}}" x2="${{x2}}" y2="${{y2}}"
      stroke="#fe8019" stroke-opacity="${{o}}" stroke-width="${{w}}"
      stroke-linecap="round"/><circle cx="${{x2}}" cy="${{y2}}" r="${{w * 1.4}}"
      fill="#fe8019" fill-opacity="${{o}}"/>`;
  }}).join("") : "";
  for (const b of document.getElementById("views").children)
    b.classList.toggle("on", b.dataset.v === view);
  document.getElementById("note").innerHTML = view === "saliency"
    ? (D.signed
        ? `Each occupied square, shaded by the change in the side to move's win `
          + `probability when that piece is deleted. <b>Orange = removing it helps `
          + `${{D.turn}}</b>, blue = removing it hurts ${{D.turn}}. So ${{D.turn}}'s own `
          + `pieces should read blue. Base ${{D.base.toFixed(4)}}.`
        : `Each occupied square, shaded by how far deleting that piece moves the `
          + `policy's distribution over legal moves, in bits of KL. Unsigned: `
          + `large means the network changed its mind, not which way.`)
    : (D.kind === "value"
        ? `Destination squares shaded by the best win probability reachable there, `
          + `stretched across the ${{lo.toFixed(3)}}-${{hi.toFixed(3)}} range this position `
          + `actually spans, arrows for the top five moves. One ply of `
          + `make-move-and-ask, which is what the searchless engine plays; MCTS `
          + `is what sits on top.`
        : `Destination squares shaded by the policy's highest probability landing `
          + `there, arrows for the top five moves.`);
}}
draw();
</script>
"""
    return viz.html_page("sumofish saliency", "", body)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default=str(ROOT / "runs" / "value.pt"))
    ap.add_argument("--fen", default=chess.STARTING_FEN)
    ap.add_argument("--out", default="/tmp/sumofish-saliency.html")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--print", action="store_true", dest="show",
                    help="print the ranking to stdout and write no HTML")
    args = ap.parse_args()

    board = chess.Board(args.fen)
    model, info = viz.load_model(args.checkpoint, device=args.device)
    kind = info["kind"]
    hl = HLGauss(bins=info["output_size"]).to(args.device) if kind == "value" else None

    scores, summary = occlude(model, hl, board, args.device, kind)
    moves = (move_values(model, hl, board, args.device) if kind == "value"
             else move_distribution(model, board, args.device))

    print(f"{Path(args.checkpoint).name}: {kind} net, {info['params']/1e6:.1f}M params, "
          f"step {info['step']}")
    print(f"{summary['metric']} | base {summary['base']:.4f}"
          + (f" for {summary['base_move']}" if "base_move" in summary else ""))
    order = sorted(scores.items(), key=lambda kv: -abs(kv[1]))
    print("\nmost load-bearing pieces:")
    for sq, v in order[:8]:
        p = board.piece_at(sq)
        print(f"  {chess.square_name(sq)} {p.symbol()}  {v:+.4f}{summary['unit']}")
    print("\ntop moves:")
    for mv, v in moves[:6]:
        print(f"  {board.san(mv):<8} {mv.uci()}  {v:.4f}")

    if args.show:
        return
    out = Path(args.out)
    out.write_text(render(board, scores, summary, moves, info, args.fen, kind))
    print(f"\n{out}  ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
