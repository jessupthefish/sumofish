#!/usr/bin/env python
"""Shared plumbing for the visualisation scripts. Not an entry point.

`attention.py` and `saliency.py` both need the same four things: a checkpoint
loaded without assuming which preset it is, the map from a chess square to the
token index that carries it, a board SVG, and a self-contained HTML wrapper.
They live here so the two tools cannot drift on any of them.

Three facts about this model that everything downstream depends on, all of
them read out of `sumofish/model.py` and `sumofish/tokenizer.py` rather than
assumed:

1. **The model input is 78 tokens, not 77.** `ChessTransformer.forward`
   prepends a BOS zero and drops nothing, so every index in the tokenizer's
   77-token layout is shifted by one inside the network. Off-by-one here
   silently mislabels the whole board by one square, which looks like a
   plausible attention map rather than like a bug.
2. **Position 77 is the readout.** Only `h[:, -1]` reaches the head. It is the
   last digit of the fullmove counter, which is a strange thing for the whole
   evaluation to hang off, and it is upstream's design.
3. **The board is not in board order.** FEN writes rank 8 first, python-chess
   numbers a1 as 0, so token index and square index run backwards by rank.
   `tokenizer._SQUARE_TO_TOKEN_INDEX` is the verified mapping and is imported
   rather than re-derived; `tests/verify_search.py` is what keeps it honest.
"""

from __future__ import annotations

import base64
import dataclasses
import html
import json
import sys
from pathlib import Path

import chess
import chess.svg
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sumofish.model import ChessTransformer, ModelConfig  # noqa: E402
from sumofish.tokenizer import (  # noqa: E402
    SEQUENCE_LENGTH,
    _SQUARE_TO_TOKEN_INDEX,
    tokenize_board,
)

# The BOS shift. See note 1 above.
BOS = 1
MODEL_SEQUENCE_LENGTH = SEQUENCE_LENGTH + BOS      # 78
READOUT = MODEL_SEQUENCE_LENGTH - 1                # 77, the only position read

# Square -> index into the model's 78-token input.
SQUARE_TO_MODEL_INDEX = tuple(BOS + i for i in _SQUARE_TO_TOKEN_INDEX)
MODEL_INDEX_TO_SQUARE = {t: sq for sq, t in enumerate(SQUARE_TO_MODEL_INDEX)}


def load_model(path: str | Path, device: str = "cpu") -> tuple[ChessTransformer, dict]:
    """Load a checkpoint the way the engine does, and say what it loaded.

    Two things are copied deliberately from `search_engine.py`: the config comes
    out of the checkpoint rather than from a preset name, and unknown keys are
    filtered before `ModelConfig(**cfg)`, so a checkpoint written against an
    older dataclass signature still loads instead of raising TypeError.

    EMA weights when present, because those are what gets promoted and played.
    Scoring the raw weights would describe a network that never ships.

    Note this runs in float32. The deployed engine evaluates under bfloat16
    autocast, so these numbers are the model's, not byte-for-byte the bot's.
    At 78 tokens the accuracy is worth more than the speed.
    """
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in ck["cfg"].items() if k in fields})
    model = ChessTransformer(cfg)
    state = ck.get("ema") or ck["model"]
    model.load_state_dict({k: v.float() for k, v in state.items()})
    model.to(device).eval()
    info = {
        "path": str(path),
        "step": ck.get("step"),
        "params": model.num_parameters(),
        "ema": bool(ck.get("ema")),
        "output_size": cfg.output_size,
        "layers": cfg.num_layers,
        "heads": cfg.num_heads,
        "embedding_dim": cfg.embedding_dim,
        "causal": cfg.causal,
        "device": device,
        # 64 output bins is the HL-Gauss value head; 1968 is the move space.
        "kind": "value" if cfg.output_size == 64 else "policy",
    }
    return model, info


def tokens_for(board: chess.Board, device: str = "cpu") -> torch.Tensor:
    """The [1, 77] int64 input for one position."""
    return torch.from_numpy(tokenize_board(board).astype(np.int64))[None].to(device)


def token_labels(board: chess.Board) -> list[dict]:
    """One label per model position, 78 of them.

    `kind` is what a caller switches on: `square` positions get drawn on the
    board, everything else gets drawn as a chip, and nothing is dropped. The
    castling, en-passant and clock fields are 12 of the 78 positions and they
    are exactly the part a board-shaped picture would hide.
    """
    from sumofish.tokenizer import CHARACTERS

    toks = tokenize_board(board)
    labels: list[dict] = [{"kind": "bos", "text": "BOS", "square": None}]
    for i, t in enumerate(toks):
        ch = CHARACTERS[int(t)]
        model_i = i + BOS
        if model_i == BOS:
            kind, text = "side", f"turn:{ch}"
        elif model_i in MODEL_INDEX_TO_SQUARE:
            sq = MODEL_INDEX_TO_SQUARE[model_i]
            kind = "square"
            text = f"{chess.square_name(sq)}:{ch}"
        elif 66 <= model_i <= 69:
            kind, text = "castling", f"castle:{ch}"
        elif 70 <= model_i <= 71:
            kind, text = "ep", f"ep:{ch}"
        elif 72 <= model_i <= 74:
            kind, text = "halfmove", f"half:{ch}"
        else:
            kind, text = "fullmove", f"full:{ch}"
        labels.append({"kind": kind, "text": text, "square":
                       MODEL_INDEX_TO_SQUARE.get(model_i)})
    assert len(labels) == MODEL_SEQUENCE_LENGTH, len(labels)
    return labels


# python-chess ships a tan/brown board, which is the one colour a heat overlay
# must not be: an unshaded light square and a lightly-shaded one were visually
# identical in the first version of this, so the board read as uniformly hot.
# Neutral greys keep every pixel of colour on the page meaningful.
BOARD_COLORS = {
    "square light": "#4a4642",
    "square dark": "#35322f",
    "margin": "#282828",
    "inner border": "#1d2021",
    "outer border": "#1d2021",
    "coord": "#a89984",
}


def board_svg(board: chess.Board, orientation: chess.Color = chess.WHITE) -> str:
    """A 360-unit board with no coordinates and no margin.

    `coordinates=False` is what makes the overlay arithmetic exact: the viewBox
    is 0 0 360 360, so square (file, rank) starts at 12.5% * file and the CSS
    grid lines up with the SVG without a fudge factor. With coordinates on it
    is a 390-unit box with a 15-unit margin and every overlay cell is off by
    that margin -- which LAB-NOTES already records as the rasteriser trap, in
    the other direction.
    """
    return chess.svg.board(board, coordinates=False, orientation=orientation,
                           colors=BOARD_COLORS, size=None)


def quantise_rows(a: np.ndarray) -> tuple[str, list]:
    """Pack a float array to base64 uint8, scaled by the max of its last axis.

    Per-ROW rather than per-matrix on purpose. Causal attention gives row 0 a
    single 1.0 entry, so a per-matrix max is ~1.0 always, and a row whose
    largest weight is 0.05 would quantise into the bottom 13 of 255 levels and
    display as flat black. Per-row costs one float per row to store and keeps
    every row at full contrast.
    """
    scales = a.max(axis=-1, keepdims=True)
    safe = np.where(scales > 0, scales, 1.0)
    q = np.rint(255.0 * a / safe).astype(np.uint8)
    return base64.b64encode(q.tobytes()).decode("ascii"), scales.squeeze(-1).tolist()


def html_page(title: str, head: str, body: str) -> str:
    """A standalone page. No CDN, no fonts, no network: it opens from a file://.

    Deliberate, and not just hygiene. The alternative here was circuitsvis or a
    notebook, which means a Jupyter kernel and three dependencies to look at an
    8x8 grid of heatmaps. This is one file you can scp anywhere.
    """
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
:root {{
  --bg: #1d2021; --fg: #ebdbb2; --dim: #a89984; --line: #3c3836;
  --panel: #282828; --accent: #fe8019; --accent2: #83a598;
}}
* {{ box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--fg); margin: 0; padding: 20px;
  font: 13px/1.5 ui-monospace, "JetBrains Mono", Menlo, Consolas, monospace; }}
h1 {{ font-size: 15px; font-weight: 600; margin: 0 0 4px; }}
a {{ color: var(--accent2); }}
.meta {{ color: var(--dim); margin-bottom: 16px; }}
.meta b {{ color: var(--fg); font-weight: 600; }}
.cols {{ display: flex; gap: 24px; flex-wrap: wrap; align-items: flex-start; }}
.panel {{ background: var(--panel); border: 1px solid var(--line);
  border-radius: 4px; padding: 12px; }}
.panel h2 {{ font-size: 12px; font-weight: 600; margin: 0 0 8px;
  color: var(--dim); text-transform: uppercase; letter-spacing: .06em; }}
.boardwrap {{ position: relative; width: 440px; height: 440px; }}
.boardwrap svg {{ width: 100%; height: 100%; display: block; }}
.arrows {{ position: absolute; inset: 0; pointer-events: none; z-index: 3; }}
.overlay {{ position: absolute; inset: 0; z-index: 2; display: grid;
  grid-template-columns: repeat(8, 1fr); grid-template-rows: repeat(8, 1fr); }}
.cell {{ position: relative; cursor: pointer; }}
.cell .paint {{ position: absolute; inset: 0; }}
.cell.sel {{ outline: 2px solid var(--accent); outline-offset: -2px; }}
.cell .val {{ position: absolute; left: 3px; top: 2px; font-size: 10px;
  color: #fff; text-shadow: 0 0 2px #000, 0 0 4px #000, 0 0 4px #000;
  pointer-events: none; letter-spacing: -.02em; }}
button {{ background: #32302f; color: var(--fg); border: 1px solid var(--line);
  border-radius: 3px; padding: 3px 8px; font: inherit; font-size: 12px;
  cursor: pointer; }}
button:hover {{ border-color: var(--dim); }}
button.on {{ background: var(--accent); color: #1d2021; border-color: var(--accent); }}
.row {{ display: flex; gap: 4px; flex-wrap: wrap; align-items: center;
  margin-bottom: 8px; }}
.row .lab {{ color: var(--dim); width: 74px; flex: none; }}
.chips {{ display: flex; gap: 3px; flex-wrap: wrap; max-width: 440px; }}
.chip {{ border: 1px solid var(--line); border-radius: 3px; padding: 1px 5px;
  font-size: 11px; cursor: pointer; position: relative; }}
.chip .paint {{ position: absolute; inset: 0; border-radius: 2px; }}
.chip span {{ position: relative; }}
.chip.sel {{ outline: 1px solid var(--accent); }}
table {{ border-collapse: collapse; font-size: 12px; }}
td, th {{ padding: 2px 10px 2px 0; text-align: left; vertical-align: top; }}
th {{ color: var(--dim); font-weight: 600; }}
canvas {{ image-rendering: pixelated; border: 1px solid var(--line);
  cursor: crosshair; }}
.note {{ color: var(--dim); max-width: 460px; margin-top: 10px; font-size: 12px; }}
{head}
</style></head>
<body>
{body}
</body></html>
"""


def js_json(obj) -> str:
    """JSON safe to drop inside a <script> tag."""
    return json.dumps(obj).replace("</", "<\\/")
