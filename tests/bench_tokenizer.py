#!/usr/bin/env python
"""Measure the tokenizer. Four paths, and the comparison that matters is the last.

  * `tokenize(fen)`       -- Python, parses the string. What a FEN-based FFI
                             contract would force on the caller.
  * `tokenize_board(b)`   -- Python, reads the bitboards. What the engine does now.
  * `core.tokenize(fen)`  -- Rust, one position per crossing.
  * `core.tokenize_batch` -- Rust, one crossing for the whole batch, returning
                             bytes straight into np.frombuffer.

The last is the one the search wants: the value and policy nets each tokenize
every leaf, so this runs twice per node in the current engine.
"""
from __future__ import annotations
import sys, time, random
from pathlib import Path
import chess, numpy as np
import sumofish_core as core

ROOT = Path(__file__).resolve().parent.parent
for c in (ROOT, ROOT.parent / "chess-gpu"):
    if (c / "chessgpu" / "tokenizer.py").exists():
        sys.path.insert(0, str(c)); break
from chessgpu.tokenizer import tokenize, tokenize_board

N = 20000
rng = random.Random(7)
fens, b = [], chess.Board()
while len(fens) < N:
    if b.is_game_over(claim_draw=False): b = chess.Board(); continue
    fens.append(b.fen()); b.push(rng.choice(list(b.legal_moves)))
boards = [chess.Board(f) for f in fens]

def t(fn):
    best = min(_time(fn) for _ in range(3))
    return best

def _time(fn):
    t0 = time.perf_counter(); fn(); return time.perf_counter() - t0

r = {}
r["python tokenize(fen)"]      = t(lambda: [tokenize(f) for f in fens])
r["python tokenize_board(b)"]  = t(lambda: [tokenize_board(x) for x in boards])
r["rust tokenize(fen)"]        = t(lambda: [core.tokenize(f) for f in fens])
r["rust tokenize_batch, 64s"]  = t(lambda: [core.tokenize_batch(fens[i:i+64]) for i in range(0, N, 64)])
r["rust batch + frombuffer"]   = t(lambda: [np.frombuffer(core.tokenize_batch(fens[i:i+64]), np.uint8).reshape(-1,77) for i in range(0, N, 64)])
# what the engine does today, per batch: stack of per-board tokenize
r["python stack per batch"]    = t(lambda: [np.stack([tokenize_board(x) for x in boards[i:i+64]]) for i in range(0, N, 64)])

print(f"{N} positions\n")
base = r["python tokenize_board(b)"]
for k, v in r.items():
    print(f"  {k:28s} {v*1e6/N:8.2f} us/pos   {base/v:6.1f}x vs python tokenize_board")
print()
fast = r["rust batch + frombuffer"]
print(f"  what the engine does now : {r['python stack per batch']*1e6/N:8.2f} us/pos")
print(f"  what the port can do     : {fast*1e6/N:8.2f} us/pos   "
      f"{r['python stack per batch']/fast:.0f}x")
print()
print("  Both nets tokenize every leaf, so this cost is paid TWICE per node in")
print("  the current engine. A FEN-based contract without this would have cost")
print(f"  {r['python tokenize(fen)']*1e6/N:.1f} us/pos instead -- slower than the engine it replaces.")
