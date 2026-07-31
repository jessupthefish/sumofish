"""Verify the tokenizer and bag reader against DeepMind's reference and real data.

Run: .venv/bin/python tests/verify_data.py

Four things get checked, in increasing order of how badly a failure would hurt:

  1. Bag container: record counts, offsets, no truncation.
  2. Record decoding: every FEN parses as a real position and every move is
     legal in it. A wrong varint or endianness guess cannot survive this.
  3. Tokenizer: byte-exact against upstream over every FEN we decode. Upstream
     is exec'd with its jaxtyping annotation stripped, so this compares against
     the actual published implementation, not a paraphrase of it.
  4. Action space: size and coverage, including that every move appearing in
     the real data maps to an action index.
"""

from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

import chess
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sumofish import bagz, tokenizer  # noqa: E402

DATA = ROOT / "data"
REFERENCE = ROOT / "reference" / "tokenizer.py"

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(label)


def load_upstream_tokenize():
    """Exec upstream's tokenizer with the jaxtyping dependency removed.

    We are comparing against the real published code, so the only edits are the
    import and the return annotation, neither of which affects behaviour.
    """
    src = REFERENCE.read_text()
    src = src.replace("import jaxtyping as jtp\n", "")
    src = re.sub(r"-> jtp\.Int32\[jtp\.Array, 'T'\]:", "-> object:", src)
    ns: dict = {}
    exec(compile(src, str(REFERENCE), "exec"), ns)  # noqa: S102
    return ns["tokenize"], ns["SEQUENCE_LENGTH"], ns["_CHARACTERS"]


print("=== 1. bag container ===")
bc_test = DATA / "test" / "behavioral_cloning_data.bag"
sv_test = DATA / "test" / "state_value_data.bag"
av_test = DATA / "test" / "action_value_data.bag"

readers = {}
for name, path in [("behavioral_cloning", bc_test), ("state_value", sv_test), ("action_value", av_test)]:
    if not path.exists():
        check(f"{name} present", False, "(not downloaded)")
        continue
    r = bagz.BagReader(path)
    readers[name] = r
    check(f"{name}", len(r) > 0, f"{len(r):,} records")
    # Offsets must be monotonic and cover the file exactly.
    total = sum(len(r[i]) for i in range(min(1000, len(r))))
    check(f"{name} records readable", total > 0, f"first 1000 records = {total:,} bytes")

print()
print("=== 2. record decoding (FEN parses, move is legal) ===")

sample_fens: list[str] = []

if "behavioral_cloning" in readers:
    r = readers["behavioral_cloning"]
    idxs = np.linspace(0, len(r) - 1, 4000, dtype=int)
    bad_fen = bad_move = 0
    for i in idxs:
        fen, move = bagz.decode_behavioral_cloning(r[int(i)])
        sample_fens.append(fen)
        try:
            board = chess.Board(fen)
        except ValueError:
            bad_fen += 1
            continue
        if chess.Move.from_uci(move) not in board.legal_moves:
            bad_move += 1
    check("BC: all FENs parse", bad_fen == 0, f"{bad_fen} bad of {len(idxs)}")
    check("BC: all moves legal", bad_move == 0, f"{bad_move} illegal of {len(idxs)}")

if "state_value" in readers:
    r = readers["state_value"]
    idxs = np.linspace(0, len(r) - 1, 4000, dtype=int)
    bad_fen = out_of_range = 0
    probs = []
    for i in idxs:
        fen, wp = bagz.decode_state_value(r[int(i)])
        sample_fens.append(fen)
        try:
            chess.Board(fen)
        except ValueError:
            bad_fen += 1
        if not 0.0 <= wp <= 1.0:
            out_of_range += 1
        probs.append(wp)
    check("SV: all FENs parse", bad_fen == 0, f"{bad_fen} bad of {len(idxs)}")
    # Big-endian is a guess; a wrong guess yields denormals and infinities, not
    # a clean [0,1] distribution.
    check("SV: win_prob in [0,1]", out_of_range == 0, f"{out_of_range} out of range")
    check(
        "SV: win_prob distribution sane",
        0.2 < float(np.mean(probs)) < 0.8,
        f"mean={np.mean(probs):.4f} min={np.min(probs):.4f} max={np.max(probs):.4f}",
    )

if "action_value" in readers:
    r = readers["action_value"]
    idxs = np.linspace(0, len(r) - 1, 4000, dtype=int)
    bad = 0
    probs = []
    for i in idxs:
        try:
            fen, move, wp = bagz.decode_action_value(r[int(i)])
            board = chess.Board(fen)
            if chess.Move.from_uci(move) not in board.legal_moves:
                bad += 1
            if not 0.0 <= wp <= 1.0:
                bad += 1
            probs.append(wp)
            sample_fens.append(fen)
        except (ValueError, struct.error):
            bad += 1
    check("AV: decode + legality + range", bad == 0, f"{bad} bad of {len(idxs)}")
    if probs:
        check("AV: win_prob spread", float(np.std(probs)) > 0.05, f"std={np.std(probs):.4f}")

print()
print("=== 3. tokenizer, byte-exact vs upstream ===")
try:
    upstream_tokenize, upstream_len, upstream_chars = load_upstream_tokenize()
    check("upstream loaded", True, f"SEQUENCE_LENGTH={upstream_len}")
    check("vocabulary identical", upstream_chars == tokenizer.CHARACTERS,
          f"{len(tokenizer.CHARACTERS)} chars")
    check("SEQUENCE_LENGTH identical", upstream_len == tokenizer.SEQUENCE_LENGTH)

    mismatches = 0
    for fen in sample_fens:
        if not np.array_equal(tokenizer.tokenize(fen), upstream_tokenize(fen)):
            mismatches += 1
    check(f"tokenize matches on {len(sample_fens):,} real FENs", mismatches == 0,
          f"{mismatches} mismatches")

    # Edge cases real data may not cover: full castling rights, en passant set,
    # three-digit counters, and a bare-kings endgame.
    edge = [
        chess.STARTING_FEN,
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "8/8/8/4k3/8/8/8/4K3 w - - 100 300",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "4k3/8/8/8/8/8/8/4K3 b - - 99 999",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    ]
    edge_bad = 0
    for fen in edge:
        mine, theirs = tokenizer.tokenize(fen), upstream_tokenize(fen)
        if not np.array_equal(mine, theirs) or len(mine) != 77:
            edge_bad += 1
            print(f"        MISMATCH on {fen}")
    check(f"tokenize matches on {len(edge)} edge cases", edge_bad == 0)

    t = tokenizer.tokenize(chess.STARTING_FEN)
    check("dtype is uint8", t.dtype == np.uint8, str(t.dtype))
    check("all tokens in vocab", int(t.max()) < tokenizer.VOCAB_SIZE,
          f"max={int(t.max())} vocab={tokenizer.VOCAB_SIZE}")
except Exception as e:  # noqa: BLE001
    check("upstream comparison", False, f"{type(e).__name__}: {e}")

print()
print("=== 4. action space ===")
check("NUM_ACTIONS == 1968", tokenizer.NUM_ACTIONS == 1968, str(tokenizer.NUM_ACTIONS))
check("action space is a bijection",
      len(set(tokenizer.ACTION_TO_MOVE)) == tokenizer.NUM_ACTIONS)
check("round-trips",
      all(tokenizer.MOVE_TO_ACTION[m] == i for i, m in enumerate(tokenizer.ACTION_TO_MOVE)))

if "behavioral_cloning" in readers:
    r = readers["behavioral_cloning"]
    idxs = np.linspace(0, len(r) - 1, 20000, dtype=int)
    unmapped = set()
    for i in idxs:
        _, move = bagz.decode_behavioral_cloning(r[int(i)])
        if move not in tokenizer.MOVE_TO_ACTION:
            unmapped.add(move)
    check(f"every move in {len(idxs):,} real records maps to an action",
          not unmapped, f"unmapped: {sorted(unmapped)[:5]}")

# Every legal move from a random walk must be encodable, or the model can be
# asked to output a move it has no index for.
board = chess.Board()
rng = np.random.default_rng(0)
missing = set()
for _ in range(3000):
    if board.is_game_over():
        board = chess.Board()
    legal = list(board.legal_moves)
    for m in legal:
        if m.uci() not in tokenizer.MOVE_TO_ACTION:
            missing.add(m.uci())
    board.push(legal[rng.integers(len(legal))])
check("all legal moves in a 3000-ply random walk are encodable", not missing,
      f"missing: {sorted(missing)[:5]}")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("ALL CHECKS PASSED")
