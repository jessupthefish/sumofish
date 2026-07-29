#!/usr/bin/env python
"""Oracle 5: the Stage C acceptance test. A byte-identical visit vector.

Runs the REAL `chessgpu.mcts.MCTS` and the Rust `sumofish_core.Mcts` on the same
position with the same budget, and requires the root visit counts to match
exactly, move for move, in order.

# Why a mock evaluator rather than the real networks

The identity test has a chicken-and-egg problem with the GPU. Floating-point
reduction order in a matmul can depend on batch shape, so two runs that build
*different* batches can legitimately produce slightly different values -- but the
batches only differ if the search already diverged. Testing with the real net
would confound "the search is wrong" with "the network is not bit-reproducible
across batch shapes", and the first is what we care about.

So both sides get the same deterministic pseudo-evaluator, seeded from the FEN.
Then any difference in the visit vector is a search bug, full stop.

# What this actually pins down

Everything that decides a move: the PUCT expression and its association order,
the c_puct log schedule, the FPU clamp, tie-breaking by child order (strict `>`,
first child wins), virtual loss magnitude and sign, backup's per-level flip,
terminal handling, the batch boundary, and the root-expansion special case.

It also pins down the defects. The Python search duplicates leaves in a batch and
lets virtual loss corrupt Q rather than only N. Both are reproduced, so a "fix"
applied during the port would FAIL this test -- which is the intended behaviour,
because a fix must be a separate, separately-measured change.

Usage:
    tests/identity_search.py --positions 40 --sims 400 --batch 64
"""

from __future__ import annotations

import argparse
import random
import sys
import zlib
from pathlib import Path

import chess
import numpy as np

import sumofish_core as core

ROOT = Path(__file__).resolve().parent.parent
for candidate in (ROOT, ROOT.parent / "chess-gpu"):
    if (candidate / "chessgpu" / "mcts.py").exists():
        sys.path.insert(0, str(candidate))
        break
from chessgpu.mcts import MCTS, _softmax_over_legal  # noqa: E402

ACTION_SPACE = 1968


# --- the deterministic pseudo-evaluator, shared by both sides ---------------


def logit_row(fen: str) -> np.ndarray:
    """A fixed (1968,) logit vector for a position. Same bytes every call.

    **float32, deliberately.** `policy._logprobs` ends in `.float()`, so the real
    engine's logits are float32, and `_softmax_over_legal` therefore builds a
    float32 `scores` array and does the whole softmax -- exp, sum, divide -- in
    single precision. An earlier version of this file used float64 (numpy's
    `standard_normal` default), which meant the identity test was verifying the
    port in a precision the engine never runs. Everything still passed, which is
    exactly what made it worth catching: the test was right about a regime that
    does not ship.
    """
    seed = zlib.crc32(fen.encode())
    return np.random.default_rng(seed).standard_normal(ACTION_SPACE).astype(np.float32)


def mock_value(fen: str) -> float:
    """A fixed value in [0, 1] for a position, from the side-to-move view."""
    return (zlib.crc32(b"value|" + fen.encode()) % 1_000_003) / 1_000_003.0


class MockLogits:
    """Quacks like the torch tensor `policy._logprobs` returns."""

    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def __getitem__(self, i):
        return MockLogits(self._arr[i])


class MockPolicy:
    def _logprobs(self, boards):
        return MockLogits(np.stack([logit_row(b.fen()) for b in boards]))


class MockValuePolicy:
    def value_of(self, board) -> float:
        return mock_value(board.fen())

    def evaluate(self, boards):
        return [mock_value(b.fen()) for b in boards], None


def rust_evaluate(fens: list[str], actions: list[list[int]]):
    """The Rust side's callback.

    Rust supplies the action index for each legal move, so this does NOT touch
    `python-chess`: no board reconstruction, no move generation. That was the
    whole point of the contract change -- with FENs alone, this function cost
    78 ms of a 200 ms callback rebuilding boards Rust had already generated.

    The softmax is still numpy, and deliberately so: it must reproduce
    `_softmax_over_legal` bit for bit, and numpy's pairwise `sum` does not agree
    with a naive sequential f64 sum on 41% of positions (by 1 ULP). The gather
    and the arithmetic below are copied from that function rather than
    paraphrased.
    """
    priors, values = [], []
    for fen, acts in zip(fens, actions):
        # Hoisted deliberately. Leaving `logit_row(fen)` inside the
        # comprehension calls it once per LEGAL MOVE rather than once per
        # position -- ~30 draws of 1968 normals each -- and made the callback so
        # expensive that Rust measured 4x SLOWER than Python. The engine was
        # fine; the harness was not. A benchmark can only ever be as honest as
        # its cheapest path.
        row = logit_row(fen)
        scores = np.array([row[a] if a >= 0 else -1e9 for a in acts])
        exp = np.exp(scores - scores.max())
        priors.append((exp / exp.sum()).tolist())
        values.append(mock_value(fen))
    return priors, values


# --- corpus -----------------------------------------------------------------


def corpus(n: int, seed: int) -> list[list[str]]:
    """Games as move lists, so both sides can be built with real history.

    History matters: repetition depends on the moves before the root, so a
    position handed over as a bare FEN would make the two implementations
    disagree about threefold draws for reasons that have nothing to do with the
    tree.
    """
    rng = random.Random(seed)
    out = []
    # The startpos, so the simplest case is covered first and fails loudest.
    out.append([])
    for _ in range(n - 1):
        board = chess.Board()
        moves: list[str] = []
        for _ in range(rng.randint(1, 60)):
            legal = list(board.legal_moves)
            if not legal:
                break
            mv = rng.choice(legal)
            board.push(mv)
            moves.append(mv.uci())
        if board.is_game_over(claim_draw=False):
            continue
        out.append(moves)
    return out


def run_one(moves: list[str], sims: int, batch: int) -> str | None:
    # --- Python ---
    py_board = chess.Board()
    for u in moves:
        py_board.push(chess.Move.from_uci(u))

    py_mcts = MCTS(
        MockValuePolicy(),
        policy=MockPolicy(),
        c_puct=2.0,
        c_puct_base=19652.0,
        c_puct_init=1.25,
        simulations=sims,
        batch=batch,
        fpu=-0.2,
        reuse=False,
    )
    _root, py_visits = py_mcts.search(py_board)
    py_pairs = [(m.uci(), v) for m, v in py_visits.items()]

    # --- Rust ---
    rs_pos = core.Position()
    for u in moves:
        rs_pos.push_uci(u)
    rs_mcts = core.Mcts(
        c_puct=2.0, c_puct_base=19652.0, c_puct_init=1.25, fpu=-0.2, batch=batch
    )
    rs_pairs = rs_mcts.search(rs_pos, sims, rust_evaluate)

    if rs_pairs == py_pairs:
        return None

    # Report the first divergence in a form that points at a cause.
    lines = [f"visit vectors differ ({len(moves)} plies of history, {sims} sims, batch {batch})"]
    if [m for m, _ in rs_pairs] != [m for m, _ in py_pairs]:
        lines.append("  MOVE ORDER differs -- a Stage A regression, not a tree bug")
        lines.append(f"    rust:   {' '.join(m for m, _ in rs_pairs)}")
        lines.append(f"    python: {' '.join(m for m, _ in py_pairs)}")
    else:
        diffs = [
            (m, r, p) for (m, r), (_, p) in zip(rs_pairs, py_pairs) if r != p
        ]
        lines.append(f"  same moves, {len(diffs)} of {len(rs_pairs)} counts differ")
        for m, r, p in diffs[:8]:
            lines.append(f"    {m}: rust={r} python={p}")
        lines.append(f"  totals: rust={sum(v for _, v in rs_pairs)} python={sum(v for _, v in py_pairs)}")
    lines.append(f"  fen: {py_board.fen()}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", type=int, default=40)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-report", type=int, default=3)
    args = ap.parse_args()

    games = corpus(args.positions, args.seed)
    print(
        f"{len(games)} positions, {args.sims} simulations, batch {args.batch}, "
        f"deterministic mock evaluator"
    )

    failures = 0
    for i, moves in enumerate(games):
        problem = run_one(moves, args.sims, args.batch)
        if problem:
            failures += 1
            if failures <= args.max_report:
                print(f"\nFAIL [{i}] {problem}")
        elif i and i % 10 == 0:
            print(f"  {i} identical")

    print()
    if failures:
        print(f"FAIL: {failures} of {len(games)} visit vectors differ")
        return 1
    print(f"OK: {len(games)} visit vectors byte-identical to chessgpu.mcts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
