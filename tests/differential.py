#!/usr/bin/env python
"""Oracle 2: differential testing of the Rust core against `python-chess`.

Perft (see `rust/tests/perft.rs`) checks move *sets* against published counts
from outside this project. This checks move *order*, which perft cannot see
because it only counts, and which the search depends on:

    `mcts._expand` inserts children in `board.legal_moves` order, and
    `_select_child` iterates that dict taking a strict `>`, so the FIRST move at
    a given PUCT score wins a tie. Reorder the generator and the engine plays a
    different move while every move it plays stays legal.

The two oracles are deliberately independent. Perft would catch a shared
misconception that this file cannot, and this file catches an ordering
divergence that perft cannot.

Usage:

    tests/differential.py --positions 100000          # from the training bags
    tests/differential.py --fens some_file.txt
    tests/differential.py --quick                     # the perft suite only

Exit status is 0 only if every checked position agrees exactly.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent

# The same six positions as the Rust perft suite, so a failure can be compared
# side by side across both oracles.
SUITE = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    # En passant, both colours. The pseudo-legal generator does not emit ep at
    # all -- `generate_legal_moves` adds it separately -- so where it lands in
    # the sequence is asserted here rather than assumed in the Rust docstring.
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "rnbqkbnr/pppp1ppp/8/8/3pP3/8/PPP2PPP/RNBQKBNR b KQkq e3 0 3",
    # The pinned-pawn en passant case: capturing would expose the king along
    # the rank. python-chess excludes it; a naive generator does not.
    "8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1",
    # Promotion under check, and promotion with capture.
    "8/P6k/8/8/8/8/6K1/8 w - - 0 1",
    "n6k/1P6/8/8/8/8/6K1/8 w - - 0 1",
]


def load_rust():
    try:
        import sumofish_core
    except ImportError as exc:
        print(
            "sumofish_core is not importable. Build it first:\n"
            "    cd rust && maturin develop --release\n"
            f"({exc})",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return sumofish_core


def fens_from_bag(limit: int) -> list[str]:
    """Real positions from the training data, which is what the engine sees.

    Falls back to random self-play if the bags are not present in this
    worktree, since `data/` is gitignored and a worktree starts without it.
    """
    # The bags live in the main checkout; a worktree starts without data/,
    # since it is gitignored. Try the sibling checkout before giving up.
    for root in (ROOT, ROOT.parent / "chess-gpu"):
        bag = root / "data/train/state_value_data.bag"
        if not bag.exists():
            continue
        sys.path.insert(0, str(root))
        try:
            # NOTE: the class is BagReader, not BagFileReader. Getting this
            # wrong made the harness fall back to random play SILENTLY, which
            # is worse than failing: random games are far tamer than real
            # positions, so coverage collapses without the output changing.
            from chessgpu.bagz import BagReader, decode_state_value
        except Exception as exc:
            print(f"  (bag present but chessgpu not importable: {exc})")
            break

        reader = BagReader(str(bag))
        n = len(reader)
        # Stride across the whole bag rather than taking a prefix: consecutive
        # records are one game, so a prefix samples a handful of openings.
        step = max(1, n // limit)
        out = []
        for i in range(0, min(n, step * limit), step):
            try:
                out.append(decode_state_value(reader[i])[0])
            except Exception:
                continue
            if len(out) >= limit:
                break
        if out:
            print(f"  sampled {len(out)} positions from {bag}")
            return out

    print("  no bag found; falling back to random legal play")
    return fens_from_random_play(limit)


def fens_from_random_play(limit: int, seed: int = 1234) -> list[str]:
    """Random legal play, which reaches messier positions than opening theory.

    Deliberately includes the positions right before a game ends, because that
    is where terminal detection and legality interact.
    """
    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < limit:
        board = chess.Board()
        for _ in range(rng.randint(0, 120)):
            if board.is_game_over(claim_draw=False):
                break
            out.append(board.fen())
            if len(out) >= limit:
                break
            moves = list(board.legal_moves)
            board.push(rng.choice(moves))
    return out[:limit]


def compare(core, fen: str) -> str | None:
    """Return a description of the first disagreement, or None if identical."""
    ref = chess.Board(fen)

    # 1. FEN round-trip. Checks the board representation independently of
    #    move generation, which is the part still stubbed.
    got_fen = core.fen_roundtrip(fen)
    if got_fen != ref.fen():
        return f"FEN round-trip\n  rust:   {got_fen}\n  python: {ref.fen()}"

    # 2. Legal moves, ORDER SENSITIVE. Not a set comparison.
    expected = [m.uci() for m in ref.legal_moves]
    got = core.Board(fen).legal_moves()
    if got != expected:
        if sorted(got) == sorted(expected):
            return (
                "move ORDER differs (same set, so perft would pass)\n"
                f"  rust:   {' '.join(got)}\n"
                f"  python: {' '.join(expected)}"
            )
        missing = sorted(set(expected) - set(got))
        extra = sorted(set(got) - set(expected))
        return (
            "move SET differs\n"
            f"  missing from rust: {' '.join(missing) or '(none)'}\n"
            f"  extra in rust:     {' '.join(extra) or '(none)'}"
        )
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", type=int, default=20000)
    ap.add_argument("--fens", type=Path, help="file of FENs, one per line")
    ap.add_argument("--quick", action="store_true", help="the curated suite only")
    ap.add_argument("--max-report", type=int, default=5)
    args = ap.parse_args()

    core = load_rust()
    print(f"sumofish_core {core.__version__}")

    fens = list(SUITE)
    if args.fens:
        fens += [ln.strip() for ln in args.fens.read_text().splitlines() if ln.strip()]
    elif not args.quick:
        fens += fens_from_bag(args.positions)

    print(f"checking {len(fens)} positions, order-sensitive")

    failures = 0
    for i, fen in enumerate(fens):
        try:
            problem = compare(core, fen)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            # PyO3 turns a Rust panic into `pyo3_runtime.PanicException`, which
            # inherits from BaseException, NOT Exception. Catching `Exception`
            # here silently lets every panic escape and kills the run on the
            # first bad position instead of reporting all of them.
            problem = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        if problem:
            failures += 1
            if failures <= args.max_report:
                print(f"\nFAIL [{i}] {fen}\n  {problem}")
        if i and i % 5000 == 0:
            print(f"  {i} checked, {failures} failing")

    print()
    if failures:
        print(f"FAIL: {failures} of {len(fens)} positions disagree")
        return 1
    print(f"OK: {len(fens)} positions agree exactly, including order")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
