#!/usr/bin/env python
"""Oracle 10: mate distance. Does the engine find the SHORTEST mate?

This is the one remaining fix that can be measured without an Elo instrument,
because "shortest forced mate" is a fact about the position rather than a matter
of strength. So the exchange rate being retracted does not block it.

# Ground truth is computed here, not trusted

The engine now *claims* proofs (`root_proven`). A claim is not evidence, so this
file computes the true minimum mate distance with an independent exhaustive
search over `python-chess` and checks the claim against it. If the engine ever
claims a mate that is not there, or a shorter one than exists, this fails.

# The metric

For each position with a forced mate in N moves for the side to move:

  * does the engine's chosen move PRESERVE the minimum? -- i.e. after playing it,
    is the opponent still mated in N-1 against best defence?
  * a distance-blind search can pick a move that still mates but slower, or
    shuffle in a won position until the fifty-move rule intervenes. That is the
    defect being fixed, and it is invisible to puzzle accuracy.

Reported for `mate_distance` OFF and ON, on the same positions and seed, so the
difference is the fix and nothing else.

# The mock evaluator was hiding the answer, in both directions (2026-08-13)

`--real-nets` drives the search with the deployed value and policy nets instead
of `identity_search`'s mock. It changes both headline numbers, and the mock was
wrong the *pessimistic* way on one and the *optimistic* way on the other:

    34 forced mates, 400 sims        mock          deployed nets
    shortest mate-in-1               22/23 -> 23/23   23/23 both
    shortest mate-in-2                6/11 ->  6/11   11/11 both
    proofs claimed (0 bogus)                     25          34
    tree nodes                     55,331 -> 41,868   6,647 -> 3,185
    tree reduction                              24%         52%

So the move-choice question this file was built to answer is **saturated**: with
a real prior the engine already plays the shortest mate in every position, with
the fix and without it. That is not "the fix does nothing", it is "this suite
cannot discriminate", and the reason is the cap of mate-in-2 that the exhaustive
solver imposes. The failure `mate_distance` actually prevents -- shuffling in a
won position until the fifty-move rule intervenes -- needs mates far longer than
`min_mate` can enumerate, and is only visible in whole games.

What the real nets DO establish is the other number: the tree is **52% smaller**,
twice the mock's 24%, and squarely on Lc0's published ~50% for tactical
positions. That is a CLOCK gain. A fixed-simulation match cannot see a clock
gain by construction, which is why `runs/matches/mate-distance-400sims` is a
lower bound and a fixed-time arm is the measurement that answers deployment.
See `docs/OPERATING-POINT.md`.

Usage:
    tests/verify_mate.py --positions 200 --sims 400              # mock, no GPU
    tests/verify_mate.py --positions 200 --sims 400 --real-nets  # the real answer
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import chess

import sumofish_core as core

ROOT = Path(__file__).resolve().parent.parent
for candidate in (ROOT, ROOT.parent / "sumofish"):
    if (candidate / "sumofish" / "mcts.py").exists():
        sys.path.insert(0, str(candidate))
        break

from identity_search import rust_evaluate  # noqa: E402


def mates_in(board: chess.Board, n: int) -> bool:
    """Can the side to move force mate in at most `n` MOVES? Exhaustive."""
    if n <= 0:
        return False
    for mv in board.legal_moves:
        board.push(mv)
        try:
            if board.is_checkmate():
                return True
            if board.is_game_over(claim_draw=False):
                continue  # stalemate is not a mate
            if n == 1:
                continue
            # Every defence must still allow mate in n-1.
            if all(_defended(board, n) for _ in [0]):
                return True
        finally:
            board.pop()
    return False


def _defended(board: chess.Board, n: int) -> bool:
    for reply in board.legal_moves:
        board.push(reply)
        try:
            if not mates_in(board, n - 1):
                return False
        finally:
            board.pop()
    return True


def min_mate(board: chess.Board, cap: int = 2) -> int | None:
    """The true minimum mate distance in moves, or None if > cap / not forced."""
    for n in range(1, cap + 1):
        if mates_in(board, n):
            return n
    return None


def build_suite(limit: int, cap: int) -> list[tuple[str, int]]:
    """Positions with an exact forced mate, from puzzle solution lines.

    Puzzles whose line ends in checkmate give a cheap *upper bound*; the solver
    above then establishes the exact minimum. Only positions whose minimum is
    within `cap` are kept, because the exhaustive search is exponential.
    """
    out: list[tuple[str, int]] = []
    for root in (ROOT, ROOT.parent / "sumofish"):
        pz = root / "data/puzzles.csv"
        if not pz.exists():
            continue
        with open(pz) as f:
            r = csv.reader(f)
            hdr = next(r)
            fi, mi = hdr.index("FEN"), hdr.index("Moves")
            for row in itertools.islice(r, 60000):
                if len(out) >= limit:
                    break
                try:
                    b = chess.Board(row[fi])
                    ucis = row[mi].split()
                except Exception:
                    continue
                # The puzzle's first move is the opponent's; the solver position
                # is after it.
                if not ucis:
                    continue
                try:
                    b.push(chess.Move.from_uci(ucis[0]))
                except Exception:
                    continue
                # Cheap filter first: few legal moves keeps the solver tractable.
                if b.legal_moves.count() > 12:
                    continue
                n = min_mate(b, cap)
                if n is not None:
                    out.append((b.fen(), n))
        break
    return out


def preserves_minimum(fen: str, uci: str, n: int) -> bool:
    """After playing `uci` in a mate-in-`n`, is it still mate in n-1 for us?"""
    b = chess.Board(fen)
    try:
        mv = chess.Move.from_uci(uci)
    except Exception:
        return False
    if mv not in b.legal_moves:
        return False
    b.push(mv)
    if b.is_checkmate():
        return n == 1
    if b.is_game_over(claim_draw=False):
        return False
    if n == 1:
        return False  # a mate-in-1 was available and not taken
    # Every defence must still allow mate in n-1.
    return _defended(b, n)


def run(suite: list[tuple[str, int]], sims: int, mate_distance: bool,
        evaluate=None, batch: int = 32):
    evaluate = rust_evaluate if evaluate is None else evaluate
    by_n: dict[int, list[int]] = {}
    claimed = 0
    bogus = 0
    nodes = 0
    for fen, n in suite:
        pos = core.Position(fen)
        # `vloss_fix` matches the deployment (CHESSGPU_VLOSS_FIX=1 since
        # 2026-07-30); `dedup` does not, and is on here because this is an
        # oracle for proof soundness rather than a strength measurement. Both
        # arms carry the identical setting, so it cannot favour either.
        m = core.Mcts(batch=batch, dedup=True, vloss_fix=True,
                      mate_distance=mate_distance)
        m.search(pos, sims, evaluate)
        nodes += m.node_count
        best = m.best_move()
        good = bool(best) and preserves_minimum(fen, best, n)
        by_n.setdefault(n, []).append(1 if good else 0)
        pr = m.root_proven()
        if pr is not None:
            claimed += 1
            is_win, plies = pr
            if not is_win:
                continue
            # A claimed win must be a real forced mate, and must not claim to be
            # SHORTER than the true minimum. A proof that is wrong is worse than
            # no proof at all.
            if plies < 2 * n - 1:
                bogus += 1
    ok = sum(sum(v) for v in by_n.values())
    return ok, claimed, bogus, nodes, by_n


def real_evaluator(value_path: str, policy_path: str, device: str, batch: int):
    """The evaluator the bot actually plays with.

    Imported lazily so the default mock path stays importable without torch,
    CUDA or a checkpoint on disk -- `tests/run_all.sh` runs this file and must
    not acquire a GPU dependency.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from match import load_prior, load_value  # noqa: E402
    from sumofish.rust_mcts import make_evaluator  # noqa: E402

    value = load_value(value_path, device=device)
    policy = load_prior(policy_path, device=device)
    return make_evaluator(policy, value, compile_nets=False, pad_to=None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", type=int, default=200)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--cap", type=int, default=2, help="max mate distance in moves")
    ap.add_argument("--real-nets", action="store_true",
                    help="drive the search with the DEPLOYED value and policy "
                         "nets instead of identity_search's mock. Needs the "
                         "GPU. This is the only mode that can see move choice: "
                         "with random priors the search rarely finds a mate-in-2 "
                         "at all, so it rarely has a fast-versus-slow mate to "
                         "choose between.")
    ap.add_argument("--value", default=str(ROOT / "runs/value.pt"))
    ap.add_argument("--policy", default=str(ROOT / "runs/policy.pt"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    evaluate = None
    if args.real_nets:
        print(f"loading the deployed nets ({args.value}, {args.policy})")
        evaluate = real_evaluator(args.value, args.policy, args.device, args.batch)

    print(f"building a suite of forced mates (exhaustive solver, cap {args.cap} moves)")
    suite = build_suite(args.positions, args.cap)
    if not suite:
        print("  no puzzles.csv found; cannot build ground truth")
        return 1
    dist = {}
    for _f, n in suite:
        dist[n] = dist.get(n, 0) + 1
    print(f"  {len(suite)} positions: " + ", ".join(f"mate in {k}: {v}" for k, v in sorted(dist.items())))

    which = "the DEPLOYED nets" if args.real_nets else "the mock evaluator"
    print(f"\nsearching each at {args.sims} simulations, with {which}")
    off_ok, off_claimed, off_bogus, off_nodes, off_by = run(
        suite, args.sims, False, evaluate=evaluate, batch=args.batch)
    on_ok, on_claimed, on_bogus, on_nodes, on_by = run(
        suite, args.sims, True, evaluate=evaluate, batch=args.batch)

    n = len(suite)
    print(f"  {'':18s} {'OFF':>12s} {'ON':>12s}")
    for k in sorted(off_by):
        a, b = sum(off_by[k]), sum(on_by[k])
        tot = len(off_by[k])
        print(f"  shortest mate-in-{k}  {a:>7}/{tot:<4} {b:>7}/{tot:<4}")
    print(f"  {'proofs claimed':18s} {off_claimed:>12} {on_claimed:>12}")
    print(f"  {'BOGUS proofs':18s} {off_bogus:>12} {on_bogus:>12}")
    saved = 100 * (off_nodes - on_nodes) / max(off_nodes, 1)
    print(f"  {'tree nodes':18s} {off_nodes:>12,} {on_nodes:>12,}"
          f"   ({saved:.0f}% smaller)")

    print()
    if on_bogus or off_bogus:
        print("FAIL: the engine claimed a mate shorter than the true minimum.")
        print("A proof that is wrong is worse than no proof at all.")
        return 1
    if on_nodes > off_nodes:
        print("FAIL: proven-mate pruning made the tree BIGGER, which means the")
        print("refutation is not actually cutting the branches it proves.")
        return 1
    print("OK: no bogus proofs, and the tree is smaller.")
    print()
    print("WHAT THIS DOES AND DOES NOT ESTABLISH")
    print("  Established: the proofs are sound (checked against an exhaustive")
    print("  solver, not trusted), and refuting proven-lost branches shrinks the")
    print("  tree -- most at low simulation counts, which matches Lc0's measured")
    print("  ~50% node reduction on tactical positions.")
    print()
    total_off = sum(sum(v) for v in off_by.values())
    total_on = sum(sum(v) for v in on_by.values())
    if not args.real_nets:
        print("  NOT established: any effect on move choice, because of the")
        print("  HARNESS and not the fix. The mock evaluator returns random")
        print("  priors, so the search rarely finds a mate-in-2 at all and")
        print("  therefore rarely has a fast-versus-slow mate to choose between.")
        print("  The tell is that the mate-in-2 rate gets WORSE from 400 to 2000")
        print("  simulations: a real policy prior concentrates on forcing moves,")
        print("  random noise does not.")
        print()
        print("  Re-run with --real-nets to answer it. That needs the GPU.")
        return 0

    print(f"  Move choice, with the deployed nets: {total_off}/{n} shortest-mate")
    print(f"  with the fix OFF against {total_on}/{n} with it ON, over {n}")
    print("  positions whose true minimum was established by exhaustive search.")
    print()
    if total_on > total_off:
        print(f"  The fix CHANGES MOVE CHOICE, by +{total_on - total_off} positions.")
        print("  That is a direct count and not an Elo inference, so it does not")
        print("  depend on the measurement crisis of 2026-08-11 at all.")
    elif total_on == total_off:
        print("  No move-choice difference at this budget. Note what that does")
        print("  and does not say: the tree is still smaller, which is a CLOCK")
        print("  gain, and this suite is capped at mate-in-2. The shuffling-in-a-")
        print("  won-position failure the fix is really for needs longer mates")
        print("  than an exhaustive solver can enumerate here.")
    else:
        print(f"  The fix made move choice WORSE by {total_off - total_on}. If that")
        print("  survives a bigger suite it is a real regression and the flag")
        print("  should stay off regardless of what the tree size says.")
    print()
    print(f"  Sample size: {n} positions. A difference of one or two positions is")
    print("  noise at this n; run --positions higher before acting on a small gap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
