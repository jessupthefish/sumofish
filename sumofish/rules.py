"""Is this position over, and what is it worth?

One function, and it exists because the obvious way to ask that question was
41% of the search.

`board.outcome(claim_draw=True)` is the natural call and it is a trap in a tree
search. Profiled over a 5-second search on a midgame position, 3,393 calls to
`can_claim_threefold_repetition` cost 1.95s of a 4.79s search: 573us each,
against 5% of the search spent in the neural network it exists to serve. The
cost is structural rather than an implementation detail. To decide whether a
draw is *claimable*, python-chess has to check both whether the current
position has already occurred three times AND whether any legal move reaches a
third occurrence, so it plays out every legal move and rebuilds a transposition
key for each, at every leaf.

The second half of that is not merely expensive, it is **wrong for a search**.
"A move exists that would let me claim a draw" is not a property of this
position, it is a property of a child position, and the search is about to
visit that child and discover it there. Treating it as terminal here prunes the
line before the search can decide whether the draw is even wanted.

So this asks the narrower question, in cost order, cheapest first:

    checkmate / stalemate   one legal-move generation, short-circuited
    insufficient material   a popcount
    fifty-move              an integer compare
    threefold               a stack walk, guarded by that integer compare

The guard is what makes the last one nearly free. A position cannot occur for
the third time without at least eight plies of reversible moves behind it (four
by each side to return the pieces twice), and `halfmove_clock` counts exactly
those, resetting on every capture and every pawn move. Below 8 the answer is no
without looking at the stack, which covers the overwhelming majority of nodes.

Measured on the same position: 1795 nps -> 2724 nps, +52%.

The one thing not to do here is drop repetition awareness altogether, which is
the version that benchmarks best. An engine blind to repetition shuffles away a
won game and misses the perpetual that saves a lost one, and neither shows up
in a speed measurement.

Not to be confused with `policy._drawish`, which asks the genuinely different
question "would MY move hand the opponent a draw claim". That one must stay
`can_claim_draw()` and its docstring explains why. The difference is that it is
a searchless engine's root-level guard, asked once per candidate move, where
there is no child node coming along later to notice.
"""

from __future__ import annotations

import chess

# Plies of reversible play needed before a third occurrence is possible: the
# position must be returned to twice, two plies per side per return.
_MIN_PLIES_FOR_THREEFOLD = 8


def terminal_value(board: chess.Board) -> float | None:
    """Value to the SIDE TO MOVE if the game ends here, else None.

    Follows this codebase's convention everywhere: 1.0 is a win for whoever is
    about to move, 0.0 a loss, 0.5 a draw. A player never delivers mate with
    their own move, so a checkmated position is worth 0.0 to the side facing it.
    """
    # `any()` stops at the first legal move, so this is cheap in the ~99.9% of
    # positions that have one. Asking is_checkmate() and then is_stalemate()
    # would generate legal moves twice for the same answer.
    if not any(board.generate_legal_moves()):
        return 0.0 if board.is_check() else 0.5

    if board.is_insufficient_material():
        return 0.5

    # Both draw rules below are the *claimable* ones rather than the automatic
    # 75-move and fivefold versions. Under lichess and FIDE rules a player who
    # wants the draw gets it, so a search that assumed otherwise would be
    # evaluating positions that cannot be reached against a competent opponent.
    if board.halfmove_clock >= 100:
        return 0.5

    if board.halfmove_clock >= _MIN_PLIES_FOR_THREEFOLD and board.is_repetition(3):
        return 0.5

    return None


def terminal_value_legacy(board: chess.Board) -> float | None:
    """What the search used to do. Kept only so the change can be measured.

    Identical to `terminal_value` except that it also treats a position as over
    when a draw is merely *reachable* by one legal move. That is the half this
    module argues is wrong for a tree search, and "wrong" was an argument, not
    a measurement: the four search changes in this session landed on a stopwatch
    with nothing showing they were strength-neutral.

    The other three changes cannot move strength and are proved not to --
    `tokenize_board` is byte-identical to `tokenize` over 100,836 positions and
    the action-key table agrees with `MOVE_TO_ACTION` over 1.6M legal moves --
    and tree reuse has its own switch. This is the only behavioural difference
    left, so a fixed-simulation match between the two settles the whole claim.

    Do not use it to play. It costs 573us a call.
    """
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    return 0.0 if outcome.winner is not None else 0.5
