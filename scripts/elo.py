"""Match statistics. Four functions, no dependencies, derived rather than imported.

Shared by `match.py`, which produces the games, and `lab.py`, which reads the
result and decides what to do about it. It lives on its own so the lab does not
have to import torch to work out whether a match concluded anything.

Everything here comes out of one trinomial (W, D, L) from A's point of view.
"""

from __future__ import annotations

import math


def expected_score(elo: float) -> float:
    """The score an engine `elo` points ahead is expected to average."""
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def score_to_elo(score: float) -> float:
    """Invert the above. Clamped, because a clean sweep is infinite Elo."""
    score = min(max(score, 1e-6), 1.0 - 1e-6)
    return -400.0 * math.log10(1.0 / score - 1.0)


def likelihood_of_superiority(w: int, l: int) -> float:  # noqa: E741
    """P(A is actually stronger), from the decisive games alone.

    Draws carry no information about which side is better, so they drop out.
    What remains is a coin-flip test on W vs L, and the normal approximation to
    it is the one every chess tournament tool reports.
    """
    if w + l == 0:
        return 0.5
    return 0.5 * (1.0 + math.erf((w - l) / math.sqrt(2.0 * (w + l))))


# Floor on the per-game score variance, in both the interval and the LLR.
#
# 0.02 is the variance of a match that is 92% draws with the rest split. Past
# that the observed spread stops being an estimate of anything: a run of draws
# is evidence the engines are close, not evidence that the score is known
# exactly, and the formulae below both divide by this number.
#
# A constant, deliberately, and not the 0.25/n it was first written as. A floor
# that decays as 1/n while the LLR's numerator grows as n makes the LLR grow as
# n-squared whenever the floor binds, so an all-draw match crossed the bound at
# n=34 while the docstring claimed "n in the hundreds". A constant floor puts
# that crossing at ~91 games, which is a conclusion 91 straight draws has
# actually earned.
MIN_SCORE_VAR = 0.02


def _score_variance(w: int, d: int, l: int, n: int, s: float) -> float:  # noqa: E741
    """Per-game variance of the score, floored. One definition, two callers.

    It was two definitions and one floor, which is how `score_stats` kept the
    bug `sprt_llr` had just been fixed for: `score_stats(0, 400, 0)` reported
    +0.0 Elo with a zero-width 95% interval, and `decide_promote` reads
    `score_stats`, not `sprt_llr`.
    """
    var = (w * (1 - s) ** 2 + d * (0.5 - s) ** 2 + l * (0 - s) ** 2) / n
    return max(var, MIN_SCORE_VAR)


def score_stats(w: int, d: int, l: int) -> dict:  # noqa: E741
    """Score, its standard error, and the Elo interval that implies.

    The per-game score is 1, 0.5 or 0, so the sample variance is taken over
    those three outcomes directly rather than assuming a binomial. That matters:
    a drawish match has far less variance per game than a decisive one with the
    same score, and treating draws as half a win each would overstate the error
    bars by a wide margin.
    """
    n = w + d + l
    if n == 0:
        return {"n": 0, "score": 0.5, "elo": 0.0, "err": float("inf"), "los": 0.5}
    s = (w + 0.5 * d) / n
    var = _score_variance(w, d, l, n, s)
    sigma = math.sqrt(var / n)
    lo, hi = max(1e-6, s - 1.96 * sigma), min(1 - 1e-6, s + 1.96 * sigma)
    elo = score_to_elo(s)
    return {
        "n": n,
        "score": s,
        "elo": elo,
        # Half-width of the 95% interval. Asymmetric in Elo space, so report
        # the larger side and never claim more precision than exists.
        "err": max(score_to_elo(hi) - elo, elo - score_to_elo(lo)),
        "los": likelihood_of_superiority(w, l),
    }


# A sequential test may not stop before this many games however certain it
# looks. See the variance note in `sprt_llr`: early on, the estimate of the
# spread is itself the least reliable quantity in the formula, and the formula
# divides by it.
MIN_SPRT_GAMES = 32


def sprt_llr(w: int, d: int, l: int, elo0: float, elo1: float) -> float:  # noqa: E741
    """Log-likelihood ratio for H1 (elo>=elo1) against H0 (elo<=elo0).

    A fixed-length match wastes games: once the result is obvious it keeps
    playing, and when it is genuinely close no number of games was going to
    settle it. A sequential test watches the evidence accumulate and stops the
    moment it crosses a bound, which usually halves the games needed.

    This is the Gaussian approximation to the LLR, using the observed per-game
    score variance. It is the simple form and it is slightly optimistic
    compared to the pentanomial version fishtest uses, which exploits the fact
    that paired games are correlated. Good enough to decide whether to keep a
    change; do not publish a rating list with it.

    ## The small-sample trap, which this got wrong and which cost nothing only
    ## because it was caught before the first match read it

    The formula divides by the observed variance, and the observed variance of
    two drawn games is exactly zero. Clamping that to 1e-9 to avoid the
    division by zero does not avoid anything: it asserts the score is known to
    nine decimal places, and the LLR comes out at -1,289,954 against a bound of
    -2.944. The first pair of a match decides it. Measured, all three ways:

        first pair 2 draws     LLR    -1,289,954   ("A is not better")
        first pair 2 A-wins    LLR    34,625,973   ("A is better")
        first pair 2 A-losses  LLR   -37,205,881   ("A is not better")

    Two drawn games is the single most likely opening to a match between two
    configurations of the same engine, so the default outcome was a match that
    stopped after ten minutes, exited 0, and recorded a conclusion.

    Two guards, because either alone is thin: the variance floor at
    `MIN_SCORE_VAR` (see its note for why it is a constant and not 1/n), and a
    hard refusal to conclude anything before `MIN_SPRT_GAMES`.
    """
    n = w + d + l
    if n < MIN_SPRT_GAMES:
        return 0.0
    s = (w + 0.5 * d) / n
    var = _score_variance(w, d, l, n, s)
    s0, s1 = expected_score(elo0), expected_score(elo1)
    return n * (s1 - s0) * (s - 0.5 * (s0 + s1)) / var


def sprt_bounds(alpha: float = 0.05, beta: float = 0.05) -> tuple[float, float]:
    """(lower, upper). Cross the upper and H1 wins; cross the lower and H0 does."""
    return math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)


def tally(records: list[dict]) -> tuple[int, int, int]:
    """(W, D, L) from A's point of view, over `match.py` game records."""
    w = sum(r["score"] == 1.0 for r in records)
    d = sum(r["score"] == 0.5 for r in records)
    return w, d, len(records) - w - d


# ---------------------------------------------------------------------------
# scoring the PAIR instead of the game
#
# The match plays every opening twice with the colours swapped, and then throws
# that structure away by counting wins, draws and losses. Counting the pair
# instead is worth roughly a factor of two in games, for about twenty lines.
#
# Why it works, and why the direction is the opposite of what the docstring
# above assumed. A pair's two games share an opening, so its bias appears in
# both -- once for A and once against. Summing the pair cancels it, which is
# the entire reason for playing paired colours in the first place. Measured on
# 231 games of a real match: within-pair correlation rho = -0.56, and
#
#     r = Var(pair sum) / (2 * Var(game)) = 0.44
#
# Below 1 means the pair is *less* variable than two independent games, so the
# trinomial form was not "slightly optimistic" as first written here. It was
# conservative by 2.3x, and it cost a real conclusion: that match reads
# +22.4 +-29.3 ("no difference") counted by game, and +22.5 [+3.2, +41.9]
# counted by pair, which excludes zero.
#
# Two cautions, both from the Council that found this. The gain is largest when
# the engines are most alike, because r is (opening bias) / (engine difference),
# so it shrinks as the arms genuinely diverge -- do not expect 2.3x on a match
# between different-sized nets. And r must be re-estimated from each match's own
# pairs rather than stored as a constant, which is what `pairing_efficiency`
# is for.

# Floor on the variance of a pair sum. Two games' worth of `MIN_SCORE_VAR`,
# which is what a pair would carry if its games were independent.
MIN_PAIR_VAR = 2 * MIN_SCORE_VAR


def pair_sums(records: list[dict]) -> list[float]:
    """A's total score over each COMPLETE colour-swapped pair, in order.

    `match.py` plays game 2k and 2k+1 from opening k with the colours swapped,
    so the pair index is the game index over two. An unfinished trailing pair
    is dropped rather than counted as half: the whole point is that the two
    halves cancel each other's bias, and one half alone cancels nothing.
    """
    by_pair: dict[int, list[float]] = {}
    for record in records:
        by_pair.setdefault(record["game"] // 2, []).append(record["score"])
    return [sum(scores) for _, scores in sorted(by_pair.items()) if len(scores) == 2]


def pair_stats(sums: list[float]) -> dict:
    """Score, Elo and a 95% interval, taking the pair as the unit of sampling.

    Same shape as `score_stats` so the two are interchangeable at call sites,
    plus `pairs`. A pair sum runs over {0, 0.5, 1, 1.5, 2} and its expectation
    under a dead-even match is 1, so the score is half the mean pair sum.
    """
    n = len(sums)
    if n == 0:
        return {"n": 0, "pairs": 0, "score": 0.5, "elo": 0.0,
                "err": float("inf"), "los": 0.5}
    mean = sum(sums) / n
    var = max(sum((x - mean) ** 2 for x in sums) / n, MIN_PAIR_VAR)
    sigma = math.sqrt(var / n) / 2.0          # of the score, not of the pair sum
    s = mean / 2.0
    lo, hi = max(1e-6, s - 1.96 * sigma), min(1 - 1e-6, s + 1.96 * sigma)
    elo = score_to_elo(s)
    # LOS from the same normal approximation. Draws drop out of the trinomial
    # version because they carry no information; here a drawn PAIR is a pair
    # sum of exactly 1 and contributes nothing to the mean's distance from
    # even, which is the same statement made continuously.
    los = 0.5 * (1.0 + math.erf((s - 0.5) / (sigma * math.sqrt(2.0)))) if sigma else 0.5
    return {
        "n": 2 * n,
        "pairs": n,
        "score": s,
        "elo": elo,
        "err": max(score_to_elo(hi) - elo, elo - score_to_elo(lo)),
        "los": los,
    }


def sprt_llr_pairs(sums: list[float], elo0: float, elo1: float) -> float:
    """The sequential test, on pairs. Reduces exactly to `sprt_llr` at r = 1.

    Worth checking by hand once: substituting Var(pair) = 2*Var(game) and
    target = 2*score turns this expression into the game-based one, so the
    pairing buys nothing when the two games really are independent and buys
    the whole 1/r when they are not. Nothing is assumed about r; it is measured
    by the variance of the sums that were actually played.
    """
    n = len(sums)
    if 2 * n < MIN_SPRT_GAMES:
        return 0.0
    mean = sum(sums) / n
    var = max(sum((x - mean) ** 2 for x in sums) / n, MIN_PAIR_VAR)
    t0, t1 = 2 * expected_score(elo0), 2 * expected_score(elo1)
    return n * (t1 - t0) * (mean - 0.5 * (t0 + t1)) / var


def pairing_efficiency(sums: list[float], w: int, d: int, l: int) -> float:  # noqa: E741
    """r = Var(pair sum) / (2 * Var(game)). Below 1 means pairing is paying.

    Report it. It is the number that says how much the colour swap is actually
    doing on THIS match, and 1/r is the factor by which counting pairs beats
    counting games.
    """
    n = w + d + l
    if not sums or n == 0:
        return 1.0
    mean = sum(sums) / len(sums)
    var_pair = sum((x - mean) ** 2 for x in sums) / len(sums)
    s = (w + 0.5 * d) / n
    var_game = (w * (1 - s) ** 2 + d * (0.5 - s) ** 2 + l * (0 - s) ** 2) / n
    if var_game <= 0:
        return 1.0
    # Floored away from zero. Two identical pairs give var_pair = 0, which is
    # "infinitely efficient" and is a small-sample artifact rather than a
    # finding -- and callers report 1/r, so it crashed a completed two-game
    # match on the summary line after every game had been played and logged.
    return max(var_pair / (2 * var_game), 1e-3)
