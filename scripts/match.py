#!/usr/bin/env python
"""Head-to-head match play. The missing instrument.

Every "did that help?" question in this project has been unanswerable, and the
two things standing in for an answer both fail at the scale of the changes
being made:

  * **Puzzle accuracy** measures tactics on 1000 positions, which carries a
    binomial sigma of +-1.5%. The entire 150k->300k half of the state-value run
    moved it 1.7 points. That is noise wearing a number's clothes.
  * **The lichess rating** has an RD of +-72 in bullet and needs days to move.

This script answers the question directly: play the two configurations against
each other a few hundred times and count. It is the only measurement here whose
error bars shrink on demand, by playing more games.

## What makes a match fair

Three things, and skipping any one of them produces a number that looks
rigorous and is not.

**Paired openings.** Each opening is played twice with the colours swapped, so
a book line that happens to favour White cannot favour whichever engine drew
White more often. Games are therefore reported in pairs and `--games` is
rounded down to an even number.

**A real book.** From the same starting position two deterministic engines play
one game, forever. `data/eco_openings.pgn` supplies a few thousand distinct
6-12 ply openings; the match walks them in a seeded shuffle, so a rerun with the
same seed sees the same openings and a different seed is an independent sample.

**Fixed sims OR fixed time, chosen deliberately.** These answer different
questions and confusing them is the classic mistake:

    --sims N     equal thinking, so this measures the QUALITY of the search
                 and the nets. Use it to compare checkpoints. A speed change
                 must not move this number.
    --time S     equal wall clock, so this measures STRENGTH AS DEPLOYED, and
                 a speedup shows up here as extra simulations. Use it to
                 decide whether an optimisation was worth it.

Run both. A change that wins on time and is flat on sims is a pure speedup; a
change that wins on sims is a real improvement in judgement.

## Reading the output

    +-------------------------------------------------------------+
    |  A: value=runs/value.pt sims=400                            |
    |  B: value=runs/9M-sv-warm-full/best.pt sims=400             |
    |  120 games  W47 D38 L35   score 55.0%  elo +34.9 +-31.2     |
    |  LOS 78.4%   LLR 0.62 (-2.94, 2.94)                         |
    +-------------------------------------------------------------+

`elo` is A's advantage over B. The `+-` is a 95% interval, and until it
excludes zero the match has not concluded anything. `LOS` is the probability
that A is genuinely better, which is the honest way to read a match that has
not reached significance. `LLR` is the sequential test: it stops the match
early the moment the evidence is decisive in either direction, which typically
saves half the games.

## Resuming

Results append to `runs/matches/<name>/games.jsonl` and the match skips game
indices already present. Ctrl-C is safe and rerunning the same command
continues. The full PGN is written beside it, because PHILOSOPHY.md says to
watch real games and a match is a few hundred of them.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sumofish.engines.neural_engine import load_policy  # noqa: E402
from sumofish.hlgauss import HLGauss  # noqa: E402
from sumofish.mcts import MCTS  # noqa: E402
from sumofish.model import ChessTransformer, ModelConfig  # noqa: E402
from sumofish.rules import terminal_value, terminal_value_legacy  # noqa: E402
from sumofish.rust_mcts import select_mcts_class
from sumofish.uci import Limits
# The SAME time management the bot runs, imported rather than reimplemented:
# a copy would drift, and the whole point of --tc is to exercise the
# deployed code path.
from sumofish.engines.search_engine import (
    INSTAMOVE_SECONDS, decided, think_time,
)  # noqa: E402
from sumofish.value_policy import ValuePolicy  # noqa: E402

# The match statistics live in `elo.py` so that `lab.py` can read a result
# without importing torch to do it.
from elo import (  # noqa: E402
    pair_stats,
    pair_sums,
    pairing_efficiency,
    score_stats,
    sprt_bounds,
    sprt_llr_pairs,
    tally,
)

# ---------------------------------------------------------------------------
# players


class Clock:
    """A real game clock: base plus increment, per side, in seconds.

    Why this exists (2026-08-13): every arm in `runs/matches` is fixed-
    simulation or fixed-movetime, and the deployed engine is neither. It is
    clock-bound, and `docs/OPERATING-POINT.md` measures the gap at 9.2
    doublings of search. Fixed movetime also cannot express the thing early
    stopping is FOR -- time saved on one move raising the budget of later ones
    -- because each move's allowance is independent under it.

    Times are seconds here and milliseconds in `Limits`, matching UCI. The
    conversion happens in `limits()` and nowhere else, because a factor of 1000
    applied in two places is a factor of 1000 applied in one of them.
    """

    def __init__(self, base: float, increment: float):
        self.base = base
        self.increment = increment
        self.remaining = {chess.WHITE: base, chess.BLACK: base}
        self.spent = {chess.WHITE: 0.0, chess.BLACK: 0.0}

    def limits(self) -> Limits:
        return Limits(
            wtime=int(self.remaining[chess.WHITE] * 1000),
            btime=int(self.remaining[chess.BLACK] * 1000),
            winc=int(self.increment * 1000),
            binc=int(self.increment * 1000),
        )

    def charge(self, side: bool, elapsed: float) -> bool:
        """Bill `side` for a move. Returns False if it flagged.

        The increment is added AFTER the deduction and only if the side did not
        flag, which is the FIDE rule and also the only order that lets a player
        lose on time in a position with an increment.
        """
        self.remaining[side] -= elapsed
        self.spent[side] += elapsed
        if self.remaining[side] < 0:
            return False
        self.remaining[side] += self.increment
        return True


@dataclass
class Spec:
    """One side of the match, fully described."""

    label: str
    value: str
    policy: str
    # The CONTENT of each checkpoint, not just where it was. `value` and
    # `policy` are paths, and `runs/value.pt` is a path that gets overwritten
    # by every promotion, so a resume keyed on the path alone will happily
    # extend a match with a different network and report the two halves as one
    # population. PHILOSOPHY:197-199 names this hazard and `release.py:95`
    # already content-hashes for exactly this reason. These fields land inside
    # the resume fingerprint automatically, because it hashes `Spec.__dict__`.
    value_sha: str
    policy_sha: str
    sims: int
    batch: int
    c_puct: float
    # The exploration constant that ACTUALLY BINDS under AlphaZero's schedule.
    # `c_puct_at()` reads `c_puct` only when `c_puct_base is None`, i.e. only
    # under --fixed-cpuct; otherwise it returns
    # `ln((1 + visits + base)/base) + c_puct_init` and `c_puct` is dead. So
    # --cpuct is a silent no-op in the SHIPPED configuration, and every match
    # before 2026-08-09 ran the hardcoded 1.25 because nothing passed this.
    c_puct_init: float
    fpu: float
    movetime: float | None
    searchless: bool
    reuse: bool
    legacy_draws: bool
    fixed_cpuct: bool
    # Which search implementation. "python" is sumofish.mcts; "rust" is
    # sumofish_core, which is byte-identical to it in the plain configuration.
    core: str = "python"
    # The two speed flags. Each is identity-preserving in the SEARCH but changes
    # the number of rows in the forward pass, and the network is not batch-shape
    # invariant, so with real weights they change what gets played in about one
    # position in eight. That is what this match exists to price.
    dedup: bool = False
    compile_nets: bool = False
    # Two search-quality fixes (rust only). Neither touches the forward pass
    # shape, so neither is a speed question -- but both are genuine behaviour
    # changes against the faithful port and need the same Elo verdict from
    # this harness before either earns a default.
    mate_distance: bool = False
    vloss_fix: bool = False
    # Time management, exercised only under --tc. Both are OFF in deployment as
    # of 2026-08-13 and both exist to be priced here, which was impossible
    # before --tc: `Player.move` calls `mcts.search()` directly and never
    # touched `search_engine.choose`, so `think_time` was not in the loop at
    # all and a fixed movetime cannot express "banked time raises later moves'
    # budgets", which is early stopping's entire payoff.
    instamove: bool = False
    early_stop: bool = False
    # True under --tc. Like `movetime`, it means the CLOCK stops the search, so
    # the simulation cap must not also bind -- see the comment at the
    # `simulations=` argument below, which documents this exact bug happening
    # once already for --time. It happened again for --tc, and was caught by a
    # 2-game smoke test in which side A spent 3x side B's clock: A took the
    # sliced search path and ran to its deadline while B stopped at 400 sims.
    # The match was measuring "400 sims vs clock-bound", not the flags.
    clocked: bool = False
    # Set (not None) makes this side an external UCI engine (Stockfish) at a
    # fixed node budget instead of the neural MCTS. See `Player.__init__` and
    # the Arbiter docstring above: fixed NODES, full strength, is the same
    # discipline the adjudicator already uses, applied to a playing side.
    stockfish_nodes: int | None = None

    def describe(self) -> str:
        if self.stockfish_nodes is not None:
            return f"stockfish nodes={self.stockfish_nodes} (full strength, node-limited)"
        if self.searchless:
            return f"value={self.value} searchless"
        if self.clocked:
            budget = "clock"
        elif self.movetime:
            budget = f"{self.movetime}s/move"
        else:
            budget = f"{self.sims} sims"
        flags = "".join(
            c for c, on in (
                ("D", self.dedup),
                ("C", self.compile_nets),
                ("M", self.mate_distance),
                ("V", self.vloss_fix),
                ("I", self.instamove),
                ("E", self.early_stop),
            ) if on
        )
        return (
            f"core={self.core}{'+' + flags if flags else ''} "
            f"value={self.value} policy={self.policy} {budget} "
            f"batch={self.batch} "
            f"{'cpuct=' + str(self.c_puct) if self.fixed_cpuct else 'cpuct_init=' + str(self.c_puct_init)} "
            f"fpu={self.fpu} "
            f"reuse={'on' if self.reuse else 'off'}"
        )


_MODEL_CACHE: dict[tuple[str, str], object] = {}


def load_value(path: str, device: str = "cuda:0") -> ValuePolicy:
    """Load a state-value checkpoint, reusing it if both sides ask for it.

    A config-vs-config match (same net, different c_puct) would otherwise put
    two copies of the same weights on the card for no reason. The architecture
    comes out of the checkpoint rather than being assumed to be the 9M preset,
    so this keeps working the day a 136M net exists.
    """
    key = (path, device)
    if key not in _MODEL_CACHE:
        ck = torch.load(path, map_location=device, weights_only=False)
        model = ChessTransformer(ModelConfig(**ck["cfg"]))
        state = ck.get("ema") or ck["model"]
        model.load_state_dict({k: v.float() for k, v in state.items()})
        vp = ValuePolicy(model, HLGauss(bins=ck["cfg"]["output_size"]), device=device)
        vp.step = ck.get("step")
        _MODEL_CACHE[key] = vp
    return _MODEL_CACHE[key]  # type: ignore[return-value]


_POLICY_CACHE: dict[tuple[str, str], object] = {}


def load_prior(path: str, device: str = "cuda:0"):
    key = (path, device)
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = load_policy(path, device=device)[0]
    return _POLICY_CACHE[key]


class Player:
    """Something that answers `move(board) -> (move, win probability)`.

    The win probability is from the side to move's perspective, matching the
    convention everywhere else in this codebase, and it exists for adjudication
    rather than for display.

    # Stockfish as a playing side, not just an adjudicator
    #
    # `Arbiter` above already established the rule this project uses for a
    # reference engine: fixed NODES, full strength, never Skill Level (random
    # blundering, rejected by PHILOSOPHY even for building difficulty on
    # purpose) and never fixed depth (not reproducible under CPU contention,
    # and this box runs Stockfish CPU-side while two training jobs and the
    # live bot contend for the GPU). A playing side is held to the identical
    # rule, for the identical reason: the only degradation this harness ever
    # allows is on the SAME axis the article under test is allowed to spend --
    # nodes for Stockfish, sims for SumoFish -- never in either one's judgement.
    #
    # A Stockfish-backed `Player` therefore has no value net, no policy net,
    # no MCTS: `spec.stockfish_nodes` being set short-circuits all of that and
    # routes `move()` through a UCI `SimpleEngine.play()` at that fixed node
    # count instead. It still returns a (move, win-probability) pair in the
    # same side-to-move convention as the neural players, computed from
    # Stockfish's own `info["score"]` via the same `wdl(model="sf12")` used by
    # `Arbiter.agrees`, so the adjudication curve and PGN/game-record plumbing
    # in `play_game` do not need to know or care which kind of player produced
    # a move.
    """

    def __init__(
        self, spec: Spec, device: str = "cuda:0", stockfish_path: str | None = None
    ) -> None:
        self.spec = spec
        self.engine = None
        # Identifies the current game to python-chess; a new object per game
        # triggers `ucinewgame`. See new_game().
        self._game_token = object()
        if spec.stockfish_nodes is not None:
            import chess.engine

            path = stockfish_path or str(
                ROOT / "tools/stockfish/stockfish-ubuntu-x86-64-bmi2"
            )
            self.engine = chess.engine.SimpleEngine.popen_uci(path)
            self.value = None
            self.mcts = None
            return
        self.value = load_value(spec.value, device)
        if spec.searchless:
            self.mcts = None
        elif spec.core == "rust":
            from sumofish.rust_mcts import RustMCTS

            self.mcts = RustMCTS(
                self.value,
                policy=load_prior(spec.policy, device),
                c_puct=spec.c_puct,
                c_puct_init=spec.c_puct_init,
                fpu=spec.fpu,
                simulations=10**9 if (spec.movetime or spec.clocked) else spec.sims,
                batch=spec.batch,
                reuse=spec.reuse,
                dedup=spec.dedup,
                compile_nets=spec.compile_nets,
                mate_distance=spec.mate_distance,
                vloss_fix=spec.vloss_fix,
                # MUST accompany compile_nets. Without it the row count is
                # ragged (dedup makes it vary, and root expansion sends 1), so
                # CUDA graphs record a fresh graph per distinct size -- torch
                # warns "observed 9 distinct sizes" -- and the recompiles land
                # inside a search on a running clock. Omitting it silently
                # DEGRADES the arm being measured, which would understate the
                # very thing this match exists to price.
                pad_batches=spec.compile_nets,
                # The Rust core has no injectable terminal hook, so
                # --legacy-draws is not available to it. Fail rather than
                # silently ignore the flag: a match that quietly did not test
                # what was asked is worse than one that refused.
                c_puct_base=None if spec.fixed_cpuct else 19652.0,
            )
            if spec.legacy_draws:
                raise SystemExit(
                    "--legacy-draws is not supported by the rust core "
                    "(no injectable terminal hook)"
                )
        else:
            self.mcts = MCTS(
                self.value,
                policy=load_prior(spec.policy, device),
                c_puct=spec.c_puct,
                c_puct_init=spec.c_puct_init,
                fpu=spec.fpu,
                # `--time` means the CLOCK decides, so the simulation count
                # must not also bind. It did: `simulations=spec.sims` with a
                # 400 default meant `--time 3.0` ran min(400 sims, 3s) ~= 0.12s,
                # and the deadline never applied. Both matches feeding the lab's
                # promotion gate were equal-simulation matches while the gate
                # applied an equal-TIME bar. `smoke.py` and `bench_search.py`
                # already write 10**9 for exactly this reason; this was the one
                # place that forgot.
                simulations=10**9 if (spec.movetime or spec.clocked) else spec.sims,
                batch=spec.batch,
                reuse=spec.reuse,
                terminal=terminal_value_legacy if spec.legacy_draws else terminal_value,
                # None restores the pre-schedule constant c_puct.
                c_puct_base=None if spec.fixed_cpuct else 19652.0,
            )
            # The Python core implements none of these four. They were dropped
            # SILENTLY until 2026-08-11, and worse than silently: `Spec.describe()`
            # builds its flag string from the same fields regardless of core, and
            # `config.json` records `"vloss_fix": true`. So
            # `--core python --a-vloss-fix` printed `core=python+V`, wrote
            # provenance asserting the flag was on, and ran without it. That is
            # exactly the "records the REQUEST rather than the EFFECT" failure
            # LAB-NOTES 2026-08-09 names, in the file that fixed it elsewhere.
            #
            # Refuse, on the pattern the rust branch already uses for
            # --legacy-draws directly above: a match that quietly did not test
            # what was asked is worse than one that refused. No archived result
            # is affected, because every in-repo caller passes --core rust, but
            # --core python is a supported path for the oracle comparison.
            unsupported = [
                name for name, on in (
                    ("--dedup", spec.dedup),
                    ("--compile", spec.compile_nets),
                    ("--mate-distance", spec.mate_distance),
                    ("--vloss-fix", spec.vloss_fix),
                ) if on
            ]
            if unsupported:
                raise SystemExit(
                    f"{', '.join(unsupported)} {'is' if len(unsupported) == 1 else 'are'} "
                    "implemented only in the rust core, but --core python was "
                    "requested. Re-run with --core rust, or drop the flag: "
                    "running without it would record a config that did not play."
                )

    def new_game(self) -> None:
        if self.engine is not None:
            # CORRECTED 2026-08-10. This used to return here without telling
            # Stockfish a new game had started, on the reasoning that each
            # move() sends the full position so ucinewgame is not needed for
            # CORRECTNESS, and that hash reuse between games is "noise next to
            # the game-to-game variance". The first half is true and the second
            # is not, in a way that is not noise but bias.
            #
            # Stockfish plays at a fixed NODE budget here. A warm transposition
            # table finds better moves inside the same budget, so its strength
            # depends on how many games it has already played in this process.
            # Consequences, all of them measured or provable rather than
            # theoretical:
            #
            #   * a game's result depended on which games came BEFORE it, so a
            #     match was not reproducible from its seed alone. Splitting a
            #     20-game match across two shards changed 18 of the 20 games
            #     while partitioning them perfectly (2026-08-10);
            #   * that makes sharding BIASED, not merely different: a shard
            #     plays 1/N as many games, so its Stockfish runs colder and
            #     therefore weaker throughout, inflating our score;
            #   * and a resumed match differed from an uninterrupted one, since
            #     the resumed half starts cold.
            #
            # python-chess sends `ucinewgame` when the `game` object handed to
            # play() changes, so a fresh token per game clears the hash and
            # makes every game start identically.
            self._game_token = object()
            return
        # Tree reuse, once it exists, must not carry a subtree from the
        # previous game into this one.
        self.evals = self.unique_evals = self.searches = 0
        reset = getattr(self.mcts, "reset", None)
        if reset is not None:
            reset()

    def move(self, board: chess.Board,
             clock: "Clock | None" = None) -> tuple[chess.Move, float]:
        """Play a move. `clock` is the live game clock under --tc, else None.

        With a clock, the neural side goes through `search_engine.think_time`
        -- the SAME function the bot uses, not a copy -- so instamove and early
        stopping are exercised exactly as deployed. Without one the behaviour is
        unchanged from before --tc existed.
        """
        if self.engine is not None:
            import chess.engine

            limit = (
                chess.engine.Limit(
                    white_clock=clock.remaining[chess.WHITE],
                    black_clock=clock.remaining[chess.BLACK],
                    white_inc=clock.increment, black_inc=clock.increment,
                )
                if clock is not None
                # Fixed NODES, never depth and never Skill Level: the Arbiter
                # docstring gives the reasoning and a playing side is held to
                # the same rule.
                else chess.engine.Limit(nodes=self.spec.stockfish_nodes)
            )
            result = self.engine.play(
                board,
                limit,
                info=chess.engine.INFO_SCORE,
                # Changing this token is what makes python-chess emit
                # `ucinewgame`. See new_game().
                game=self._game_token,
            )
            score = (result.info or {}).get("score")
            # Side-to-move's perspective, matching the neural players' `root.q`
            # convention, via the same WDL model the Arbiter already uses.
            # Falls back to 0.5 (unknown) rather than raising: a UCI engine
            # that omits `score` on some position must not crash a match hours
            # into a run over a value that is only ever used for the
            # adjudication curve, never for choosing the move itself.
            wp = score.pov(board.turn).wdl(model="sf12").expectation() if score else 0.5
            assert result.move is not None
            return result.move, wp
        if self.mcts is None:
            ranked = self.value.rank_moves(board)
            return ranked[0][0], ranked[0][1]

        should_stop = None
        if clock is not None:
            budget = think_time(clock.limits(), board.turn)
            if self.spec.instamove and board.legal_moves.count() == 1:
                budget = min(budget, INSTAMOVE_SECONDS)
            deadline = time.perf_counter() + budget
            if self.spec.early_stop:
                started = time.perf_counter()

                def should_stop(pairs, done, remaining):
                    return decided([n for _, n in pairs], done,
                                   time.perf_counter() - started, remaining, budget)
        else:
            deadline = (
                time.perf_counter() + self.spec.movetime
                if self.spec.movetime else None
            )

        kwargs = {}
        if should_stop is not None:
            # Only the Rust wrapper accepts it, and only it is ever configured
            # with early_stop -- see the guard in main(). Passed conditionally
            # so the Python core keeps working as the identity oracle.
            kwargs["should_stop"] = should_stop
        root, visits = self.mcts.search(board, deadline=deadline, **kwargs)
        # The quantity a slower net actually loses, and until 2026-08-14 the one
        # the harness threw away. Under --time and --tc `simulations` is set to
        # 10**9 so the clock binds, which makes the evaluation count the
        # DEPENDENT variable of the experiment -- and `games.jsonl` recorded
        # result, plies, seconds and curve, but never this. A wall-clock match
        # therefore could not see what it was measuring. PHILOSOPHY also
        # mandates reporting unique/s rather than raw nps, which needs the
        # second counter.
        self.evals += getattr(self.mcts, "evaluations", 0)
        self.unique_evals += getattr(self.mcts, "unique_evaluations", 0)
        self.searches += 1
        move = max(visits.items(), key=lambda kv: kv[1])[0]
        return move, root.q

    def warmup(self, seconds: float = 2.0) -> None:
        """Compile the CUDA kernels before the first move that is charged for.

        Deliberately NOT `self.move(chess.Board())`, which is what this used to
        be. That call takes its budget from the match's mode, and under --tc
        the mode is "the clock decides" -- so with no clock passed it got
        `deadline=None` on top of `simulations=10**9` and searched forever. The
        first --tc smoke test hung in the warmup before playing a single move.

        A warmup has nothing to do with the match's budget: its job is to make
        the first real search pay for arithmetic rather than for compilation.
        So it states its own bound and takes it from nothing else.
        """
        if self.engine is not None:
            self.move(chess.Board())      # a UCI handshake, already bounded
            return
        if self.mcts is None:
            self.value.rank_moves(chess.Board())
            return
        self.mcts.search(chess.Board(), deadline=time.perf_counter() + seconds)

    def close(self) -> None:
        if self.engine is not None:
            try:
                self.engine.quit()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# openings


def load_openings(path: Path, min_ply: int, max_ply: int, seed: int) -> list[list[str]]:
    """Distinct book lines, seeded-shuffled, as UCI strings.

    Deduplicated by the resulting position rather than by the move list,
    because the ECO file reaches the same position by several transpositions
    and playing all of them would silently weight one opening several times.
    """
    lines: list[list[str]] = []
    seen: set[str] = set()
    with open(path) as fh:
        while (game := chess.pgn.read_game(fh)) is not None:
            moves = list(game.mainline_moves())
            if not min_ply <= len(moves) <= max_ply:
                continue
            board = chess.Board()
            for mv in moves:
                board.push(mv)
            key = board.epd()
            if key in seen:
                continue
            seen.add(key)
            lines.append([mv.uci() for mv in moves])
    random.Random(seed).shuffle(lines)
    return lines


import functools  # noqa: E402
import hashlib  # noqa: E402
import subprocess  # noqa: E402


@functools.lru_cache(maxsize=1)
def contending_units() -> list[str]:
    """Anything on this box that would bias a wall-clock match, by name.

    Deliberately includes the live bot, which `lab.py::wait_for_quiet` excludes:
    a fixed-simulation arm is immune to contention (it costs time, not validity)
    and a clock arm is not.
    """
    import subprocess
    busy = []
    # The unit THIS process is running under, if any. Without this the guard
    # blocks itself: a match launched as `systemd-run --user --unit=
    # sumofish-compile-gate` matches its own `sumofish-*` glob, is listed as
    # running, and refuses to start on the grounds that it is already running.
    # Reading the cgroup is the reliable way to ask; $INVOCATION_ID says that
    # you are under systemd but not which unit.
    own_unit = ""
    try:
        for line in open("/proc/self/cgroup"):
            for part in line.strip().split("/"):
                if part.endswith((".service", ".scope")):
                    own_unit = part
    except OSError:
        pass
    r = subprocess.run(["systemctl", "--user", "list-units", "--state=running",
                        "--no-legend", "--plain", "sumofish-*"],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        unit = line.split()[0] if line.split() else ""
        # The rating sampler and the watchdogs are timers doing nothing on the
        # GPU; only long-running GPU work matters here.
        if unit and not unit.startswith(("sumofish-rating", "sumofish-watchdog",
                                         "sumofish-train-watchdog")):
            if unit == own_unit:
                continue
            busy.append(f"{unit} (running)")
    # Anything on the card whose working directory is this repo. Matching on the
    # COMMAND LINE would miss `.venv/bin/python scripts/match.py --name ...`,
    # which contains no "sumofish" at all, and would also self-match -- the trap
    # LAB-NOTES records for `pgrep -f`, and the reason `lab.py` reads /proc too.
    import os as _os
    smi = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True)
    if smi.returncode == 0:
        for pid in (p.strip() for p in smi.stdout.split()):
            if not pid.isdigit() or int(pid) == _os.getpid():
                continue
            try:
                cwd = _os.path.realpath(f"/proc/{pid}/cwd")
                cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode()
            except OSError:
                continue
            if cwd.startswith(str(ROOT)) or "sumofish" in cmd:
                busy.append(f"pid {pid}: {cmd[:90].strip()}")
    return busy


def checkpoint_sha(path: str | None) -> str:
    """sha256 of a checkpoint's bytes, or a reason it has none.

    Stockfish arms carry no checkpoint, and a missing file must be a distinct
    value rather than the empty string, or two different absences hash equal.
    Truncated to 12 hex chars, matching `release.py`.
    """
    if not path:
        return "none"
    f = Path(path)
    if not f.exists():
        return "missing"
    h = hashlib.sha256()
    with f.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def git_sha() -> str:
    """The commit, recorded as METADATA and deliberately NOT in the fingerprint.

    It is what you want when reading an old result. It is not what you want
    when deciding whether two halves of a match are the same experiment: it
    moves on a one-line edit to a notes file, and it does not move on an
    uncommitted edit to the engine.
    """
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True,
                              check=False).stdout.strip() or "?"
    except OSError:
        return "?"


def code_fingerprint() -> str:
    """A hash of the source that decides what a GAME IS.

    Corrected 2026-08-15. This used to be `git short sha + sha256(sumofish/)`,
    and it was wrong in both directions at once.

    TOO BROAD: the bare SHA meant that committing anything at all invalidated
    every in-flight match. A one-line edit to LAB-NOTES.md moves the SHA while
    the package digest is byte-identical, so a ten-hour match became
    non-resumable for a documentation commit. In practice that made the
    repository read-only to git for the length of any long run, which is not a
    rule anyone wrote down or would choose.

    TOO NARROW: `scripts/match.py` was not hashed. But match.py decides
    adjudication, drives the clock, picks the openings, writes the records and
    computes the result. It is the code that most directly determines what a
    game is, and editing it mid-match was invisible to the guard whose whole
    job is catching that. It fell through because the old reasoning was "hash
    the source that gets imported", and match.py is the entry point rather than
    an import.

    So: hash `sumofish/**/*.py` plus this file, and nothing else. The SHA rides
    alongside in config.json where it belongs.

    This changes the value for every match, so nothing recorded under the old
    scheme can be resumed. The only live partial when it landed was
    `matedist-time` at 314/600, a contended clock match already judged not
    quotable, so the real cost was zero. Taking it once beats carrying two
    schemes forever.
    """
    digest = hashlib.sha256()
    sources = sorted((ROOT / "sumofish").rglob("*.py"))
    sources.append(Path(__file__).resolve())
    for path in sources:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


# ---------------------------------------------------------------------------
# one game


class Arbiter:
    """A third party that decides adjudicated games, instead of the players.

    # Why this exists
    #
    # Adjudication was decided by the win probability of the engines UNDER TEST,
    # and it ended 22-34% of every match in this project's archive. Both arms
    # share the value net, so a position the net is jointly and wrongly confident
    # about -- a fortress, opposite-coloured bishops, a drawn rook ending --
    # adjudicated as a win for whoever was materially ahead. Re-scoring those
    # games as draws moved the exchange-rate ladder's rungs by +113 to +148 Elo.
    #
    # That is not a bias to correct with more games. It is a sensor wired inside
    # the system it measures, and the only fix is a reference that cannot share
    # the fault.
    #
    # # Fixed NODES, full strength
    #
    # Not `Skill Level`, which is full-strength Stockfish with randomised move
    # degradation: it would reward punishing random blunders, which is the exact
    # pathology PHILOSOPHY rejects when BUILDING difficulty and would be silly to
    # invite back in when measuring. Not fixed depth either, which is not
    # reproducible under CPU contention.
    #
    # A reference may be degraded along the same axis as the article's own
    # allowance -- nodes -- never in its judgement.
    #
    # # It reduces bias, it does not eliminate it
    #
    # Stockfish at a modest node budget is itself unreliable in fortresses and
    # opposite-coloured-bishop endings, which is the same class that broke
    # self-adjudication. This is a smaller, INDEPENDENT error in place of a
    # larger, correlated one. Do not report it as an elimination.
    """

    def __init__(self, path: str | None, nodes: int):
        self.nodes = nodes
        self.engine = None
        self.path = path
        self._game_token = object()
        if path:
            try:
                import chess.engine

                self.engine = chess.engine.SimpleEngine.popen_uci(path)
            except Exception as exc:
                print(f"arbiter unavailable ({exc}); adjudication disabled",
                      file=sys.stderr)
                self.engine = None

    def new_game(self) -> None:
        """Clear the arbiter's hash between games. See `Player.new_game`.

        ADDED 2026-08-11. The `Player` side of this was fixed on 08-10 and the
        arbiter, which is a second Stockfish process, was missed. It called
        `analyse()` with no `game=` token, and python-chess emits `ucinewgame`
        only when the token CHANGES (`first_game or self.game != game`), so
        `None` on every call fired it once at process start and never again.
        The adjudicator therefore accumulated a transposition table across every
        probe of every game in a match, at 200,000 nodes a probe.

        This is not the harmless half of the bug. The arbiter DECIDES THE
        RESULT, and it ended 35.0% of the 700-node anchor and 29.8% of the
        1600-node one. Same three consequences as the player side: a match was
        not reproducible from its seed, a resumed match differed from an
        uninterrupted one, and sharding was biased rather than merely different.

        And one the player side does not have. LAB-NOTES argues `scale_D`
        survives the harness fix because "a bias roughly constant across rungs
        cancels in a DIFFERENCE". This term is not constant across rungs: each
        rung is a separate process warming over a different game count and a
        different position distribution, so the differencing argument never
        covered it. `scripts/arbiter_bias.py` sizes it from the stored
        `final_fen` of every adjudicated game.
        """
        self._game_token = object()

    def agrees(self, board: chess.Board, white_winning: bool) -> bool:
        """Does the arbiter agree the game is decided in that direction?

        Returns False when it cannot tell, so an unavailable or uncertain arbiter
        means the game keeps playing rather than being adjudicated on the word of
        the engine under test. Playing on costs time; a wrong adjudication costs
        the result.
        """
        if self.engine is None:
            return False
        try:
            import chess.engine

            info = self.engine.analyse(
                board, chess.engine.Limit(nodes=self.nodes),
                # Changing this token is what makes python-chess emit
                # `ucinewgame`. See new_game().
                game=self._game_token,
            )
        except Exception:
            return False
        score = info.get("score")
        if score is None:
            return False
        # White's frame, so the two sides are symmetric.
        wp = score.white().wdl(model="sf12").expectation()
        return wp >= 0.97 if white_winning else wp <= 0.03

    def close(self) -> None:
        if self.engine is not None:
            try:
                self.engine.quit()
            except Exception:
                pass


def play_game(
    white: Player,
    black: Player,
    opening: list[str],
    max_plies: int,
    adj_wp: float,
    adj_plies: int,
    arbiter: "Arbiter | None" = None,
    arbiter_id: str | None = None,
    tc: tuple[float, float] | None = None,
) -> dict:
    """Play one game from a book position and return how it ended.

    Adjudication is a match-harness convenience, not a playing decision: when
    both engines have agreed for `adj_plies` consecutive plies that one side is
    winning by more than `adj_wp`, the remaining moves are not going to change
    the result and playing them costs GPU time that another game wants. Both
    sides have to agree, because the run of plies alternates between them.
    """
    board = chess.Board()
    for uci in opening:
        board.push(chess.Move.from_uci(uci))
    opening_plies = board.ply()

    white.new_game()
    black.new_game()
    if arbiter is not None:
        arbiter.new_game()

    # Win probability of each ply, in WHITE's frame. Converting once here is
    # the same discipline as `panels.ours()`: convert at one place or a sign
    # error is guaranteed.
    curve: list[float] = []
    started = time.perf_counter()
    clock = Clock(*tc) if tc is not None else None

    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            result, reason = outcome.result(), outcome.termination.name.lower()
            break
        if board.ply() - opening_plies >= max_plies:
            result, reason = "1/2-1/2", "move-limit"
            break
        if len(curve) >= adj_plies:
            tail = curve[-adj_plies:]
            # The engines' own curve only PROPOSES. A third party decides, and
            # if there is no third party the game plays on: adjudicating on the
            # word of the engine under test is what corrupted the archive.
            if all(p >= adj_wp for p in tail):
                if arbiter is None:
                    result, reason = "1-0", "adjudicated"
                    break
                if arbiter.agrees(board, white_winning=True):
                    result, reason = "1-0", "adjudicated-arbiter"
                    break
            if all(p <= 1.0 - adj_wp for p in tail):
                if arbiter is None:
                    result, reason = "0-1", "adjudicated"
                    break
                if arbiter.agrees(board, white_winning=False):
                    result, reason = "0-1", "adjudicated-arbiter"
                    break

        player = white if board.turn == chess.WHITE else black
        mover = board.turn
        move_started = time.perf_counter()
        move, wp = player.move(board, clock)
        if clock is not None and not clock.charge(mover, time.perf_counter() - move_started):
            # Losing on time is a real result and is recorded as one. It is also
            # a LOUD one: think_time is explicitly conservative (a thirtieth of
            # what remains, capped at a third) so a flag here means either the
            # budget rule is wrong or a move overran its deadline, and both are
            # worth failing visibly rather than absorbing into the draw rate.
            result = "0-1" if mover == chess.WHITE else "1-0"
            reason = "time-forfeit"
            break
        curve.append(wp if mover == chess.WHITE else 1.0 - wp)
        board.push(move)

    return {
        "result": result,
        "reason": reason,
        "plies": board.ply() - opening_plies,
        "seconds": round(time.perf_counter() - started, 2),
        "opening": opening,
        "moves": [m.uci() for m in board.move_stack[opening_plies:]],
        "final_fen": board.fen(),
        # The per-ply win probability, in White's frame, one entry per move
        # played after the book. It was computed and thrown away 400 times a
        # match, and it is the only raw material in this project from which
        # anything about the TEXTURE of a game can be derived: whether the
        # evaluation collapses in a single ply or slides, whether a game was
        # decided early or late, whether mistakes cluster in sharp positions
        # the way a human's do or land at random the way a handicapped engine's
        # do. Four bytes a ply. Storing it costs nothing and not storing it
        # means the question cannot be asked retrospectively.
        "curve": [round(p, 4) for p in curve],
        # Clock usage per side, seconds, under --tc only. This is the raw
        # material for pricing time management: instamove and early stopping
        # are supposed to show up here as time BANKED, and if a flag is on and
        # this does not move, the flag is not reaching the search.
        "clock": ({"white_spent": round(clock.spent[chess.WHITE], 2),
                   "black_spent": round(clock.spent[chess.BLACK], 2),
                   "white_left": round(clock.remaining[chess.WHITE], 2),
                   "black_left": round(clock.remaining[chess.BLACK], 2)}
                  if clock is not None else None),
        # How much search each side actually got. Under --time and --tc this is
        # the DEPENDENT variable -- `simulations` is pinned at 10**9 so the clock
        # binds -- and it is precisely what a dearer network loses. Recording
        # only seconds, as this file did until 2026-08-14, meant a wall-clock
        # match could not see the quantity it existed to measure: seconds/ply
        # sits pinned at the movetime by construction, so the archive's six
        # movetime arms are blind to their own confound. `unique` is the count
        # PHILOSOPHY mandates reporting; with dedup off it equals `evals` by
        # construction and is a row count, not a distinct-position count.
        "search": {"white_evals": white.evals, "black_evals": black.evals,
                   "white_unique": white.unique_evals,
                   "black_unique": black.unique_evals,
                   "white_moves": white.searches, "black_moves": black.searches},
        # What was actually playing. A match spans hours and the working tree
        # is editable throughout; without this, a match that straddles an edit
        # is indistinguishable from one that did not. Cheap insurance against
        # a silently invalidated result.
        "code": code_fingerprint(),
        # Who ended an adjudicated game. None for a natural finish, the arbiter
        # id otherwise, so a later re-scoring can tell which games were decided
        # by whom rather than having to parse `reason`.
        "adjudicated_by": arbiter_id if reason.startswith("adjudicated") else None,
    }


def to_pgn(record: dict, white_label: str, black_label: str, index: int) -> str:
    board = chess.Board()
    for uci in record["opening"] + record["moves"]:
        board.push(chess.Move.from_uci(uci))
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "SumoFish match"
    game.headers["Round"] = str(index)
    game.headers["White"] = white_label
    game.headers["Black"] = black_label
    game.headers["Result"] = record["result"]
    game.headers["Termination"] = record["reason"]
    return str(game)


# ---------------------------------------------------------------------------
# the match


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Play two SumoFish configurations against each other.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Shared defaults. Anything not given per-side falls back to these, so the
    # common case (two checkpoints, everything else identical) is two flags.
    ap.add_argument("--value", default=str(ROOT / "runs/value.pt"))
    ap.add_argument("--policy", default=str(ROOT / "runs/policy.pt"))
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--cpuct", type=float, default=2.0,
                    help="the CONSTANT exploration term. Read only under "
                         "--fixed-cpuct; under AlphaZero's schedule (the "
                         "default, and what ships) it is ignored entirely and "
                         "--cpuct-init is the knob. Passing it alone changes "
                         "nothing.")
    ap.add_argument("--cpuct-init", type=float, default=0.875,
                    help="the additive term in AlphaZero's schedule, "
                         "ln((1+N+base)/base) + INIT. This is the exploration "
                         "constant that binds in the shipped configuration. "
                         "Added 2026-08-09; before that nothing could vary it "
                         "and every match ran the hardcoded 1.25.")
    ap.add_argument("--fpu", type=float, default=-0.05)
    ap.add_argument("--time", type=float, default=None,
                    help="seconds per MOVE; overrides --sims when set. This is "
                         "not a game clock -- see --tc, which is.")
    ap.add_argument("--tc", default=None, metavar="BASE+INC",
                    help="a real game clock in seconds, e.g. '60+1'. Overrides "
                         "--sims and --time. This is the only mode that "
                         "exercises search_engine.think_time, and therefore the "
                         "only one that can price --a-instamove/--a-early-stop: "
                         "under --time each move's allowance is independent, so "
                         "time saved on one move goes nowhere, which is exactly "
                         "what early stopping exists to exploit.")

    for side in ("a", "b"):
        ap.add_argument(f"--{side}-value")
        ap.add_argument(f"--{side}-policy")
        ap.add_argument(f"--{side}-sims", type=int)
        ap.add_argument(f"--{side}-batch", type=int)
        ap.add_argument(f"--{side}-cpuct", type=float)
        ap.add_argument(f"--{side}-cpuct-init", type=float)
        ap.add_argument(f"--{side}-fpu", type=float)
        ap.add_argument(f"--{side}-time", type=float)
        ap.add_argument(f"--{side}-searchless", action="store_true")
        ap.add_argument(f"--{side}-no-reuse", action="store_true",
                        help="rebuild the tree from scratch every move")
        ap.add_argument(f"--{side}-fixed-cpuct", action="store_true",
                        help="use a constant c_puct instead of AlphaZero's "
                             "visit-count schedule")
        ap.add_argument(f"--{side}-legacy-draws", action="store_true",
                        help="the pre-rules.py terminal test: treat a draw that "
                             "is merely reachable by one move as already drawn")
        ap.add_argument(
            f"--{side}-stockfish-nodes", type=int, default=None,
            help="play this side with Stockfish at a fixed NODE budget instead "
                 "of the neural MCTS -- full strength, never Skill Level, never "
                 "fixed depth, same discipline as --arbiter-nodes. All other "
                 "--%s-* flags (value/policy/sims/...) are ignored for this "
                 "side when set." % side,
        )
        ap.add_argument(f"--{side}-label")

    ap.add_argument("--games", type=int, default=200,
                    help="rounded down to an even number; openings are paired")
    ap.add_argument("--book", default=str(ROOT / "data/eco_openings.pgn"))
    ap.add_argument("--book-min-ply", type=int, default=6)
    ap.add_argument("--book-max-ply", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--adjudicate-wp", type=float, default=0.97)
    ap.add_argument("--adjudicate-plies", type=int, default=10)
    # Resolved through `select_mcts_class`, not hardcoded, so ONE place decides
    # which core is "the" core and a typo in CHESSGPU_CORE still fails loudly
    # there rather than silently selecting the other engine.
    #
    # This default was `"python"` until 2026-07-31, and it had been wrong since
    # the Rust core went live on 2026-07-30. The cost is specific to fixed-TIME
    # matches and it is severe: measured the same day, the Python tree runs the
    # 9M at 3,041 nps against Rust's 8,135 (see runs/lab/profile-2026-07-31.json),
    # so a `--time` match on the defaults handed both arms ~2.7x less search than
    # the deployed engine gets and then reported the result as strength. Fixed-SIMS
    # matches were unaffected -- the identity proof means the two cores build the
    # same tree -- which is exactly why this survived: the failure is invisible in
    # the experiment most often run and total in the one that decides promotions.
    #
    # Third instance of the same family. `bench_search.py` imported the Python
    # MCTS outright (its `scale_m` was 1.7x too cheap as a result) and
    # `rust_mcts.select_mcts_class` itself defaulted to Python until 07-31.
    # LAB-NOTES already says "re-profile after every port"; the generalisation it
    # was missing is that a port has to sweep the DEFAULTS of every instrument
    # that consumes it, because each one keeps its own idea of what normal is.
    ap.add_argument("--core", default=select_mcts_class()[1],
                    choices=("python", "rust"),
                    help="search implementation for both sides unless overridden. "
                         "Defaults to what CHESSGPU_CORE resolves to, i.e. the core "
                         "that actually plays. Pass --core python for the oracle "
                         "comparison in tests/identity_*.py.")
    for _s in ("a", "b"):
        ap.add_argument(f"--{_s}-core", default=None, choices=("python", "rust"))
        ap.add_argument(f"--{_s}-dedup", action="store_true",
                        help="dedupe the network call for repeated leaves (rust only)")
        ap.add_argument(f"--{_s}-compile", action="store_true",
                        help="torch.compile the nets, padded to a static shape "
                             "for CUDA graphs (rust only)")
        ap.add_argument(f"--{_s}-mate-distance", action="store_true",
                        help="prefer the shortest proven mate instead of "
                             "backing up mate-in-2 and mate-in-14 identically "
                             "(rust only)")
        ap.add_argument(f"--{_s}-instamove", action="store_true",
                        help="spend 50 ms rather than the full budget when "
                             "there is only one legal move (--tc only)")
        ap.add_argument(f"--{_s}-early-stop", action="store_true",
                        help="stop once the runner-up cannot be caught even if "
                             "every remaining simulation went to it (--tc only, "
                             "rust core only). Provably cannot change which "
                             "move is played, only when.")
        ap.add_argument(f"--{_s}-vloss-fix", action="store_true",
                        help="virtual loss affects only the PUCT selection "
                             "denominator, not the backed-up value_sum "
                             "(rust only)")
    ap.add_argument("--no-adjudicate", action="store_true")
    ap.add_argument(
        "--arbiter",
        default=str(ROOT / "tools/stockfish/stockfish-ubuntu-x86-64-bmi2"),
        help="third party that must AGREE before a game is adjudicated. "
             "The engines under test share a value net, so letting them decide "
             "ended 22-34%% of every match in this project's archive on their own "
             "word. Pass --arbiter '' to disable and adjudicate as before.",
    )
    ap.add_argument(
        "--arbiter-nodes", type=int, default=200_000,
        help="fixed NODES for the arbiter. Fixed nodes rather than depth so it "
             "reproduces under CPU contention, and full strength rather than a "
             "Skill Level, because a reference may be degraded in its allowance "
             "and never in its judgement.",
    )
    ap.add_argument(
        "--stockfish-path",
        default=str(ROOT / "tools/stockfish/stockfish-ubuntu-x86-64-bmi2"),
        help="UCI binary used by --a-stockfish-nodes/--b-stockfish-nodes. "
             "Independent of --arbiter, though it defaults to the same binary; "
             "a match can use one Stockfish build to play and a different one "
             "to adjudicate.",
    )
    ap.add_argument("--elo0", type=float, default=0.0, help="SPRT null hypothesis")
    ap.add_argument("--elo1", type=float, default=20.0, help="SPRT alternative")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--allow-contended", action="store_true",
                    help="run a wall-clock match anyway while the GPU is "
                         "shared. The contention is then stamped into "
                         "config.json, so a later reader can see it. Exists so "
                         "that accepting an invalid measurement is a visible "
                         "choice; two arms in the archive made it invisibly.")
    ap.add_argument("--no-sprt", action="store_true",
                    help="play every game; do not stop early")
    ap.add_argument("--min-pairs", type=int, default=0,
                    help="do not let the SPRT stop the match below this many "
                         "pairs. The sequential test crosses its bound in ~8 "
                         "pairs on a large effect (measured: LLR grows ~0.38 "
                         "per pair on sims-1600-vs-800), but a CONSUMER of the "
                         "result may require a larger sample before it will act "
                         "-- lab.py's decide_promote wants MIN_DECISIVE_PAIRS. "
                         "Stopping below the consumer's floor produces a "
                         "correct verdict nobody is allowed to use. Default 0 "
                         "keeps standalone behaviour unchanged; the lab passes "
                         "its own floor.")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="this process plays only pairs where "
                         "pair %% SHARD_COUNT == SHARD_INDEX. With "
                         "--shard-count it splits one match across processes.")
    ap.add_argument("--shard-count", type=int, default=1,
                    help="how many processes are splitting this match. Pairs "
                         "are the unit, never games, so a pair's two "
                         "colour-swapped halves always land in one process and "
                         "the pair statistics stay intact. Openings are chosen "
                         "from the same --seed in every shard, so pair N is the "
                         "SAME opening in every shard and the union is exactly "
                         "the match a single process would have played. "
                         "Merge with scripts/merge_matches.py.")
    ap.add_argument("--name", default=None, help="output directory under runs/matches")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.shard_count < 1 or not (0 <= args.shard_index < args.shard_count):
        raise SystemExit(
            f"bad shard {args.shard_index}/{args.shard_count}: need "
            f"shard_count >= 1 and 0 <= shard_index < shard_count"
        )
    if (args.time or args.tc) and not args.allow_contended:
        # PHILOSOPHY: "A wall-clock match requires an idle machine. Contention
        # biases a time-budgeted experiment and nothing else, so it is the one
        # experiment where 'the bot was also running' invalidates the result."
        # That rule was stated and never enforced, and `scripts/lab.py`'s own
        # `wait_for_quiet` docstring says a --time match "must not share the
        # card" while its code does not implement it. Both clock arms in the
        # 2026-08-13 flag queue ran with the bot up, and because seconds/ply is
        # pinned at the movetime by construction, nothing in games.jsonl could
        # show it afterwards. Enforced here instead of trusted.
        busy = contending_units()
        if busy:
            raise SystemExit(
                "refusing to start a wall-clock match while the GPU is shared:\n"
                + "".join(f"  {u}\n" for u in busy)
                + "  wrap it:  scripts/gpu_lock.py run --drain-bot -- "
                  "scripts/match.py ...\n"
                  "  or pass --allow-contended to record the contention and "
                  "proceed anyway."
            )

    if args.shard_count > 1 and not args.no_sprt:
        # A shard sees a biased subset (every Nth pair) and cannot run the
        # sequential test on it: stopping on a shard's own LLR would stop the
        # whole match on a fraction of the evidence.
        raise SystemExit("--shard-count > 1 requires --no-sprt; "
                         "run the SPRT over the merged result instead")

    def spec(side: str) -> Spec:
        def pick(field: str, shared: str | None = None):
            v = getattr(args, f"{side}_{field}")
            return v if v is not None else getattr(args, shared or field)

        return Spec(
            label=getattr(args, f"{side}_label") or side.upper(),
            value=pick("value"),
            policy=pick("policy"),
            value_sha=checkpoint_sha(pick("value")),
            policy_sha=checkpoint_sha(pick("policy")),
            sims=pick("sims"),
            batch=pick("batch"),
            c_puct=pick("cpuct"),
            c_puct_init=pick("cpuct_init"),
            fpu=pick("fpu"),
            # --time is shared and legitimately None, so it cannot use `pick`:
            # None means "use sims", not "fall through to the shared default".
            movetime=getattr(args, f"{side}_time") or args.time,
            searchless=getattr(args, f"{side}_searchless"),
            reuse=not getattr(args, f"{side}_no_reuse"),
            legacy_draws=getattr(args, f"{side}_legacy_draws"),
            fixed_cpuct=getattr(args, f"{side}_fixed_cpuct"),
            core=pick("core"),
            dedup=getattr(args, f"{side}_dedup"),
            compile_nets=getattr(args, f"{side}_compile"),
            mate_distance=getattr(args, f"{side}_mate_distance"),
            vloss_fix=getattr(args, f"{side}_vloss_fix"),
            instamove=getattr(args, f"{side}_instamove"),
            early_stop=getattr(args, f"{side}_early_stop"),
            clocked=bool(args.tc),
            stockfish_nodes=getattr(args, f"{side}_stockfish_nodes"),
        )

    a, b = spec("a"), spec("b")

    tc = None
    if args.tc:
        try:
            base_s, inc_s = (float(x) for x in args.tc.replace("/", "+").split("+"))
        except ValueError:
            print(f"--tc must be BASE+INC in seconds, e.g. '60+1'; got {args.tc!r}",
                  file=sys.stderr)
            return 2
        if base_s <= 0 or inc_s < 0:
            print(f"--tc needs a positive base and a non-negative increment; "
                  f"got {base_s}+{inc_s}", file=sys.stderr)
            return 2
        tc = (base_s, inc_s)
        print(f"time control: {base_s:g}+{inc_s:g} (a real clock; "
              f"think_time is in the loop)")

    # A flag that cannot reach the search is worse than one that is off, because
    # it produces a null result that reads as "the feature does not help". Both
    # of these are silently inert without --tc, and early stopping additionally
    # needs the Rust core, whose search() is the only one with a should_stop
    # parameter. Refuse rather than run a match that cannot measure its subject.
    for side, spec_ in (("a", a), ("b", b)):
        if (spec_.instamove or spec_.early_stop) and tc is None:
            print(f"--{side}-instamove/--{side}-early-stop do nothing without "
                  f"--tc: without a game clock `think_time` is never called and "
                  f"banked time has nowhere to go. Refusing to run a match that "
                  f"cannot measure what it was asked to measure.", file=sys.stderr)
            return 2
        if spec_.early_stop and spec_.core != "rust":
            print(f"--{side}-early-stop needs the rust core; sumofish.mcts."
                  f"search has no should_stop parameter.", file=sys.stderr)
            return 2

    if args.no_adjudicate:
        args.adjudicate_wp, args.adjudicate_plies = 2.0, 10**9

    name = args.name or time.strftime("%Y%m%d-%H%M%S")
    outdir = ROOT / "runs" / "matches" / name
    outdir.mkdir(parents=True, exist_ok=True)
    log_path, pgn_path = outdir / "games.jsonl", outdir / "games.pgn"

    # ---- the resume fingerprint ----
    #
    # On 2026-07-29 all four rungs of the exchange-rate ladder were found to be
    # REPLAYS. Resume keyed on `rec["game"]` alone, so a job with different code,
    # a different checkpoint and a different budget landed on an existing
    # directory, skipped every game as "already played", and reported the old
    # numbers as its own -- in 5 seconds, against hours of logged play. Worse,
    # `config.json` was then rewritten with the NEW spec over the OLD games, so
    # the directory actively asserted a provenance it never had.
    #
    # The fix has two halves. This one refuses to resume across a spec change.
    # The other is `scripts/verify_replays.py`, which finds the damage already
    # done via `sum(game.seconds) <= job.seconds` -- an inequality that cannot
    # be violated legitimately.
    #
    # Deliberately excluded from the hash: `games` and `name`, so extending a
    # match from 300 to 400 games still resumes, which is the one case resume is
    # actually for. Everything that changes what a GAME is, is included.
    fp_args = {
        k: v for k, v in vars(args).items()
        if k not in ("games", "name", "device", "quiet")
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            {"a": a.__dict__, "b": b.__dict__, "args": fp_args,
             "code": code_fingerprint()},
            sort_keys=True, default=str,
        ).encode()
    ).hexdigest()[:16]

    cfg_path = outdir / "config.json"
    # Keyed on the GAMES existing, not on the config existing. A directory with a
    # log and no config is the unprovenanced case, and requiring the config to be
    # present in order to check it would wave through exactly the state this is
    # meant to catch.
    if log_path.exists() and log_path.stat().st_size > 0:
        prior = None
        if cfg_path.exists():
            try:
                prior = json.loads(cfg_path.read_text()).get("fingerprint")
            except Exception:
                prior = None
        if prior is None:
            print(
                f"REFUSING to resume {outdir}: it has games but no fingerprint, so\n"
                f"it predates this check and its provenance cannot be established.\n"
                f"Run `scripts/verify_replays.py` to audit it, then use a new --name.",
                file=sys.stderr,
            )
            return 2
        if prior != fingerprint:
            # Say WHICH field moved. A bare "the fingerprint differs" on a
            # swapped checkpoint reads as a harness bug, and the whole point of
            # hashing content is that this case is now diagnosable.
            diff = []
            try:
                pc = json.loads(cfg_path.read_text())
                for side, now in (("a", a), ("b", b)):
                    was = pc.get(side) or {}
                    for k, v in now.__dict__.items():
                        if k in was and was[k] != v:
                            diff.append(f"    {side}.{k}: {was[k]!r} -> {v!r}")
            except Exception:                                # noqa: BLE001
                pass
            what = ("  what moved:\n" + "\n".join(diff) + "\n") if diff else (
                "  the difference is in the args or the code fingerprint, not in\n"
                "  either arm's spec.\n")
            print(
                f"REFUSING to resume {outdir}: spec fingerprint differs.\n"
                f"  on disk: {prior}\n"
                f"  now:     {fingerprint}\n"
                f"{what}"
                f"Those games were played by a different configuration. Resuming\n"
                f"would report them as this one's -- which is how the exchange-rate\n"
                f"ladder came to be four replays. Use a new --name.",
                file=sys.stderr,
            )
            return 2

    cfg_path.write_text(
        json.dumps(
            {"fingerprint": fingerprint, "code": code_fingerprint(),
             # Metadata, not identity. See code_fingerprint's docstring.
             "git_sha": git_sha(),
             # Machine state at launch. Absent from every archived match, which
             # is why the 2026-08-13 clock arms cannot be audited for
             # contention after the fact: seconds/ply is pinned at the movetime,
             # so wall clock cannot see it, and sims were not recorded either.
             "machine": {"contending": contending_units(),
                         "clock_match": bool(args.time or args.tc),
                         "at": __import__("time").time()},
             "a": a.__dict__, "b": b.__dict__, "args": vars(args)},
            indent=2, default=str,
        )
    )

    print(f"A: {a.label}  {a.describe()}")
    print(f"B: {b.label}  {b.describe()}")
    print(f"log: {log_path}\n")

    openings = load_openings(
        Path(args.book), args.book_min_ply, args.book_max_ply, args.seed
    )
    pairs = args.games // 2
    if pairs > len(openings):
        print(f"book has {len(openings)} distinct lines; capping at "
              f"{len(openings) * 2} games")
        pairs = len(openings)

    # Resume. Anything already in the log counts and is not replayed.
    done: dict[int, dict] = {}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["game"]] = rec
        if done:
            print(f"resuming: {len(done)} games already played\n")

    arbiter_path = args.arbiter or None
    if arbiter_path and not Path(arbiter_path).exists():
        print(f"arbiter not found at {arbiter_path}; adjudication will be "
              f"disabled, so decided games play to a natural finish",
              file=sys.stderr)
        arbiter_path = None
    arbiter = Arbiter(arbiter_path, args.arbiter_nodes) if arbiter_path else None
    arbiter_id = (
        f"stockfish@{args.arbiter_nodes}nodes" if arbiter and arbiter.engine else None
    )
    if arbiter is not None and arbiter.engine is None:
        arbiter = None
    print(f"arbiter: {arbiter_id or 'none (games play to a natural finish)'}")

    players = {
        "a": Player(a, args.device, stockfish_path=args.stockfish_path),
        "b": Player(b, args.device, stockfish_path=args.stockfish_path),
    }
    # Warm the kernels before the first timed move, so a --time or --tc match
    # does not charge one side for CUDA's first-call latency.
    for p in players.values():
        p.warmup()

    # Every record, because the pair statistics need the colour-swapped partner
    # and not just a running W/D/L.
    records: list[dict] = list(done.values())
    lower, upper = sprt_bounds(args.alpha, args.beta)
    log_fh = log_path.open("a")
    pgn_fh = pgn_path.open("a")

    try:
        for pair in range(pairs):
            # Sharding. The unit is the PAIR: splitting a pair across processes
            # would put its two colour-swapped halves in different logs, and
            # every statistic here is computed over pairs.
            if args.shard_count > 1 and pair % args.shard_count != args.shard_index:
                continue
            for game_in_pair in range(2):
                index = pair * 2 + game_in_pair
                if index in done:
                    continue
                # Colours swap within the pair. A plays White on the even game.
                a_is_white = game_in_pair == 0
                white = players["a" if a_is_white else "b"]
                black = players["b" if a_is_white else "a"]

                rec = play_game(
                    white, black, openings[pair],
                    args.max_plies, args.adjudicate_wp, args.adjudicate_plies,
                    arbiter, arbiter_id, tc,
                )
                # Score from A's point of view, which is what everything below
                # counts. This is the one place the colour swap is undone.
                if rec["result"] == "1/2-1/2":
                    score = 0.5
                elif rec["result"] == "1-0":
                    score = 1.0 if a_is_white else 0.0
                else:
                    score = 0.0 if a_is_white else 1.0

                rec |= {"game": index, "a_white": a_is_white, "score": score}
                log_fh.write(json.dumps(rec) + "\n")
                log_fh.flush()
                pgn_fh.write(
                    to_pgn(rec, white.spec.label, black.spec.label, index) + "\n\n"
                )
                pgn_fh.flush()

                records.append(rec)
                w, d, l = tally(records)
                sums = pair_sums(records)
                st = pair_stats(sums)
                llr = sprt_llr_pairs(sums, args.elo0, args.elo1)
                print(
                    f"[{len(records):4d}|{st['pairs']:3d}pr] W{w} D{d} L{l}  "
                    f"score {st['score'] * 100:5.1f}%  "
                    f"elo {st['elo']:+7.1f} +-{st['err']:5.1f}  "
                    f"LOS {st['los'] * 100:5.1f}%  LLR {llr:+5.2f}  "
                    f"({rec['result']} {rec['reason']} {rec['plies']}p "
                    f"{rec['seconds']}s)",
                    flush=True,
                )

                # A live SPRT status file, separate from `games.jsonl` and the
                # scrolling log, so a match that will run for hours can be
                # checked without tailing a log or waiting for the final
                # summary. Written every game, atomically (write-then-rename),
                # so a reader never sees a half-written file.
                status = {
                    "games": len(records),
                    "pairs": st["pairs"],
                    "w": w, "d": d, "l": l,
                    "score": st["score"],
                    "elo": st["elo"],
                    "err": st["err"],
                    "los": st["los"],
                    "llr": llr,
                    "bounds": {"lower": lower, "upper": upper},
                    "sprt": {"elo0": args.elo0, "elo1": args.elo1,
                             "alpha": args.alpha, "beta": args.beta},
                    "concluded": None,
                    "last_game": {"result": rec["result"], "reason": rec["reason"],
                                   "plies": rec["plies"], "seconds": rec["seconds"]},
                    "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                tmp_path = outdir / "status.json.tmp"
                tmp_path.write_text(json.dumps(status, indent=2))
                tmp_path.replace(outdir / "status.json")

                # Only ever at a pair boundary. Stopping mid-pair leaves the
                # match one game long in one colour, and since the stop is
                # triggered by a game that MOVED the statistic, that unpaired
                # game is systematically A's -- a bias built into the stopping
                # rule itself.
                if (not args.no_sprt and game_in_pair == 1
                        and st["pairs"] >= args.min_pairs
                        and (llr >= upper or llr <= lower)):
                    verdict = "A is better" if llr >= upper else "A is not better"
                    print(f"\nSPRT concluded after {st['pairs']} pairs "
                          f"({len(records)} games): {verdict} "
                          f"(LLR {llr:+.2f}, bounds {lower:+.2f} / {upper:+.2f})")
                    status["concluded"] = verdict
                    tmp_path.write_text(json.dumps(status, indent=2))
                    tmp_path.replace(outdir / "status.json")
                    return
    except KeyboardInterrupt:
        print("\ninterrupted; rerun the same command to continue")
    finally:
        log_fh.close()
        pgn_fh.close()
        # The arbiter is a subprocess; a match that ends by SPRT, by Ctrl-C or by
        # exception must not leave a Stockfish behind holding a core. A
        # Stockfish-backed player is exactly the same kind of subprocess.
        if arbiter is not None:
            arbiter.close()
        for p in players.values():
            p.close()

    w, d, l = tally(records)
    sums = pair_sums(records)
    st, by_game = pair_stats(sums), score_stats(w, d, l)
    r = pairing_efficiency(sums, w, d, l)
    print(
        f"\n{len(records)} games in {st['pairs']} complete pairs  W{w} D{d} L{l}\n"
        f"by pair   score {st['score'] * 100:.1f}%   "
        f"elo {st['elo']:+.1f} +-{st['err']:.1f} (95%)   LOS {st['los'] * 100:.1f}%\n"
        f"by game   elo {by_game['elo']:+.1f} +-{by_game['err']:.1f}   "
        f"LOS {by_game['los'] * 100:.1f}%\n"
        # r is the whole reason the two lines differ, and it is a property of
        # THIS match rather than a constant to be carried anywhere else.
        f"pairing   r = {r:.3f}, so counting pairs is worth {1 / r:.2f}x the games"
    )
    if st["err"] > abs(st["elo"]):
        print("The interval includes zero. This match has not shown a difference.")

    # Final status write. Distinguishes "budget exhausted without an SPRT
    # decision" from an interrupted run, for anyone checking status.json
    # instead of re-deriving state from games.jsonl.
    status_path = outdir / "status.json"
    try:
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
    except Exception:
        status = {}
    status.update({
        "games": len(records), "pairs": st["pairs"], "w": w, "d": d, "l": l,
        "score": st["score"], "elo": st["elo"], "err": st["err"], "los": st["los"],
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    if status.get("concluded") is None:
        status["concluded"] = "exhausted (games budget reached without an SPRT decision)"
    status_path.write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    # sys.exit(main()), NOT main(). The refusal paths above return a non-zero
    # code, and `lab.py` gates on `exit 0`. Calling main() bare discards the
    # return value, so a match that REFUSED to run would report success to the
    # automation and the job would be marked completed. Caught by deliberately
    # inducing the refusal and checking $?, which is the only way this class of
    # bug is ever found.
    sys.exit(main())
