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

from chessgpu.engines.neural_engine import load_policy  # noqa: E402
from chessgpu.hlgauss import HLGauss  # noqa: E402
from chessgpu.mcts import MCTS  # noqa: E402
from chessgpu.model import ChessTransformer, ModelConfig  # noqa: E402
from chessgpu.rules import terminal_value, terminal_value_legacy  # noqa: E402
from chessgpu.value_policy import ValuePolicy  # noqa: E402

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


@dataclass
class Spec:
    """One side of the match, fully described."""

    label: str
    value: str
    policy: str
    sims: int
    batch: int
    c_puct: float
    fpu: float
    movetime: float | None
    searchless: bool
    reuse: bool
    legacy_draws: bool
    fixed_cpuct: bool
    # Which search implementation. "python" is chessgpu.mcts; "rust" is
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

    def describe(self) -> str:
        if self.searchless:
            return f"value={self.value} searchless"
        budget = f"{self.movetime}s" if self.movetime else f"{self.sims} sims"
        flags = "".join(
            c for c, on in (
                ("D", self.dedup),
                ("C", self.compile_nets),
                ("M", self.mate_distance),
                ("V", self.vloss_fix),
            ) if on
        )
        return (
            f"core={self.core}{'+' + flags if flags else ''} "
            f"value={self.value} policy={self.policy} {budget} "
            f"batch={self.batch} cpuct={self.c_puct} fpu={self.fpu} "
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
    """

    def __init__(self, spec: Spec, device: str = "cuda:0") -> None:
        self.spec = spec
        self.value = load_value(spec.value, device)
        if spec.searchless:
            self.mcts = None
        elif spec.core == "rust":
            from chessgpu.rust_mcts import RustMCTS

            self.mcts = RustMCTS(
                self.value,
                policy=load_prior(spec.policy, device),
                c_puct=spec.c_puct,
                fpu=spec.fpu,
                simulations=10**9 if spec.movetime else spec.sims,
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
                fpu=spec.fpu,
                # `--time` means the CLOCK decides, so the simulation count
                # must not also bind. It did: `simulations=spec.sims` with a
                # 400 default meant `--time 3.0` ran min(400 sims, 3s) ~= 0.12s,
                # and the deadline never applied. Both matches feeding the lab's
                # promotion gate were equal-simulation matches while the gate
                # applied an equal-TIME bar. `smoke.py` and `bench_search.py`
                # already write 10**9 for exactly this reason; this was the one
                # place that forgot.
                simulations=10**9 if spec.movetime else spec.sims,
                batch=spec.batch,
                reuse=spec.reuse,
                terminal=terminal_value_legacy if spec.legacy_draws else terminal_value,
                # None restores the pre-schedule constant c_puct.
                c_puct_base=None if spec.fixed_cpuct else 19652.0,
            )

    def new_game(self) -> None:
        # Tree reuse, once it exists, must not carry a subtree from the
        # previous game into this one.
        reset = getattr(self.mcts, "reset", None)
        if reset is not None:
            reset()

    def move(self, board: chess.Board) -> tuple[chess.Move, float]:
        if self.mcts is None:
            ranked = self.value.rank_moves(board)
            return ranked[0][0], ranked[0][1]
        deadline = (
            time.perf_counter() + self.spec.movetime if self.spec.movetime else None
        )
        root, visits = self.mcts.search(board, deadline=deadline)
        move = max(visits.items(), key=lambda kv: kv[1])[0]
        return move, root.q


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
def code_fingerprint() -> str:
    """git SHA plus a hash of the package, so a stale result can be spotted.

    The SHA alone is not enough: this project's own Lab Notes record that
    editing `chessgpu/` mid-match makes the first half of the games play a
    different engine than the second, and an uncommitted edit does not move the
    SHA. Hashing the source that actually gets imported does.
    """
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        sha = "?"
    digest = hashlib.sha256()
    for path in sorted((ROOT / "chessgpu").rglob("*.py")):
        digest.update(path.read_bytes())
    return f"{sha or '?'}+{digest.hexdigest()[:12]}"


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
        if path:
            try:
                import chess.engine

                self.engine = chess.engine.SimpleEngine.popen_uci(path)
            except Exception as exc:
                print(f"arbiter unavailable ({exc}); adjudication disabled",
                      file=sys.stderr)
                self.engine = None

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
                board, chess.engine.Limit(nodes=self.nodes)
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

    # Win probability of each ply, in WHITE's frame. Converting once here is
    # the same discipline as `panels.ours()`: convert at one place or a sign
    # error is guaranteed.
    curve: list[float] = []
    started = time.perf_counter()

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
        move, wp = player.move(board)
        curve.append(wp if board.turn == chess.WHITE else 1.0 - wp)
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
    ap.add_argument("--cpuct", type=float, default=2.0)
    ap.add_argument("--fpu", type=float, default=-0.2)
    ap.add_argument("--time", type=float, default=None,
                    help="seconds per move; overrides --sims when set")

    for side in ("a", "b"):
        ap.add_argument(f"--{side}-value")
        ap.add_argument(f"--{side}-policy")
        ap.add_argument(f"--{side}-sims", type=int)
        ap.add_argument(f"--{side}-batch", type=int)
        ap.add_argument(f"--{side}-cpuct", type=float)
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
    ap.add_argument("--core", default="python", choices=("python", "rust"),
                    help="search implementation for both sides unless overridden")
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
    ap.add_argument("--elo0", type=float, default=0.0, help="SPRT null hypothesis")
    ap.add_argument("--elo1", type=float, default=20.0, help="SPRT alternative")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--no-sprt", action="store_true",
                    help="play every game; do not stop early")
    ap.add_argument("--name", default=None, help="output directory under runs/matches")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    def spec(side: str) -> Spec:
        def pick(field: str, shared: str | None = None):
            v = getattr(args, f"{side}_{field}")
            return v if v is not None else getattr(args, shared or field)

        return Spec(
            label=getattr(args, f"{side}_label") or side.upper(),
            value=pick("value"),
            policy=pick("policy"),
            sims=pick("sims"),
            batch=pick("batch"),
            c_puct=pick("cpuct"),
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
        )

    a, b = spec("a"), spec("b")
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
            print(
                f"REFUSING to resume {outdir}: spec fingerprint differs.\n"
                f"  on disk: {prior}\n"
                f"  now:     {fingerprint}\n"
                f"Those games were played by a different configuration. Resuming\n"
                f"would report them as this one's -- which is how the exchange-rate\n"
                f"ladder came to be four replays. Use a new --name.",
                file=sys.stderr,
            )
            return 2

    cfg_path.write_text(
        json.dumps(
            {"fingerprint": fingerprint, "code": code_fingerprint(),
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

    players = {"a": Player(a, args.device), "b": Player(b, args.device)}
    # Warm the kernels before the first timed move, so a --time match does not
    # charge one side for CUDA's first-call latency.
    for p in players.values():
        p.move(chess.Board())

    # Every record, because the pair statistics need the colour-swapped partner
    # and not just a running W/D/L.
    records: list[dict] = list(done.values())
    lower, upper = sprt_bounds(args.alpha, args.beta)
    log_fh = log_path.open("a")
    pgn_fh = pgn_path.open("a")

    try:
        for pair in range(pairs):
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
                    arbiter, arbiter_id,
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

                # Only ever at a pair boundary. Stopping mid-pair leaves the
                # match one game long in one colour, and since the stop is
                # triggered by a game that MOVED the statistic, that unpaired
                # game is systematically A's -- a bias built into the stopping
                # rule itself.
                if not args.no_sprt and game_in_pair == 1 and (llr >= upper or llr <= lower):
                    verdict = "A is better" if llr >= upper else "A is not better"
                    print(f"\nSPRT concluded after {st['pairs']} pairs "
                          f"({len(records)} games): {verdict} "
                          f"(LLR {llr:+.2f}, bounds {lower:+.2f} / {upper:+.2f})")
                    return
    except KeyboardInterrupt:
        print("\ninterrupted; rerun the same command to continue")
    finally:
        log_fh.close()
        pgn_fh.close()
        # The arbiter is a subprocess; a match that ends by SPRT, by Ctrl-C or by
        # exception must not leave a Stockfish behind holding a core.
        if arbiter is not None:
            arbiter.close()

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


if __name__ == "__main__":
    # sys.exit(main()), NOT main(). The refusal paths above return a non-zero
    # code, and `lab.py` gates on `exit 0`. Calling main() bare discards the
    # return value, so a match that REFUSED to run would report success to the
    # automation and the job would be marked completed. Caught by deliberately
    # inducing the refusal and checking $?, which is the only way this class of
    # bug is ever found.
    sys.exit(main())
