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

    def describe(self) -> str:
        if self.searchless:
            return f"value={self.value} searchless"
        budget = f"{self.movetime}s" if self.movetime else f"{self.sims} sims"
        return (
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
        else:
            self.mcts = MCTS(
                self.value,
                policy=load_prior(spec.policy, device),
                c_puct=spec.c_puct,
                fpu=spec.fpu,
                simulations=spec.sims,
                batch=spec.batch,
                reuse=spec.reuse,
                terminal=terminal_value_legacy if spec.legacy_draws else terminal_value,
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


def play_game(
    white: Player,
    black: Player,
    opening: list[str],
    max_plies: int,
    adj_wp: float,
    adj_plies: int,
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
            if all(p >= adj_wp for p in tail):
                result, reason = "1-0", "adjudicated"
                break
            if all(p <= 1.0 - adj_wp for p in tail):
                result, reason = "0-1", "adjudicated"
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
    ap.add_argument("--no-adjudicate", action="store_true")
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
        )

    a, b = spec("a"), spec("b")
    if args.no_adjudicate:
        args.adjudicate_wp, args.adjudicate_plies = 2.0, 10**9

    name = args.name or time.strftime("%Y%m%d-%H%M%S")
    outdir = ROOT / "runs" / "matches" / name
    outdir.mkdir(parents=True, exist_ok=True)
    log_path, pgn_path = outdir / "games.jsonl", outdir / "games.pgn"
    (outdir / "config.json").write_text(
        json.dumps({"a": a.__dict__, "b": b.__dict__, "args": vars(args)}, indent=2)
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
    main()
