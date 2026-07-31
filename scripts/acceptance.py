#!/usr/bin/env python
"""The acceptance test. The only instrument here that measures goal 2.

    scripts/acceptance.py            play the next game
    scripts/acceptance.py status     how far through you are
    scripts/acceptance.py reveal     unblind, once you have finished

Everything else in this project is a *difference* test: two configurations of
the same engine play each other a few hundred times and the harness reports
which is stronger. That is a well-built instrument and it answers a question
PHILOSOPHY.md ranks THIRD. The document's second goal is that the engine is fun
to play against, it says in plain terms that fun and strong are different
targets and that pursuing the first does not deliver the second, and nothing in
this repository has ever measured it.

The reason it has never been measured is that the panel is the product tasting
itself. A mirror match is structurally blind to anything both arms share, and
no statistic over wins, draws and losses has a field for enjoyment. The only
instrument for goal 2 is a person, and for this project the population is
exactly one person, which makes n=1 the correct sample rather than a compromise.

## The protocol

Twenty games, ten against each arm, in pairs. Within each pair the order is
shuffled and your colour alternates. After every game you answer one question
on a five-point scale:

    Would you sit down with that opponent again?

That is deliberately not "was it strong", "did you win", or "was it accurate".
It is the acceptance question, and it is the one the charter cares about.

## Blinding, which is the part that is easy to get wrong

The assignment of arms to A and B is drawn from `os.urandom` at session start,
written into the session file, and never printed until `reveal`. Two leaks are
closed on purpose:

**Nothing on screen names an arm.** The header says "Opponent" and the move
list says nothing about the search.

**Both arms take the same time to move.** This is the leak that would have
sunk the whole thing. If one arm thinks for 0.4s and the other for 1.6s, you
learn which is which on move two and every rating afterwards is contaminated.
So each reply is padded to `RESPONSE_SECONDS` of wall clock regardless of what
the search actually cost. A sensory panel would call this serving both samples
at the same temperature, and would refuse to run the test without it.

You will be tempted to look in the session file. Don't.

## Reading the result

`reveal` prints the per-pair differences and a paired interval. Twenty games in
ten pairs has roughly 80% power to detect a one-point difference on a five-point
scale, which is a large effect -- this test can only see a difference you would
have noticed anyway, and its value is that it makes you notice it on purpose
rather than after eighteen months of not opening the repo.

If it comes out flat, that is a real finding: it means the thing the rest of
the roadmap optimises is not the thing that decides whether this is worth
playing, and the roadmap should say so out loud.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import chess
import chess.pgn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sumofish.mcts import MCTS  # noqa: E402
from match import load_prior, load_value  # noqa: E402

SESSIONS = ROOT / "runs" / "acceptance"

# Every reply is padded to this. See the blinding note above: an arm that
# answers faster than the other is an unblinded arm.
RESPONSE_SECONDS = 2.5

PIECES = {"P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
          "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚"}

QUESTION = "Would you sit down with that opponent again?"
SCALE = {
    "1": "no, that was unpleasant",
    "2": "not really",
    "3": "it was fine",
    "4": "yes, that was good",
    "5": "yes, immediately",
}


# ---------------------------------------------------------------------------
# board


def render(board: chess.Board, flip: bool) -> str:
    ranks = range(7, -1, -1) if not flip else range(8)
    files = range(8) if not flip else range(7, -1, -1)
    out = []
    for rank in ranks:
        row = [f" {rank + 1} "]
        for file in files:
            piece = board.piece_at(chess.square(file, rank))
            row.append(f" {PIECES[piece.symbol()] if piece else '·'} ")
        out.append("".join(row))
    letters = "abcdefgh" if not flip else "hgfedcba"
    out.append("   " + "".join(f" {c} " for c in letters))
    if board.move_stack:
        out.append("")
        out.append(f"   last: {board.peek().uci()}"
                   + ("   CHECK" if board.is_check() else ""))
    return "\n".join(out)


def move_list(board: chess.Board) -> str:
    """SAN, two columns per move number, wrapped."""
    replay, sans = chess.Board(), []
    for mv in board.move_stack:
        sans.append(replay.san(mv))
        replay.push(mv)
    out, line = [], []
    for i in range(0, len(sans), 2):
        line.append(f"{i // 2 + 1}.{sans[i]}" + (f" {sans[i + 1]}" if i + 1 < len(sans) else ""))
        if len(line) == 6:
            out.append("  ".join(line))
            line = []
    if line:
        out.append("  ".join(line))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# session


def new_session(name: str, games: int, arms: dict) -> dict:
    """Draw the blinding, once, from the OS.

    Both the assignment and the per-pair order are drawn here rather than at
    play time, so that a resumed session cannot re-roll into a different design
    and so the whole schedule is fixed before a single rating exists.
    """
    seed = int.from_bytes(os.urandom(8), "big")
    rng = random.Random(seed)
    pairs = games // 2
    schedule = []
    for pair in range(pairs):
        order = ["a", "b"]
        rng.shuffle(order)
        for slot, arm in enumerate(order):
            schedule.append({
                "game": len(schedule),
                "pair": pair,
                "arm": arm,
                # Colour alternates by pair so each arm is met from both sides.
                "human_white": (pair + slot) % 2 == 0,
            })
    return {"name": name, "seed": seed, "arms": arms, "schedule": schedule,
            "results": [], "started": time.time()}


def load(name: str) -> tuple[dict, Path]:
    path = SESSIONS / name / "session.json"
    if not path.exists():
        print(f"no session at {path}. Start one with --games N.", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text()), path


def save(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, path)


def build_engine(arm: dict):
    value = load_value(arm["value"])
    mcts = MCTS(value, policy=load_prior(arm["policy"]),
                simulations=arm["sims"], batch=arm.get("batch", 64),
                c_puct=arm.get("c_puct", 2.0), fpu=arm.get("fpu", -0.2))
    return mcts


# ---------------------------------------------------------------------------
# playing one game


def ask_move(board: chess.Board, flip: bool) -> chess.Move | None:
    """A legal move, or None if the human quits or resigns."""
    while True:
        try:
            raw = input("your move (SAN or UCI, or resign / board / quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not raw:
            continue
        low = raw.lower()
        if low in ("quit", "q"):
            return None
        if low == "resign":
            return "resign"          # type: ignore[return-value]
        if low == "board":
            print("\n" + render(board, flip) + "\n")
            continue
        for parse in (board.parse_san, board.parse_uci):
            try:
                return parse(raw)
            except (ValueError, chess.InvalidMoveError,
                    chess.IllegalMoveError, chess.AmbiguousMoveError):
                continue
        legal = ", ".join(board.san(m) for m in list(board.legal_moves)[:12])
        print(f"  not legal here. some legal moves: {legal} ...")


def play_game(entry: dict, arms: dict, index: int, total: int) -> dict | None:
    mcts = build_engine(arms[entry["arm"]])
    mcts.reset()
    board = chess.Board()
    human = chess.WHITE if entry["human_white"] else chess.BLACK
    flip = human == chess.BLACK

    print("\n" + "=" * 52)
    print(f"  game {index + 1} of {total}   you play "
          f"{'White' if human == chess.WHITE else 'Black'}")
    print("  the opponent is one of two configurations, hidden until the end")
    print("=" * 52)

    while board.outcome(claim_draw=True) is None:
        print("\n" + render(board, flip))
        if board.turn == human:
            move = ask_move(board, flip)
            if move is None:
                print("  session saved; rerun to continue this game from the start")
                return None
            if move == "resign":
                result = "0-1" if human == chess.WHITE else "1-0"
                break
            board.push(move)
        else:
            print("\n  thinking ...", flush=True)
            started = time.perf_counter()
            _root, visits = mcts.search(board)
            reply = max(visits.items(), key=lambda kv: kv[1])[0]
            # The padding that keeps the arms indistinguishable. Never remove
            # it to "make the test quicker": an arm that answers faster than
            # the other is an unblinded arm, and every rating after the second
            # move would be measuring the clock.
            remaining = RESPONSE_SECONDS - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)
            print(f"  opponent plays {board.san(reply)}")
            board.push(reply)
    else:
        result = board.outcome(claim_draw=True).result()

    print("\n" + render(board, flip))
    print("\n" + move_list(board))
    verdict = {"1-0": "White wins", "0-1": "Black wins", "1/2-1/2": "draw"}[result]
    won = (result == "1-0") == (human == chess.WHITE) and result != "1/2-1/2"
    print(f"\n  {verdict}  ({'you won' if won else 'you did not win'})")

    print(f"\n  {QUESTION}")
    for k in sorted(SCALE):
        print(f"    {k}  {SCALE[k]}")
    while True:
        try:
            rating = input("  rating 1-5: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  a game without a rating is not data; discarding it")
            return None
        if rating in SCALE:
            break
    note = input("  one line on why (optional): ").strip()

    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "SumoFish acceptance test"
    game.headers["White"] = "human" if human == chess.WHITE else "opponent"
    game.headers["Black"] = "opponent" if human == chess.WHITE else "human"
    game.headers["Result"] = result

    return {**entry, "rating": int(rating), "note": note, "result": result,
            "human_won": won, "plies": board.ply(), "pgn": str(game),
            "finished": time.time()}


# ---------------------------------------------------------------------------
# the result


def analyse(state: dict) -> dict:
    by_pair: dict[int, dict[str, int]] = {}
    for r in state["results"]:
        by_pair.setdefault(r["pair"], {})[r["arm"]] = r["rating"]
    diffs = [p["a"] - p["b"] for p in by_pair.values() if "a" in p and "b" in p]
    means = {arm: statistics.mean([r["rating"] for r in state["results"] if r["arm"] == arm])
             for arm in ("a", "b")
             if any(r["arm"] == arm for r in state["results"])}
    out = {"pairs": len(diffs), "means": means, "diffs": diffs}
    if len(diffs) >= 2:
        mean = statistics.mean(diffs)
        sd = statistics.stdev(diffs)
        se = sd / len(diffs) ** 0.5
        out.update({"mean_diff": mean, "sd": sd,
                    # Normal rather than t, and said out loud: at ten pairs the
                    # t correction is real. Treat the interval as indicative.
                    "ci": (mean - 1.96 * se, mean + 1.96 * se)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", nargs="?", default="play",
                    choices=["play", "status", "reveal"])
    ap.add_argument("--name", default="default")
    ap.add_argument("--games", type=int, default=None,
                    help="start a new session of this many games (even)")
    # The default comparison is the one the whole roadmap assumes the answer
    # to: more search is what everything else is buying, and nothing has ever
    # asked whether more search is more fun or merely stronger.
    ap.add_argument("--a-sims", type=int, default=400)
    ap.add_argument("--b-sims", type=int, default=1600)
    ap.add_argument("--value", default=str(ROOT / "runs/value.pt"))
    ap.add_argument("--policy", default=str(ROOT / "runs/policy.pt"))
    args = ap.parse_args()

    path = SESSIONS / args.name / "session.json"

    if args.games:
        if path.exists():
            print(f"{path} already exists. Pick another --name or delete it.",
                  file=sys.stderr)
            sys.exit(1)
        arms = {"a": {"value": args.value, "policy": args.policy, "sims": args.a_sims},
                "b": {"value": args.value, "policy": args.policy, "sims": args.b_sims}}
        state = new_session(args.name, args.games - args.games % 2, arms)
        save(state, path)
        print(f"session '{args.name}': {len(state['schedule'])} games, "
              f"{len(state['schedule']) // 2} pairs, blinding drawn and hidden.")
        print("Run again with no arguments to play the first game.\n")
        return

    state, path = load(args.name)
    played = len(state["results"])
    total = len(state["schedule"])

    if args.action == "status":
        print(f"session '{state['name']}': {played}/{total} games")
        if played:
            a = analyse(state)
            print(f"  {a['pairs']} complete pairs")
            print("  ratings so far are deliberately not broken down by arm")
        return

    if args.action == "reveal":
        if played < total:
            print(f"only {played}/{total} games played. Unblinding early turns the "
                  f"rest of the session into a different experiment.")
            if input("reveal anyway? [y/N] ").strip().lower() != "y":
                return
        a = analyse(state)
        print(f"\n  session '{state['name']}': {played} games, {a['pairs']} pairs\n")
        for arm in ("a", "b"):
            cfg = state["arms"][arm]
            mean = a["means"].get(arm)
            print(f"  arm {arm.upper()}: {cfg['sims']} sims, "
                  f"value={Path(cfg['value']).name}"
                  + (f"   mean rating {mean:.2f}" if mean is not None else ""))
        if "mean_diff" in a:
            lo, hi = a["ci"]
            print(f"\n  A minus B, per pair: {a['mean_diff']:+.2f} "
                  f"[{lo:+.2f}, {hi:+.2f}]  (sd {a['sd']:.2f}, n={a['pairs']})")
            if lo <= 0 <= hi:
                print("  The interval includes zero. On this evidence the two are")
                print("  equally worth sitting down with, which is a real finding:")
                print("  it says the difference the match harness can measure is not")
                print("  a difference you can feel.")
            else:
                better = "A" if a["mean_diff"] > 0 else "B"
                print(f"  Arm {better} is the one worth playing.")
        notes = [r for r in state["results"] if r["note"]]
        if notes:
            print("\n  what you said at the time:")
            for r in notes:
                print(f"    [{r['arm'].upper()} {r['rating']}] {r['note']}")
        return

    if played >= total:
        print(f"all {total} games played. `scripts/acceptance.py reveal --name "
              f"{args.name}` when you are ready.")
        return

    entry = state["schedule"][played]
    result = play_game(entry, state["arms"], played, total)
    if result is None:
        return
    state["results"].append(result)
    save(state, path)
    pgn_path = path.parent / "games.pgn"
    with pgn_path.open("a") as fh:
        fh.write(result.pop("pgn") + "\n\n")
    save(state, path)
    left = total - len(state["results"])
    print(f"\n  recorded. {left} game{'s' if left != 1 else ''} to go."
          + ("  Run it again when you have ten minutes." if left else
             "  Done -- `acceptance.py reveal` to unblind."))


if __name__ == "__main__":
    main()
