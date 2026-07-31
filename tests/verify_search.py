"""Verify the search's rules layer and its tree reuse.

Run: .venv/bin/python tests/verify_search.py

Both things checked here replaced something that worked, purely to make it
faster, which is the exact situation where a silent behaviour change is most
likely and least visible. A search that quietly stops seeing checkmate does not
crash, does not fail a type check, and loses games slowly.

  1. `rules.terminal_value` against `board.outcome(claim_draw=True)` over
     thousands of positions from random playouts. Every disagreement has to
     fall into the one class this deliberately changed: a draw that is
     claimable only by *making a move*, which is the child's business.
  2. Mate in one is still the highest-visited move. This is the regression the
     sign-convention bug produced once already, and it is the cheapest possible
     canary for anything that touches terminal values.
  3. Tree reuse re-roots when and only when it can prove the board is the node
     it thinks it is, and the moves it produces are legal.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sumofish.rules import terminal_value  # noqa: E402

PASS, FAIL = "\033[32mok\033[0m", "\033[31mFAIL\033[0m"
failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"  [{PASS if ok else FAIL}] {name}{' ' + detail if detail else ''}")
    failures += not ok


def reference(board: chess.Board) -> float | None:
    """What the old code computed, in the new code's units."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    return 0.0 if outcome.winner is not None else 0.5


# ---------------------------------------------------------------------------
print("terminal_value, by hand")

cases = [
    ("checkmate (fool's mate)", "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", 0.0),
    ("stalemate", "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 0.5),
    ("bare kings", "7k/8/6K1/8/8/8/8/8 w - - 0 1", 0.5),
    ("king and bishop", "7k/8/6K1/8/8/8/8/6B1 w - - 0 1", 0.5),
    ("king and knight", "7k/8/6K1/8/8/8/8/6N1 w - - 0 1", 0.5),
    ("fifty-move claimable", "7k/8/6K1/8/8/8/6R1/8 w - - 100 80", 0.5),
    ("one short of fifty", "7k/8/6K1/8/8/8/6R1/8 w - - 99 80", None),
    ("ordinary position", chess.STARTING_FEN, None),
]
for name, fen, want in cases:
    got = terminal_value(chess.Board(fen))
    check(name, got == want, f"got {got}, want {want}")

# Threefold has to be built by playing, not by a FEN: the history is the point.
board = chess.Board()
for _ in range(2):
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        board.push_uci(uci)
check("threefold after two round trips", terminal_value(board) == 0.5,
      f"got {terminal_value(board)}")
board.pop()
check("not threefold one ply earlier", terminal_value(board) is None,
      f"got {terminal_value(board)}")

# ---------------------------------------------------------------------------
print("\nterminal_value vs outcome(claim_draw=True) over random playouts")

rng = random.Random(20260728)
positions = agreed = claimable_only = 0
mismatches: list[str] = []

for game in range(400):
    board = chess.Board()
    # Short-sighted random play reaches repetitions and fifty-move draws far
    # more often than real play does, which is exactly what needs covering.
    for _ in range(220):
        legal = list(board.legal_moves)
        if not legal:
            break
        # Bias hard toward quiet moves so the halfmove clock climbs and
        # repetitions actually happen. Pure random play captures its way to a
        # bare-kings draw and never exercises either rule.
        quiet = [m for m in legal if not board.is_capture(m)
                 and board.piece_type_at(m.from_square) != chess.PAWN]
        board.push(rng.choice(quiet if quiet and rng.random() < 0.9 else legal))

        positions += 1
        ours, theirs = terminal_value(board), reference(board)
        if ours == theirs:
            agreed += 1
        elif (
            ours is None
            and theirs == 0.5
            and board.can_claim_draw()
            and not board.is_repetition(3)
            and board.halfmove_clock < 100
        ):
            # The deliberate divergence: a draw reachable by making a move.
            # The child node is where that gets decided.
            claimable_only += 1
        else:
            mismatches.append(f"{board.fen()} ours={ours} theirs={theirs}")
        if theirs is not None:
            break

check(f"{positions} positions, {agreed} identical, "
      f"{claimable_only} claimable-by-a-move only",
      not mismatches, "" if not mismatches else f"\n      {mismatches[0]}")
check("the divergence class was actually exercised", claimable_only > 0,
      f"{claimable_only} cases")

# ---------------------------------------------------------------------------
print("\ntokenize_board against tokenize(board.fen())")

import numpy as np  # noqa: E402

from sumofish.tokenizer import tokenize, tokenize_board  # noqa: E402

rng = random.Random(4242)
checked = differed = 0
first: str | None = None
for _ in range(300):
    board = chess.Board()
    for _ in range(160):
        legal = list(board.legal_moves)
        if not legal:
            break
        board.push(rng.choice(legal))
        checked += 1
        if not np.array_equal(tokenize(board.fen()), tokenize_board(board)):
            differed += 1
            first = first or board.fen()

check(f"{checked} positions from random games agree", not differed, first or "")

# Puzzle positions specifically, because that is the distribution every
# reported accuracy number is measured on.
import csv  # noqa: E402

with open(ROOT_DIR := Path(__file__).resolve().parent.parent / "data/puzzles.csv") as fh:
    puzzles = [row["FEN"] for _, row in zip(range(4000), csv.DictReader(fh))]
bad = [f for f in puzzles
       if not np.array_equal(tokenize(chess.Board(f).fen()), tokenize_board(chess.Board(f)))]
check(f"{len(puzzles)} puzzle positions agree", not bad, bad[0] if bad else "")

# ---------------------------------------------------------------------------
print("\nMCTS with the real nets")

import torch  # noqa: E402

from sumofish.engines.neural_engine import load_policy  # noqa: E402
from sumofish.hlgauss import HLGauss  # noqa: E402
import sumofish.mcts as sumofish_mcts  # noqa: E402
from sumofish.mcts import MCTS  # noqa: E402
from sumofish.model import ChessTransformer, ModelConfig  # noqa: E402
from sumofish.value_policy import ValuePolicy  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ck = torch.load(ROOT / "runs/value.pt", map_location="cuda:0", weights_only=False)
model = ChessTransformer(ModelConfig(**ck["cfg"]))
model.load_state_dict({k: v.float() for k, v in (ck.get("ema") or ck["model"]).items()})
value = ValuePolicy(model, HLGauss(bins=ck["cfg"]["output_size"]), device="cuda:0")
prior = load_policy(str(ROOT / "runs/policy.pt"))[0]
mcts = MCTS(value, policy=prior, simulations=200, batch=16)

# Back-rank mate in one. The sign-convention bug scored this as the WORST move
# on the board and played a king move instead.
mate = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
mcts.reset()
check("mate in one is played", mcts.play(mate).uci() == "a1a8",
      f"played {mcts.play(mate).uci()}")

# ---------------------------------------------------------------------------
print("\ntree reuse")

mcts.reset()
board = chess.Board()
reused_at, illegal = [], []
for ply in range(12):
    root, visits = mcts.search(board)
    move = max(visits.items(), key=lambda kv: kv[1])[0]
    if move not in board.legal_moves:
        illegal.append((board.fen(), move.uci()))
    if ply > 0:
        reused_at.append(mcts.reused)
    board.push(move)

check("never produced an illegal move", not illegal, str(illegal[:1]))
check("re-rooted on later moves", sum(r > 0 for r in reused_at) >= 8,
      f"reused visits per ply: {reused_at}")
check("inherited a real head start", max(reused_at) > 20,
      f"best was {max(reused_at)} visits")

# Declines when it cannot prove the position.
mcts.reset()
check("declines with no stored tree", mcts._reroot(chess.Board()) is None)

mcts.reset()
b = chess.Board()
mcts.search(b)
b.push_uci("e2e4")
b.push_uci("e7e5")
b.push_uci("g1f3")
check("declines a three-ply jump", mcts._reroot(b) is None)

mcts.reset()
b = chess.Board()
mcts.search(b)
check("declines an unrelated position",
      mcts._reroot(chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")) is None)

mcts.reset()
b = chess.Board()
mcts.search(b)
b.push_uci("e2e4")
node = mcts._reroot(b)
check("accepts the one-ply continuation", node is not None and node.visits > 0,
      f"node={node and node.visits}")
check("reset() forgets it", (mcts.reset() or mcts._reroot(b)) is None)

# ---------------------------------------------------------------------------
print("\n_select_child agrees with the readable _puct it replaced")

rng = random.Random(99)
disagreements = []
checked = 0
for trial in range(4000):
    n_children = rng.randint(2, 50)
    parent = sumofish_mcts.Node(prior=1.0, to_move=chess.WHITE)
    parent.visits = rng.randint(0, 4000)
    parent.value_sum = parent.visits * rng.random()
    for i in range(n_children):
        c = sumofish_mcts.Node(prior=rng.random(), to_move=chess.BLACK)
        # A realistic mix: most children unvisited, a few heavily visited.
        c.visits = 0 if rng.random() < 0.55 else rng.randint(1, 400)
        c.value_sum = c.visits * rng.random()
        parent.children[chess.Move(i % 64, (i * 7 + 3) % 64)] = c

    fast = mcts._select_child(parent)
    slow = max(parent.children.items(), key=lambda kv: mcts._puct(parent, kv[1]))
    checked += 1
    if fast[0] != slow[0]:
        # A genuine tie broken differently by float association is acceptable;
        # a different SCORE is not.
        if abs(mcts._puct(parent, fast[1]) - mcts._puct(parent, slow[1])) > 1e-9:
            disagreements.append((n_children, parent.visits))

check(f"{checked} random nodes, 2-50 children, pick the same child",
      not disagreements, str(disagreements[:2]))

# ---------------------------------------------------------------------------
print(f"\n{'all checks passed' if not failures else f'{failures} FAILED'}")
sys.exit(1 if failures else 0)

