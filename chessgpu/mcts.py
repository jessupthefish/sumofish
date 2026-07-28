"""Monte Carlo Tree Search. This is where SumoFish starts thinking ahead.

## What search actually buys

The searchless engine asks one question: "given this board, what move?" It has
no way to notice that its favourite move loses a rook in three. Search fixes
that by *playing out* candidate lines and letting the evaluation function judge
where they end up.

Classical engines (Stockfish) do alpha-beta: enumerate nearly everything to a
fixed depth, prune what cannot matter. That works when your evaluation is cheap
-- Stockfish evaluates tens of millions of positions per second.

Ours costs ~4ms on a GPU, so we get maybe a few thousand evaluations per move,
not tens of millions. That rules out alpha-beta and rules *in* MCTS, which is
designed for exactly this regime: few, expensive, high-quality evaluations,
spent adaptively on the lines that look most promising.

## The four phases, repeated N times

Each iteration is one "simulation" and walks the tree once:

  1. SELECT   From the root, walk down by repeatedly picking the child with
              the highest PUCT score, until reaching a node not yet expanded.
  2. EXPAND   Ask the policy net which moves are plausible here; create a child
              for each with that prior attached.
  3. EVALUATE Ask the value net how good this new position is.
  4. BACKUP   Walk back up to the root adding that value to every node on the
              path, flipping the sign at each step because the players alternate.

After N simulations, play the move with the most VISITS -- not the best average
value. Visit count is the more robust statistic: a node can luck into a great
average from two visits, but it only accumulates visits by repeatedly surviving
PUCT's scrutiny. This is what AlphaZero does and the reason is worth
remembering.

## PUCT, the one formula that matters

For each child we compute:

    score = Q  +  c_puct * P * sqrt(N_parent) / (1 + N_child)
            ^                ^
            |                exploration: high for unvisited children with a
            |                high prior, decays as we visit them
            exploitation: average value observed through this child so far

`P` is the policy prior. It is why our first 8.5-hour training run is not
wasted: it tells the search which of 33 legal moves are worth spending
simulations on. Measured on a real position, the top 6 moves carry 98% of the
prior mass, so the search effectively looks at 6 moves deeply instead of 33
shallowly.

`c_puct` trades the two off. ~1.5-4 is the usual range; higher explores more.

## Sign conventions, which is where implementations usually break

Every value in this file is **from the perspective of the side to move at that
node**. A node's Q of 0.8 means "the player about to move here wins 80% of the
time". Because the player alternates every ply, backing a value up the tree
requires flipping it at each level. Get this wrong and the engine confidently
plays into losses; it is the classic MCTS bug and it does not announce itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import chess
import numpy as np


@dataclass
class Node:
    """One position in the tree."""

    prior: float                      # P: policy's probability for the move that got here
    to_move: chess.Color              # whose turn it is at this node
    visits: int = 0                   # N
    value_sum: float = 0.0            # W, accumulated from this node's own perspective
    children: dict[chess.Move, "Node"] = field(default_factory=dict)
    terminal_value: float | None = None   # set for checkmate/stalemate, never re-evaluated

    @property
    def q(self) -> float:
        """Average value from this node's side-to-move perspective.

        An unvisited node has no evidence. Returning 0 (a loss) would make the
        search refuse to try anything new, so the caller supplies a default
        instead -- see `_puct`.
        """
        return self.value_sum / self.visits if self.visits else 0.0

    @property
    def expanded(self) -> bool:
        return bool(self.children) or self.terminal_value is not None


class MCTS:
    """Search using a policy net for priors and a value net for leaves.

    Both are optional-ish in principle: with no policy you fall back to a
    uniform prior, which works but wastes simulations on obviously bad moves.
    """

    def __init__(
        self,
        value_policy,                 # ValuePolicy: scores positions
        policy=None,                  # NeuralPolicy: supplies move priors
        c_puct: float = 2.0,
        simulations: int = 400,
        dirichlet_alpha: float = 0.3,
        dirichlet_weight: float = 0.0,
        fpu: float = -0.2,
    ) -> None:
        self.value_policy = value_policy
        self.policy = policy
        self.c_puct = c_puct
        self.simulations = simulations
        # Root exploration noise. Essential for self-play RL (it forces variety
        # in the training data); harmful when actually trying to win, so it
        # defaults off.
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        # "First Play Urgency": the Q assumed for an unvisited child, relative
        # to the parent's own Q. Slightly pessimistic means the search finishes
        # investigating a promising line before wandering off to a sibling.
        self.fpu = fpu
        self.evaluations = 0

    # ---- phase 2: expand -------------------------------------------------

    def _priors(self, board: chess.Board) -> dict[chess.Move, float]:
        legal = list(board.legal_moves)
        if not legal:
            return {}
        if self.policy is None:
            return {m: 1.0 / len(legal) for m in legal}

        from chessgpu.tokenizer import MOVE_TO_ACTION

        row = self.policy._logprobs([board])[0].cpu().numpy()
        scores = np.array(
            [row[MOVE_TO_ACTION[m.uci()]] if m.uci() in MOVE_TO_ACTION else -1e9 for m in legal]
        )
        exp = np.exp(scores - scores.max())
        probs = exp / exp.sum()
        return dict(zip(legal, probs.tolist(), strict=True))

    def _expand(self, node: Node, board: chess.Board) -> float:
        """Create children and return this node's value, side-to-move relative."""
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            # Terminal. A player never delivers mate on their own move, so if
            # the game is over and someone won, the side to move here LOST.
            node.terminal_value = 0.0 if outcome.winner is not None else 0.5
            return node.terminal_value

        for move, prior in self._priors(board).items():
            node.children[move] = Node(prior=prior, to_move=not node.to_move)

        self.evaluations += 1
        return self.value_policy.value_of(board)

    # ---- phase 1: select -------------------------------------------------

    def _puct(self, parent: Node, child: Node) -> float:
        """Score a child FROM THE PARENT'S PERSPECTIVE.

        `child.q` is stored from the child's own side-to-move perspective, and
        the players alternate, so the parent's view of that child is
        `1 - child.q`. Missing this flip is the classic MCTS bug and it does
        not announce itself -- the engine simply plays into losses while
        looking healthy.

        It was in this file. Test case: a position with mate in one available.
        The mate node correctly stored terminal_value = 0.0, meaning "the side
        to move here (the mated player) wins 0% of the time". Read unflipped,
        that made checkmate the WORST-scoring child, and the search picked a
        pointless king move instead, 103 visits to 52.
        """
        # Unvisited children have no evidence. Assume slightly worse than the
        # parent's own value, so the search finishes investigating a promising
        # line before wandering to a sibling. Both terms below are in the
        # parent's frame.
        q = (1.0 - child.q) if child.visits else max(0.0, parent.q + self.fpu)
        u = self.c_puct * child.prior * math.sqrt(parent.visits) / (1 + child.visits)
        return q + u

    def _select_child(self, node: Node) -> tuple[chess.Move, Node]:
        return max(node.children.items(), key=lambda kv: self._puct(node, kv[1]))

    # ---- the loop --------------------------------------------------------

    def _simulate(self, root: Node, board: chess.Board) -> None:
        path: list[Node] = [root]
        node = root
        undo = 0

        # Walk down until we hit something unexpanded.
        while node.expanded and node.children:
            move, node = self._select_child(node)
            board.push(move)
            undo += 1
            path.append(node)

        value = (
            node.terminal_value
            if node.terminal_value is not None
            else self._expand(node, board)
        )

        # Phase 4: back up, flipping every level. `value` is always expressed
        # from the perspective of the node currently being updated.
        for n in reversed(path):
            n.visits += 1
            n.value_sum += value
            value = 1.0 - value

        for _ in range(undo):
            board.pop()

    def search(self, board: chess.Board) -> tuple[Node, dict[chess.Move, int]]:
        root = Node(prior=1.0, to_move=board.turn)
        self.evaluations = 0
        self._expand(root, board)

        if self.dirichlet_weight > 0 and root.children:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(root.children))
            for (move, child), n in zip(root.children.items(), noise, strict=True):
                child.prior = (1 - self.dirichlet_weight) * child.prior + self.dirichlet_weight * n

        for _ in range(self.simulations):
            self._simulate(root, board)

        return root, {m: c.visits for m, c in root.children.items()}

    # ---- what the engine calls ------------------------------------------

    def play(self, board: chess.Board) -> chess.Move:
        root, visits = self.search(board)
        if not visits:
            raise ValueError(f"no legal moves in {board.fen()}")
        # Most-visited, not best-average. A child can luck into a high average
        # from two visits; it only accumulates visits by surviving PUCT
        # repeatedly.
        return max(visits.items(), key=lambda kv: kv[1])[0]

    def play_batch(self, boards: list[chess.Board]) -> list[chess.Move]:
        return [self.play(b) for b in boards]

    def analyse(self, board: chess.Board) -> dict:
        """Search plus a human-readable account of what it concluded."""
        root, visits = self.search(board)
        ranked = sorted(root.children.items(), key=lambda kv: -kv[1].visits)
        best = ranked[0][0]

        # Principal variation: the line the search actually believes in, read
        # off by following most-visited children down from the root.
        pv, node, tmp = [], root, board.copy()
        while node.children:
            move = max(node.children.items(), key=lambda kv: kv[1].visits)[0]
            if node.children[move].visits == 0:
                break
            pv.append(tmp.san(move))
            tmp.push(move)
            node = node.children[move]

        return {
            "move": best,
            "win_prob": root.q,
            "evaluations": self.evaluations,
            "pv": pv,
            "top": [
                (board.san(m), c.visits, round(c.q, 3), round(c.prior, 3))
                for m, c in ranked[:5]
            ],
        }
