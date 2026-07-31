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
import time
from dataclasses import dataclass, field

import chess
import numpy as np

from sumofish.rules import terminal_value
from sumofish.tokenizer import ACTION_BY_MOVE_KEY, move_key


def _softmax_over_legal(row: np.ndarray, legal: list[chess.Move]) -> list[float]:
    """Renormalise a full 1968-move logit row onto the legal moves only.

    The masking is load-bearing, not a tidy-up: without it the prior leaks mass
    onto moves that cannot be played, and the search spends simulations
    proportional to that mass on children it will never create.

    A move outside the action space cannot happen -- the space is a superset of
    every legal move in any position and `tests/verify_data.py` checks that over
    real data -- but -1e9 keeps it survivable rather than a KeyError mid-game.
    """
    scores = np.array(
        [
            row[action] if (action := ACTION_BY_MOVE_KEY[move_key(m)]) >= 0 else -1e9
            for m in legal
        ]
    )
    exp = np.exp(scores - scores.max())
    return (exp / exp.sum()).tolist()


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
        # AlphaZero's published schedule. None restores the constant `c_puct`.
        c_puct_base: float | None = 19652.0,
        c_puct_init: float = 1.25,
        simulations: int = 400,
        batch: int = 1,               # >1 uses virtual loss to fill GPU batches
        dirichlet_alpha: float = 0.3,
        dirichlet_weight: float = 0.0,
        fpu: float = -0.2,
        reuse: bool = True,
        terminal=terminal_value,
    ) -> None:
        self.value_policy = value_policy
        self.policy = policy
        self.c_puct = c_puct
        self.c_puct_base = c_puct_base
        self.c_puct_init = c_puct_init
        self.simulations = simulations
        self.batch = max(1, batch)
        # Root exploration noise. Essential for self-play RL (it forces variety
        # in the training data); harmful when actually trying to win, so it
        # defaults off.
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        # "First Play Urgency": the Q assumed for an unvisited child, relative
        # to the parent's own Q. Slightly pessimistic means the search finishes
        # investigating a promising line before wandering off to a sibling.
        self.fpu = fpu
        # Injected so `rules.terminal_value_legacy` can be substituted for a
        # match. Nothing but an experiment should ever pass this.
        self.terminal = terminal
        self.evaluations = 0

        # Tree reuse. See `_reroot`. `_root_stack` is the move stack of the
        # board `_root` was searched from, copied because the caller's board
        # keeps moving.
        self.reuse = reuse
        self.reused = 0
        self._root: Node | None = None
        self._root_stack: list[chess.Move] = []

    def c_puct_at(self, visits: int) -> float:
        """How hard to explore, as a function of how much has been seen here.

        A constant is the obvious choice and it is wrong the moment the search
        budget moves, which it just did: the bot went from ~800 nodes a move in
        bullet to ~60,000 at 15+10. The exploration term is
        `c * P * sqrt(N_parent) / (1 + N_child)`, so the balance between trying
        something new and pursuing what already looks good shifts with N, and
        the c that balances it shifts too. Our 2.0 was never tuned at all, and
        against AlphaZero's schedule it over-explores below ~20,000 visits and
        under-explores above:

            visits      AZ      ours
               400    1.27      2.00
              5000    1.48      2.00
             20000    1.95      2.00
             60000    2.65      2.00

        AlphaZero's answer is to make it grow logarithmically rather than tune
        it per time control, which is the right shape here for a reason beyond
        elegance: this engine's budget is a clock, so its visit count varies
        from move to move within a single game as the clock runs down. There is
        no single N to tune for even if we wanted to.

        `c_puct_base` and `c_puct_init` are AlphaZero's published values, not
        fitted here. Set `c_puct_base` to None to get the old constant back,
        which is what `match.py --a-fixed-cpuct` does so the two can be played
        against each other rather than argued about.
        """
        if self.c_puct_base is None:
            return self.c_puct
        return math.log((1.0 + visits + self.c_puct_base) / self.c_puct_base) + self.c_puct_init

    def reset(self) -> None:
        """Forget the tree. Call between games, never within one."""
        self._root = None
        self._root_stack = []
        self.reused = 0

    # ---- tree reuse ------------------------------------------------------
    #
    # Every move, the search built a tree of a few thousand nodes, chose a
    # move, and threw the tree away. Then the opponent replied with a move the
    # search had usually already considered, and the next search rebuilt from
    # scratch a subtree it had just deleted.
    #
    # The statistics are still valid: every value in a node is stored from that
    # node's own side-to-move perspective, which does not depend on how the
    # position was reached. So the subtree under (our move, their reply) is
    # exactly the tree the next search wants, already partly built.
    #
    # The only hard part is proving the new board really is that node, and the
    # move stack proves it exactly. Comparing FENs would not: two different
    # lines can transpose into the same position with different repetition
    # histories, and reusing across a transposition would import a subtree whose
    # threefold evaluations were computed against the wrong history.

    def _reroot(self, board: chess.Board) -> Node | None:
        """The stored node for `board`, or None to start fresh.

        Declines rather than guesses. A new game, a jumped-to position, an
        analysis GUI walking backwards, or the caller handing us a board built
        without a move stack all land here and all get a fresh root, which is
        merely the old behaviour.
        """
        if self._root is None:
            return None

        stack = board.move_stack
        played = len(stack) - len(self._root_stack)
        # Two plies is the normal case: our move and their reply. One covers
        # the engine being asked to move twice in a row (analysis). More than
        # that and we have missed part of the game.
        if not 1 <= played <= 2:
            return None
        if stack[: len(self._root_stack)] != self._root_stack:
            return None

        node = self._root
        for move in stack[len(self._root_stack):]:
            child = node.children.get(move)
            # An unexpanded child has no subtree worth keeping, and `search`
            # would have to expand it anyway.
            if child is None or not child.expanded:
                return None
            node = child

        # It is a root now, so nothing scores it by its prior any more.
        node.prior = 1.0
        return node

    # ---- phase 2: expand -------------------------------------------------

    def _priors(self, board: chess.Board) -> dict[chess.Move, float]:
        legal = list(board.legal_moves)
        if not legal:
            return {}
        if self.policy is None:
            return {m: 1.0 / len(legal) for m in legal}

        row = self.policy._logprobs([board])[0].cpu().numpy()
        return dict(zip(legal, _softmax_over_legal(row, legal), strict=True))

    def _expand(self, node: Node, board: chess.Board) -> float:
        """Create children and return this node's value, side-to-move relative."""
        value = self.terminal(board)
        if value is not None:
            node.terminal_value = value
            return value

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
        u = (self.c_puct_at(parent.visits) * child.prior
             * math.sqrt(parent.visits) / (1 + child.visits))
        return q + u

    def _select_child(self, node: Node) -> tuple[chess.Move, Node]:
        """The same score as `_puct`, computed the way a hot loop should be.

        This is 29% of the search's wall clock, measured, and it was written as
        `max(children.items(), key=lambda kv: self._puct(node, kv[1]))`. That is
        the readable form and it pays for readability four times per child: a
        Python function call, a lambda call, a property lookup for `child.q`,
        and a `math.sqrt` of a quantity that is *the same for every child*.

        Hoisting the invariant and inlining the body is 2.5x on this step.
        Measured in isolation at realistic branching factors, us per selection:

            children   as written   hoisted   numpy
                  12         3.19      1.36    8.21
                  20         5.10      2.05    8.25
                  30         7.46      2.98    8.19
                  47        11.32      4.32    8.33

        Note the third column, because the obvious next move is to lay the
        children out as parallel arrays and score them with one numpy
        expression, and it **loses at every branching factor chess produces**.
        Array-of-children is the right layout for Lc0, which batches selections
        across many nodes at once; for one node with thirty children, numpy's
        per-call overhead is larger than the loop it replaces. Do not
        reintroduce it without batching the selections too.

        `_puct` is kept, unused by this path, because it is the readable
        statement of what is being computed and `tests/verify_search.py` checks
        the two against each other. If they ever disagree, this one is wrong.
        """
        # sqrt(N_parent) and the FPU fallback do not vary across children.
        c_sqrt = self.c_puct_at(node.visits) * math.sqrt(node.visits)
        fpu_q = node.value_sum / node.visits + self.fpu if node.visits else self.fpu
        if fpu_q < 0.0:
            fpu_q = 0.0

        best = None
        best_score = -1e18
        for move, child in node.children.items():
            visits = child.visits
            # Inlined `child.q` and the parent's-frame flip. A property call
            # per child per ply per simulation is not free at this rate.
            q = (1.0 - child.value_sum / visits) if visits else fpu_q
            score = q + c_sqrt * child.prior / (1 + visits)
            if score > best_score:
                best_score = score
                best = (move, child)
        return best

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

    # ---- batched search --------------------------------------------------
    #
    # The unbatched loop above evaluates exactly one position per GPU call.
    # Measured: 2.553 ms per position at batch 1, 0.219 ms at batch 64. Nearly
    # all of that is per-call overhead, so the card spends its life waiting.
    #
    # The obstacle is that MCTS is sequential by construction: each walk is
    # supposed to see the statistics left by the previous one. Walk down twice
    # without evaluating in between and both walks take the identical path,
    # because nothing changed. You get a batch of duplicates.
    #
    # VIRTUAL LOSS is the standard fix. On reaching a leaf, immediately back up
    # a *pretend* result as if that line had lost, before knowing anything about
    # it. The next walk sees the discouraged branch and goes somewhere else. Once
    # the real value arrives, the pretend result is subtracted and the true one
    # added.
    #
    # Sign note, following this file's convention: backing up 1.0 from the leaf
    # raises the leaf's own Q, and the parent sees `1 - child.q`, so the parent's
    # view of that branch DROPS. The flip then propagates the discouragement all
    # the way to the root, which is exactly what is wanted.

    def _descend(self, root: Node, board: chess.Board) -> tuple[list[Node], int]:
        path, node, undo = [root], root, 0
        while node.expanded and node.children:
            move, node = self._select_child(node)
            board.push(move)
            undo += 1
            path.append(node)
        return path, undo

    @staticmethod
    def _backup(path: list[Node], value: float, sign: int = 1) -> None:
        for n in reversed(path):
            n.visits += sign
            n.value_sum += sign * value
            value = 1.0 - value

    def _simulate_batch(self, root: Node, board: chess.Board, batch: int) -> None:
        pending: list[tuple[list[Node], chess.Board]] = []

        for _ in range(batch):
            path, undo = self._descend(root, board)
            leaf = path[-1]

            if leaf.terminal_value is not None:
                self._backup(path, leaf.terminal_value)
                for _ in range(undo):
                    board.pop()
                continue

            value = self.terminal(board)
            if value is not None:
                leaf.terminal_value = value
                self._backup(path, value)
                for _ in range(undo):
                    board.pop()
                continue

            pending.append((path, board.copy(stack=False)))
            self._backup(path, 1.0)          # virtual loss, undone below
            for _ in range(undo):
                board.pop()

        if not pending:
            return

        # One GPU call for priors, one for values, covering the whole batch.
        boards = [b for _, b in pending]
        priors = self._priors_batch(boards)
        values, _ = self.value_policy.evaluate(boards)
        self.evaluations += len(boards)

        for (path, _), prior, value in zip(pending, priors, values, strict=True):
            leaf = path[-1]
            # Another walk in this same batch may have already expanded it.
            if not leaf.children:
                for move, p in prior.items():
                    leaf.children[move] = Node(prior=p, to_move=not leaf.to_move)
            self._backup(path, 1.0, sign=-1)  # remove the virtual loss
            self._backup(path, float(value))  # add the real result

    def _priors_batch(self, boards: list[chess.Board]) -> list[dict[chess.Move, float]]:
        if self.policy is None:
            return [
                {m: 1.0 / max(1, len(list(b.legal_moves))) for m in b.legal_moves}
                for b in boards
            ]
        rows = self.policy._logprobs(boards).cpu().numpy()
        out = []
        for row, b in zip(rows, boards, strict=True):
            legal = list(b.legal_moves)
            if not legal:
                out.append({})
                continue
            out.append(dict(zip(legal, _softmax_over_legal(row, legal), strict=True)))
        return out

    def search(
        self,
        board: chess.Board,
        deadline: float | None = None,
        on_progress=None,
        progress_every: float = 0.15,
    ) -> tuple[Node, dict[chess.Move, int]]:
        """Run simulations until the budget or `deadline` (a perf_counter time).

        A real game gives you a clock, not a simulation count, so the deadline
        wins when both are set. Batches are checked between groups rather than
        mid-batch, so overshoot is bounded by one batch, which is ~15ms.

        `on_progress(root, done)` is called between batches, no more often than
        `progress_every` seconds, so an observer can watch the tree develop
        rather than only seeing the conclusion. It runs on this thread and
        therefore on the clock, which is why it is rate limited and why
        anything it raises is swallowed: a spectator must never be able to cost
        the engine a move. It defaults to None and off.
        """
        self.evaluations = 0

        # A reused root arrives already expanded and already carrying visits.
        # Expanding it again would overwrite its children and destroy the whole
        # point, so the expand is conditional, which it did not need to be when
        # every root was brand new.
        root = self._reroot(board) if self.reuse else None
        self.reused = root.visits if root is not None else 0
        if root is None:
            root = Node(prior=1.0, to_move=board.turn)
        if not root.expanded:
            self._expand(root, board)

        if self.dirichlet_weight > 0 and root.children:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(root.children))
            for (move, child), n in zip(root.children.items(), noise, strict=True):
                child.prior = (1 - self.dirichlet_weight) * child.prior + self.dirichlet_weight * n

        next_report = time.perf_counter() + progress_every

        def report(done: int) -> None:
            nonlocal next_report
            now = time.perf_counter()
            if now < next_report:
                return
            next_report = now + progress_every
            try:
                on_progress(root, done)
            except Exception:
                pass

        if self.batch > 1:
            done = 0
            while done < self.simulations:
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                n = min(self.batch, self.simulations - done)
                self._simulate_batch(root, board, n)
                done += n
                if on_progress is not None:
                    report(done)
        else:
            for i in range(self.simulations):
                if deadline is not None and i % 16 == 0 and time.perf_counter() >= deadline:
                    break
                self._simulate(root, board)
                if on_progress is not None and i % 16 == 0:
                    report(i)

        if self.reuse:
            self._root = root
            self._root_stack = list(board.move_stack)

        return root, {m: c.visits for m, c in root.children.items()}

    # ---- what the engine calls ------------------------------------------

    def play(self, board: chess.Board, deadline: float | None = None) -> chess.Move:
        root, visits = self.search(board, deadline=deadline)
        if not visits:
            raise ValueError(f"no legal moves in {board.fen()}")
        # Most-visited, not best-average. A child can luck into a high average
        # from two visits; it only accumulates visits by surviving PUCT
        # repeatedly.
        return max(visits.items(), key=lambda kv: kv[1])[0]

    def play_batch(self, boards: list[chess.Board]) -> list[chess.Move]:
        return [self.play(b) for b in boards]

    def report(
        self, root: Node, board: chess.Board, top_n: int = 5, pv_max: int = 12
    ) -> dict:
        """A human-readable account of what a searched tree currently believes.

        Split out from `analyse` because the same account is worth having
        *during* the search as well as at the end of it, and because SAN is not
        free -- python-chess generates legal moves to work out disambiguation --
        so a caller on a running clock wants to bound how much of it happens.
        Hence `top_n` and `pv_max`.

        `win_prob` is root Q, which is the probability that **the side to move**
        wins. Anything that stores or plots it across plies has to convert to a
        fixed frame first or it sawtooths; see `sumofish.telemetry.perspective`.
        """
        ranked = sorted(root.children.items(), key=lambda kv: -kv[1].visits)
        if not ranked:
            return {"move": None, "win_prob": root.q, "evaluations": self.evaluations,
                    "reused": self.reused, "pv": [], "top": [],
                    "mate": board.is_checkmate()}

        # Principal variation: the line the search actually believes in, read
        # off by following most-visited children down from the root. SAN has to
        # be taken *before* the push, because disambiguation depends on the
        # position the move is played from.
        pv, node, tmp = [], root, board.copy()
        while node.children and len(pv) < pv_max:
            move = max(node.children.items(), key=lambda kv: kv[1].visits)[0]
            if node.children[move].visits == 0:
                break
            pv.append(tmp.san(move))
            tmp.push(move)
            node = node.children[move]

        return {
            "move": ranked[0][0],
            "win_prob": root.q,
            "evaluations": self.evaluations,
            # Simulations inherited from the previous move's tree. `evaluations`
            # counts only what this search paid for, so nps stays honest; this
            # is the head start that made it cheaper.
            "reused": self.reused,
            "pv": pv,
            # Reported as visit share of the total, because that is what the
            # search actually spent, and it is what the move choice is made on.
            "top": [
                (board.san(m), c.visits, round(c.q, 3), round(c.prior, 3))
                for m, c in ranked[:top_n]
            ],
            # An honest mate flag: does the line the search believes in end in
            # checkmate. There is no mate-distance score anywhere in this
            # engine, so nothing here pretends to be UCI's `score mate N`.
            "mate": tmp.is_checkmate(),
        }

    def analyse(self, board: chess.Board) -> dict:
        """Search plus a human-readable account of what it concluded."""
        root, _visits = self.search(board)
        return self.report(root, board)
