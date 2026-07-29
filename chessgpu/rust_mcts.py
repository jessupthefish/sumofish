"""The Rust search, behind the interface `search_engine.py` already expects.

Selected with `CHESSGPU_CORE=rust`. Anything else, including unset, keeps the
pure-Python `chessgpu.mcts.MCTS`, so rollback is an environment variable rather
than a revert and the two can be A/B'd directly.

# Why this is allowed to exist before the Elo instrument is trusted

Every other change on the roadmap needs a measurement to justify. This one does
not, because `sumofish_core` is a *faithful* port: with `dedup` and
`mate_distance` off it produces byte-identical root visit vectors to
`chessgpu.mcts`, verified over 40 positions x 800 simulations at five batch sizes,
and across whole games with tree reuse on. Identical visits means identical moves.
So this is pure speed with provably no strength change, which is the only claim on
the board that does not depend on the exchange rate that was retracted.

`tests/identity_engine.py` re-checks that with the REAL networks rather than the
mock used during the port, because a mock cannot catch a mismatch in how the two
paths call torch.

# The one thing that is NOT identical, and why

`dedup=True` sends fewer rows to the network -- 66.4% fewer at the live batch of
64, since that many were the same positions again. The TREE is unchanged (k paths
to one leaf still get k backups), so the visit vector is identical and the moves
are identical. But a smaller batch can change float reductions inside the forward
pass, so with the real net the values can differ in the last bits even though the
search logic did not. That is why it is a separate flag from the port itself.

# What is deliberately left in Python

The prior softmax. `numpy`'s float32 `exp` is its own SIMD implementation and is
not correctly rounded; it differs from a correctly-rounded double `exp` on 39.7%
of inputs by ~1 ULP, which compounds to 95.4% of real positions getting at least
one differing prior. A 1-ULP prior only changes a move when it lands on a PUCT
tie, so this is exactly the class of error that would pass a casual test while
being wrong. See `rust/src/softmax.rs`.
"""

from __future__ import annotations

import os

import chess
import numpy as np
import torch

import sumofish_core as core


def make_evaluator(policy, value_policy, compile_nets: bool = False,
                   pad_to: int | None = None):
    """The callback the Rust search calls once per batch of leaves.

    Rust sends FENs and, for each, the action index of every legal move in
    generation order. It sends the action indices specifically so this function
    never touches `python-chess`: an earlier contract sent FENs alone and forced a
    `chess.Board` rebuild plus a move regeneration here, which cost 78 ms of a
    200 ms callback doing work Rust had already done.

    Returns `(priors, values)`, positionally aligned to that same move order.
    """
    device = value_policy.device
    dev_type = device.split(":")[0]
    dtype = value_policy.dtype

    # Profiled with the real nets: the forward pass is 89% of the search once the
    # tree is in Rust, and it is LAUNCH-bound below ~128 rows -- one row costs the
    # same 7.2 ms as 128. Two consequences drive everything here.
    #
    # 1. `reduce-overhead` means CUDA graphs, which is exactly the right tool for
    #    a launch-bound call. Measured 6.58 -> 3.53 ms at batch 64, so 1.9x on the
    #    dominant cost.
    # 2. CUDA graphs need a STATIC shape, and dedup makes the row count ragged
    #    (453 distinct rows over 25 batches, ~18 each, varying). Padding every
    #    call up to a fixed size fixes the shape -- and costs nothing, because the
    #    rows below 128 were free anyway. Without the padding, every new row count
    #    triggers a recompile, inside a search, on a running chess clock.
    pmodel, vmodel = policy.model, value_policy.model
    if compile_nets:
        pmodel = torch.compile(pmodel, mode="reduce-overhead")
        vmodel = torch.compile(vmodel, mode="reduce-overhead")

    @torch.inference_mode()
    def evaluate(fens: list[str], actions: list[list[int]]):
        # One tokenisation, in Rust, for the whole batch. This replaces
        # `np.stack([tokenize_board(b) for b in boards])`, which both nets did
        # SEPARATELY in the Python path -- so this is one pass where there were
        # two, and 16x faster per position. Byte-identical over 40,014 positions.
        buf = core.tokenize_batch(fens)
        # np.frombuffer gives a READ-ONLY view, and torch.from_numpy on one warns
        # about undefined behaviour on write. .copy() makes it writable for the
        # cost of a memcpy of 77 bytes per position, which is nothing next to the
        # forward pass it feeds.
        n = len(fens)
        tokens = np.frombuffer(buf, dtype=np.uint8).reshape(n, 77).copy()

        if pad_to is not None and n < pad_to:
            # Repeat the last row rather than zero-padding: an all-zero token
            # sequence is not a legal position, and feeding the net garbage to
            # discard is a good way to hit a NaN that then looks like a model
            # problem. The padding rows are sliced off below and never read.
            pad = np.repeat(tokens[-1:], pad_to - n, axis=0)
            tokens = np.concatenate([tokens, pad], axis=0)

        x = torch.from_numpy(tokens).long().to(device, non_blocking=True)

        with torch.autocast(dev_type, dtype=dtype):
            # Same two models, same order, same `.float()` as
            # `policy._logprobs` and `ValuePolicy.evaluate`. One input tensor
            # rather than two identical ones.
            plogits = pmodel(x).float()
            vlogits = vmodel(x).float()

        rows = plogits[:n].cpu().numpy()      # float32, as `.float()` gives
        values = value_policy.hl.expectation(vlogits[:n]).cpu().numpy()

        priors = [_softmax_over_legal(rows[i], actions[i]) for i in range(n)]
        return priors, [float(v) for v in values]

    return evaluate


def _softmax_over_legal(row: np.ndarray, actions: list[int]) -> list[float]:
    """`mcts._softmax_over_legal`, by action index instead of by move.

    Expression for expression, including the `-1e9` for a move outside the action
    space and the float32 arithmetic that `policy._logprobs`'s `.float()` implies.
    This must stay numpy: see the module docstring.
    """
    scores = np.array(
        [row[a] if a >= 0 else -1e9 for a in actions], dtype=np.float32
    )
    exp = np.exp(scores - scores.max())
    return (exp / exp.sum()).tolist()


class RustMCTS:
    """Duck-types the parts of `chessgpu.mcts.MCTS` that `search_engine` uses.

    Not a general replacement: `analyse`, `play_batch` and the injectable
    `terminal` hook are not here, because the engine does not call them. Adding
    them silently would invite something to depend on a path with no identity
    test behind it.
    """

    def __init__(
        self,
        value_policy,
        policy=None,
        *,
        c_puct: float = 2.0,
        c_puct_base: float | None = 19652.0,
        c_puct_init: float = 1.25,
        simulations: int = 100_000,
        batch: int = 64,
        fpu: float = -0.2,
        reuse: bool = True,
        dedup: bool = False,
        mate_distance: bool = False,
        compile_nets: bool = False,
        pad_batches: bool = False,
    ) -> None:
        if policy is None:
            raise ValueError("the Rust core needs a policy net for priors")
        self.simulations = simulations
        self._evaluate = make_evaluator(
            policy, value_policy,
            compile_nets=compile_nets,
            # Pad to the batch size, so the shape the graph is captured for is
            # the one the search asks for most of the time.
            pad_to=batch if pad_batches else None,
        )
        self._core = core.Mcts(
            c_puct=c_puct,
            c_puct_base=c_puct_base,
            c_puct_init=c_puct_init,
            fpu=fpu,
            batch=batch,
            reuse=reuse,
            dedup=dedup,
            mate_distance=mate_distance,
        )
        self.evaluations = 0
        self.reused = 0

    def reset(self) -> None:
        """Between games, never within one."""
        self._core.reset()

    def _position(self, board: chess.Board) -> core.Position:
        """Rebuild the position WITH ITS HISTORY.

        The history is not optional. Threefold repetition depends on the moves
        before the root, so a `Position` built from a bare FEN would disagree with
        `rules.terminal_value` about draws -- and it would disagree silently, in
        the direction of claiming wins in drawn positions.

        Rebuilding from the root every move is O(plies) and costs microseconds in
        Rust; the alternative is holding a mutable Position across moves and
        trusting the engine and lichess to agree about what was played.
        """
        root = board.root()
        pos = core.Position(root.fen())
        for mv in board.move_stack:
            pos.push_uci(mv.uci())
        return pos

    def search(self, board: chess.Board, deadline: float | None = None,
               on_progress=None, progress_every: float = 0.15):
        """Search and return `(root_handle, {move: visits})`.

        `deadline` arrives as a `time.perf_counter()` absolute, matching
        `mcts.MCTS.search`; the Rust side wants a duration, so it is converted
        here. `on_progress` is accepted and IGNORED -- see the note in
        `search_engine` about what that costs the dashboard.
        """
        import time as _time

        max_seconds = None
        if deadline is not None:
            max_seconds = max(0.0, deadline - _time.perf_counter())

        pos = self._position(board)
        pairs = self._core.search(pos, self.simulations, self._evaluate, max_seconds)
        self.evaluations = self._core.evaluations
        self.reused = self._core.reused
        visits = {chess.Move.from_uci(u): v for u, v in pairs}
        return _Root(self._core, board), visits

    def play(self, board: chess.Board, deadline: float | None = None) -> chess.Move:
        _root, visits = self.search(board, deadline=deadline)
        if not visits:
            raise ValueError(f"no legal moves in {board.fen()}")
        return max(visits.items(), key=lambda kv: kv[1])[0]

    def report(self, root, board: chess.Board, top_n: int = 5, pv_max: int = 12) -> dict:
        """The same shape `mcts.MCTS.report` returns, for the telemetry."""
        c = self._core
        top = c.top(top_n)
        pv_uci = c.pv(pv_max)

        # SAN needs the position the move is played FROM, so walk a copy.
        pv_san, tmp = [], board.copy()
        for u in pv_uci:
            mv = chess.Move.from_uci(u)
            if mv not in tmp.legal_moves:
                break
            pv_san.append(tmp.san(mv))
            tmp.push(mv)

        return {
            "move": chess.Move.from_uci(top[0][0]) if top else None,
            "win_prob": c.root_q,
            "evaluations": c.evaluations,
            "reused": c.reused,
            "pv": pv_san,
            "top": [
                (board.san(chess.Move.from_uci(u)), v, round(q, 3), round(p, 3))
                for u, v, q, p in top
            ],
            "mate": tmp.is_checkmate(),
        }


class _Root:
    """Opaque handle. The Rust tree is not exposed node by node on purpose:
    reaching into it from Python would create a second consumer of a layout that
    only the identity test constrains."""

    __slots__ = ("_core", "_board")

    def __init__(self, core_obj, board):
        self._core, self._board = core_obj, board

    @property
    def q(self) -> float:
        return self._core.root_q


def select_mcts_class():
    """`RustMCTS` when `CHESSGPU_CORE=rust`, else the Python `MCTS`.

    Defaults to Python. The Rust path is opt-in until it has played rated games,
    because "provably identical" is a statement about the search and not about
    the plumbing around it.
    """
    from chessgpu.mcts import MCTS

    choice = os.environ.get("CHESSGPU_CORE", "python").strip().lower()
    if choice == "rust":
        return RustMCTS, "rust"
    if choice not in ("python", ""):
        raise ValueError(
            f"CHESSGPU_CORE={choice!r} is not 'rust' or 'python'. Failing rather "
            f"than falling back, because a typo that silently selects the old "
            f"path would make an A/B measure nothing."
        )
    return MCTS, "python"
