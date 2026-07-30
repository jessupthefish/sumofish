#!/usr/bin/env python
"""Step 0's cost sanity check: is cross-game batching worth building?

`STATE.md` item 7 asserts the engine is "launch-bound below ~128 rows -- one
row costs the same 7.2ms as 128", but that is a per-call latency observation,
never measured end-to-end in games/GPU-hour. This script measures it directly:
N games played ONE AT A TIME (today's actual `scripts/match.py` behaviour --
confirmed by reading it: fully sequential, no concurrency anywhere) against
the same N games played CONCURRENTLY, sharing one batched evaluator across
their `evaluate()` callbacks.

This is a payoff-SIZING experiment, not a strength test: both arms use the
SAME nets on both sides (self-play), a small fixed simulation count, and a
hard ply cap, purely to make games finish fast enough to sample cheaply.
Nothing here produces or should produce an Elo number.

# The batching mechanism, and why it does not already exist

`CHESSGPU_BATCH` (see `chessgpu/engines/search_engine.py`) controls how many
LEAVES from one game's own tree are collected before one forward pass -- it
is entirely internal to a single `core.Mcts` instance, 1:1 with one game.
Nothing in the repo batches rows from DIFFERENT games' searches together
(grepped for ThreadPool/asyncio/Queue-based batching near game-running code:
nothing). `CrossGameBatcher` below is the minimal new plumbing: N games each
run in their own thread, blocked inside their own `core.Mcts.search()` call;
each game's `evaluate()` callback is monkey-patched to submit into a shared
queue instead of calling the model directly, and a small flusher batches
whatever has accumulated once every active game has submitted (the common
case) or a short timeout elapses (so a game that finishes early, or is
slower than its peers, cannot starve the others forever).

Usage:
    scripts/batch_payoff.py --games 24 --concurrency 8 --sims 100 --max-plies 60
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))

from chessgpu.rust_mcts import RustMCTS, make_evaluator  # noqa: E402
from scripts.match import load_prior, load_value  # noqa: E402


class CrossGameBatcher:
    """Collects `evaluate()` calls from several concurrent games and runs one
    forward pass over their union, then hands each caller back its own slice.

    Not production code -- see the module docstring. Correct enough for a
    sizing experiment: a background flusher bounds worst-case latency, and an
    immediate flush fires as soon as every STILL-PLAYING game has a request
    queued, which is the common case once the games are running steadily.
    """

    def __init__(self, policy, value_policy, active_games: int, max_wait: float = 0.02):
        self._raw = make_evaluator(policy, value_policy)
        self.active = active_games
        self.max_wait = max_wait
        self.lock = threading.Lock()
        self.pending: list[tuple[list[str], list[list[int]], dict]] = []
        self._stop = False
        self.rows_sent = 0
        self.forward_passes = 0
        self.flusher = threading.Thread(target=self._flush_loop, daemon=True)
        self.flusher.start()

    def game_finished(self) -> None:
        with self.lock:
            self.active -= 1
            if self.active > 0 and self.pending and len(self.pending) >= self.active:
                self._flush_locked()

    def stop(self) -> None:
        self._stop = True

    def _flush_loop(self) -> None:
        while not self._stop:
            time.sleep(self.max_wait)
            with self.lock:
                if self.pending:
                    self._flush_locked()

    def _flush_locked(self) -> None:
        """Caller must hold `self.lock`."""
        batch = self.pending
        self.pending = []
        fens: list[str] = []
        actions: list[list[int]] = []
        spans = []
        for f, a, holder in batch:
            start = len(fens)
            fens.extend(f)
            actions.extend(a)
            spans.append((start, len(fens), holder))
        priors, values = self._raw(fens, actions)
        self.rows_sent += len(fens)
        self.forward_passes += 1
        for start, end, holder in spans:
            holder["result"] = (priors[start:end], values[start:end])
            holder["event"].set()

    def evaluate(self, fens: list[str], actions: list[list[int]]):
        holder = {"event": threading.Event()}
        with self.lock:
            self.pending.append((fens, actions, holder))
            if len(self.pending) >= self.active:
                self._flush_locked()
        holder["event"].wait()
        return holder["result"]


def play_one_game(rmcts: RustMCTS, max_plies: int) -> int:
    board = chess.Board()
    rmcts.reset()
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        move = rmcts.play(board)
        board.push(move)
        plies += 1
    return plies


def run_unbatched(policy, value_policy, sims: int, games: int, max_plies: int) -> float:
    """Today's reality: one game at a time, exactly as `match.py` does it."""
    start = time.perf_counter()
    for _ in range(games):
        rmcts = RustMCTS(value_policy, policy=policy, simulations=sims, batch=32)
        play_one_game(rmcts, max_plies)
    return time.perf_counter() - start


def run_batched(policy, value_policy, sims: int, games: int, concurrency: int,
                 max_plies: int) -> tuple[float, int, int]:
    """`concurrency` games at once, sharing one `CrossGameBatcher`."""
    start = time.perf_counter()
    remaining = games
    total_forward_passes = 0
    total_rows = 0
    while remaining > 0:
        wave = min(concurrency, remaining)
        batcher = CrossGameBatcher(policy, value_policy, active_games=wave)
        players = []
        for _ in range(wave):
            rmcts = RustMCTS(value_policy, policy=policy, simulations=sims, batch=32)
            rmcts._evaluate = batcher.evaluate
            players.append(rmcts)

        def worker(rmcts):
            try:
                play_one_game(rmcts, max_plies)
            finally:
                batcher.game_finished()

        threads = [threading.Thread(target=worker, args=(p,)) for p in players]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        batcher.stop()
        total_forward_passes += batcher.forward_passes
        total_rows += batcher.rows_sent
        remaining -= wave
    return time.perf_counter() - start, total_forward_passes, total_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--value", default=str(ROOT / "runs/value.pt"))
    ap.add_argument("--policy", default=str(ROOT / "runs/policy.pt"))
    ap.add_argument("--games", type=int, default=24, help="per arm")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--max-plies", type=int, default=60)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print(f"loading nets: value={args.value} policy={args.policy}")
    value_policy = load_value(args.value, args.device)
    policy = load_prior(args.policy, args.device)

    print(f"\n=== unbatched: {args.games} games, one at a time (today's reality) ===")
    t_unbatched = run_unbatched(policy, value_policy, args.sims, args.games, args.max_plies)
    unbatched_gph = args.games / (t_unbatched / 3600)
    print(f"  {t_unbatched:.1f}s total, {t_unbatched / args.games:.2f}s/game, "
          f"{unbatched_gph:.1f} games/hour")

    print(f"\n=== batched: {args.games} games, concurrency={args.concurrency} ===")
    t_batched, passes, rows = run_batched(
        policy, value_policy, args.sims, args.games, args.concurrency, args.max_plies
    )
    batched_gph = args.games / (t_batched / 3600)
    print(f"  {t_batched:.1f}s total, {t_batched / args.games:.2f}s/game, "
          f"{batched_gph:.1f} games/hour")
    print(f"  {passes} forward passes, {rows} rows sent, "
          f"{rows / max(passes, 1):.1f} rows/pass average")

    mult = batched_gph / unbatched_gph
    print(f"\nMULTIPLIER: {mult:.2f}x games/hour, batched vs unbatched")
    print("(This is a WALL-CLOCK ratio on a shared, contended GPU if anything")
    print(" else is running -- check what else had the GPU during this run")
    print(" before treating the absolute games/hour numbers as clean.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
