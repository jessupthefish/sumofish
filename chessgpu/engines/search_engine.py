"""SumoFish with search. This is the engine that thinks ahead.

Two networks, both consulted on every move:

    policy net  (behavioural cloning)  which moves deserve simulations
    value net   (state value)          how good is the position we land in

That is AlphaZero's arrangement -- it uses one network with two heads, we
happen to have two networks -- and it is why the first training run is not
superseded by the second.

    bin/sumofish-search
    CHESSGPU_POLICY=... CHESSGPU_VALUE=... CHESSGPU_SIMS=800 bin/sumofish-search

## Time management

A searchless engine has nothing to manage: one forward pass, fixed cost,
answer. A searching engine has to decide how long to think, which is a real
decision with a real failure mode at both ends -- think too long and you flag,
think too briefly and you throw away the entire advantage of searching.

The scheme here is deliberately simple and conservative:

    budget = remaining / DIVISOR + increment * INC_FRACTION

Spending a fixed fraction of what remains is self-correcting: as the clock runs
down the budget shrinks with it, so you cannot flag by arithmetic. The
increment is nearly free to spend because it comes back every move. Everything
is then clamped to a floor (always look at something) and to a hard ceiling of
a third of the remaining clock (never bet the game on one move).

## Narration

The engine reports what it is thinking twice over, to two different audiences:

  * **UCI `info` lines on stderr**, the 1998 protocol every chess GUI already
    speaks. lichess picks the score up and `!eval` in chat answers from it.
    This happens whether or not anything local is watching.
  * **JSON-lines telemetry** to `logs/engine.jsonl`, emitted *during* the
    search as well as at the end of it, which is what `scripts/watch.py` reads
    to show the tree developing live rather than only its conclusion.

Neither can block the search or take it down. See `chessgpu/telemetry.py` for
why the channel is an append-only file rather than the socket or fifo it looks
like it ought to be.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import chess
import torch

from chessgpu.engines.neural_engine import load_policy
from chessgpu.hlgauss import HLGauss
from chessgpu.mcts import MCTS
from chessgpu.model import build
from chessgpu.telemetry import Telemetry, perspective
from chessgpu.uci import Limits, run
from chessgpu.value_policy import ValuePolicy

ROOT = Path(__file__).resolve().parent.parent.parent

# Time control. Conservative on purpose: losing on time is a strictly worse
# outcome than playing a slightly weaker move, and lichess-bot already subtracts
# a move_overhead on top of this.
DIVISOR = 30.0          # spend ~1/30th of the remaining clock
INC_FRACTION = 0.7      # plus most of the increment, which is renewable
MIN_SECONDS = 0.05      # always look at something
MAX_FRACTION = 0.33     # never spend more than a third of what is left


def think_time(limits: Limits, side: chess.Color) -> float:
    if limits.movetime:
        return max(MIN_SECONDS, limits.movetime / 1000.0)
    remaining = limits.time_for(side)
    if remaining is None:
        return 1.0                      # no clock given (analysis GUI, tests)
    remaining_s = remaining / 1000.0
    inc_s = limits.inc_for(side) / 1000.0
    budget = remaining_s / DIVISOR + inc_s * INC_FRACTION
    return max(MIN_SECONDS, min(budget, remaining_s * MAX_FRACTION))


def centipawns(win_prob: float) -> int:
    """Re-express a win probability on the conventional centipawn scale.

    This engine has no centipawn scale of its own -- it predicts P(win), not
    material -- so this is a presentation choice, not a measurement. The
    mapping is the Elo logistic every rating system already uses,
    `p = 1 / (1 + 10^(-cp/400))`, inverted. It is monotone, so it can never
    reorder anything, and it puts the number in the range a chess player reads
    without having to think about it.
    """
    p = min(max(win_prob, 1e-4), 1.0 - 1e-4)
    return int(round(400.0 * math.log10(p / (1.0 - p))))


def main() -> None:
    policy_path = os.environ.get("CHESSGPU_POLICY", str(ROOT / "runs/policy.pt"))
    value_path = os.environ.get("CHESSGPU_VALUE", str(ROOT / "runs/value.pt"))
    for label, path in (("policy", policy_path), ("value", value_path)):
        if not Path(path).exists():
            print(f"{label} checkpoint not found: {path}", file=sys.stderr)
            sys.exit(1)

    policy, pinfo = load_policy(policy_path)

    ck = torch.load(value_path, map_location="cuda:0", weights_only=False)
    bins = ck["cfg"]["output_size"]
    vmodel = build("9M", output_size=bins, causal=ck["cfg"]["causal"])
    state = ck.get("ema") or ck["model"]
    vmodel.load_state_dict({k: v.float() for k, v in state.items()})
    value = ValuePolicy(vmodel, HLGauss(bins=bins), device="cuda:0")

    # High cap on purpose: the CLOCK should be what stops the search, not an
    # arbitrary simulation count. At ~0.5s per 800 simulations, a 12s budget is
    # worth roughly 20k, so a cap of 800 was throwing away 95% of the thinking
    # time it had been granted.
    sims = int(os.environ.get("CHESSGPU_SIMS", "100000"))
    batch = int(os.environ.get("CHESSGPU_BATCH", "64"))
    mcts = MCTS(value, policy=policy, simulations=sims, batch=batch)

    tele = Telemetry(
        os.environ.get("CHESSGPU_TELEMETRY", str(ROOT / "logs/engine.jsonl"))
    )
    tele.emit(
        {
            "ev": "boot",
            "policy_step": pinfo["step"],
            "value_step": ck.get("step"),
            "bins": bins,
            "sims": sims,
            "batch": batch,
            "params": pinfo["params"],
        },
        durable=True,
    )

    print(
        f"policy step={pinfo['step']} | value step={ck.get('step')} bins={bins} | "
        f"cap {sims} sims, batch {batch} (clock-bound)",
        file=sys.stderr,
    )

    # Warm the kernels during the handshake, not on move one with a running clock.
    mcts.play(chess.Board(), deadline=time.perf_counter() + 2.0)

    def choose(board: chess.Board, limits: Limits) -> chess.Move:
        budget = think_time(limits, board.turn)
        start = time.perf_counter()
        ply = board.ply()
        fen = board.fen()
        white_to_move = board.turn == chess.WHITE

        def snapshot(root, done: int, final: bool) -> dict:
            elapsed = time.perf_counter() - start
            # Tighter caps mid-search: `report` generates SAN, which costs a
            # legal-move generation per move, and this runs on the clock.
            r = mcts.report(
                root, board,
                top_n=6 if final else 5,
                pv_max=14 if final else 8,
            )
            wp = r["win_prob"]
            return {
                "ev": "move" if final else "think",
                "ply": ply,
                "fen": fen,
                "stm": "w" if white_to_move else "b",
                # Both frames, deliberately. Q is side-to-move relative, and a
                # consumer that forgets to flip it plots a sawtooth.
                "wp": round(wp, 4),
                "wp_white": round(perspective(wp, white_to_move), 4),
                "nodes": r["evaluations"],
                "nps": int(r["evaluations"] / elapsed) if elapsed > 0 else 0,
                "sims": done,
                "elapsed": round(elapsed, 3),
                "budget": round(budget, 3),
                "best": r["top"][0][0] if r["top"] else None,
                "pv": r["pv"],
                "top": r["top"],
                "mate": r["mate"],
                "done": final,
            }

        # No observer, no work: `snapshot` generates SAN and walks the PV, and
        # that happens on the clock. With telemetry off this is exactly the
        # search that ran before any of this was added.
        root, visits = mcts.search(
            board,
            deadline=start + budget,
            on_progress=(lambda node, done: tele.emit(snapshot(node, done, False)))
            if tele.enabled else None,
        )
        if not visits:
            raise ValueError(f"no legal moves in {fen}")
        # Most-visited, not best-average. A child can luck into a high average
        # from two visits; it only accumulates visits by surviving PUCT
        # repeatedly.
        move = max(visits.items(), key=lambda kv: kv[1])[0]

        elapsed = time.perf_counter() - start
        final = snapshot(root, sum(visits.values()), final=True)
        final["uci"] = move.uci()
        tele.emit(final, durable=True)

        # `depth` is the length of the line the search actually believes in,
        # which is the honest MCTS analogue of alpha-beta's depth. There is no
        # `score mate N` because nothing in this engine computes a mate
        # distance; the telemetry carries a plain boolean instead.
        print(
            f"info depth {max(1, len(final['pv']))} score cp {centipawns(final['wp'])} "
            f"nodes {final['nodes']} nps {final['nps']} "
            f"time {int(elapsed * 1000)} pv {move.uci()}",
            file=sys.stderr,
            flush=True,
        )
        # This line predates the one above and stays because it carries the two
        # numbers UCI has no field for: the time budget this move was granted,
        # and how much of it the search actually used.
        print(
            f"info string budget {budget:.2f}s used {elapsed:.2f}s "
            f"evals {final['nodes']} wp {final['wp']:.3f}",
            file=sys.stderr,
            flush=True,
        )
        return move

    run(
        choose,
        name=f"SumoFish {pinfo['params']//1_000_000}M search",
        author="Jessupthefish",
        options=[("option name Simulations type string default 800", "")],
        on_option=lambda n, v: (
            setattr(mcts, "simulations", max(1, int(v))) or True
        ) if n.lower() == "simulations" and v.isdigit() else False,
    )


if __name__ == "__main__":
    main()
