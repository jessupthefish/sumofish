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
    search as well as at the end of it, which is what the `sumofish` dashboard reads
    to show the tree developing live rather than only its conclusion.

Neither can block the search or take it down. See `sumofish/telemetry.py` for
why the channel is an append-only file rather than the socket or fifo it looks
like it ought to be.
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
import time
from pathlib import Path

import chess
import torch

from sumofish.engines.neural_engine import load_policy
from sumofish.hlgauss import HLGauss
from sumofish.mcts import MCTS
from sumofish.rust_mcts import select_mcts_class
from sumofish.model import ChessTransformer, ModelConfig
from sumofish.telemetry import Telemetry, perspective
from sumofish.uci import Limits, run
from sumofish.value_policy import ValuePolicy

ROOT = Path(__file__).resolve().parent.parent.parent

# Time control. Conservative on purpose: losing on time is a strictly worse
# outcome than playing a slightly weaker move, and lichess-bot already subtracts
# a move_overhead on top of this.
DIVISOR = 30.0          # spend ~1/30th of the remaining clock
INC_FRACTION = 0.7      # plus most of the increment, which is renewable
MIN_SECONDS = 0.05      # always look at something, when the clock can afford it
MAX_FRACTION = 0.33     # never spend more than a third of what is left
# SAN generation and telemetry emission in snapshot() still happen after the
# search returns, so "the whole remaining clock" is not a safe budget even in
# the extreme case -- this is a conservative estimate, not a measurement.
OVERHEAD_MARGIN = 0.02


def think_time(limits: Limits, side: chess.Color) -> float:
    if limits.movetime:
        return max(MIN_SECONDS, limits.movetime / 1000.0)
    remaining = limits.time_for(side)
    if remaining is None:
        return 1.0                      # no clock given (analysis GUI, tests)
    remaining_s = remaining / 1000.0
    inc_s = limits.inc_for(side) / 1000.0
    budget = remaining_s / DIVISOR + inc_s * INC_FRACTION
    budget = max(MIN_SECONDS, min(budget, remaining_s * MAX_FRACTION))
    # The floor above guarantees SOME look, but it must never win against the
    # clock itself. Found 2026-07-30: at remaining_s below MIN_SECONDS /
    # MAX_FRACTION (~0.152s), MIN_SECONDS can exceed MAX_FRACTION*remaining_s,
    # so the line above returns MORE time than is actually left -- e.g.
    # remaining_s=0.03 (30ms) produced budget=0.05 (50ms), directly
    # contradicting this module's own docstring ("never bet the game on one
    # move") and this bot plays live blitz/bullet where that scramble is a
    # real, repeated scenario, not a corner case. This is the hard, absolute
    # ceiling: never more than what's actually left, minus overhead margin.
    # Floored at a small epsilon rather than 0 so a call site never divides by
    # or waits on a literal zero.
    return max(0.001, min(budget, remaining_s - OVERHEAD_MARGIN))


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


# Recognized spellings for the four CHESSGPU_* search-quality/speed booleans
# below. Anything else raises, same reasoning as `select_mcts_class`'s own
# CHESSGPU_CORE check: a typo that silently resolves to some default makes an
# A/B measure nothing, and this class of flag has already cost -168 Elo once
# on a config nobody meant to ship (CHESSGPU_DEDUP+CHESSGPU_COMPILE together,
# 2026-07-29). The bug this replaces: `os.environ.get(VAR, "0") != "0"` treats
# every string except the literal "0" as true, so `CHESSGPU_VLOSS_FIX=false`
# -- a natural rollback edit -- silently turned the flag ON instead of off,
# and nothing at runtime (not the boot telemetry, not the dashboard) would
# have shown the mistake.
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def env_flag(name: str) -> bool:
    raw = os.environ.get(name, "0")
    norm = raw.strip().lower()
    if norm in _TRUE:
        return True
    if norm in _FALSE:
        return False
    raise ValueError(
        f"{name}={raw!r} is not a recognized boolean (use 1/0, true/false, "
        f"yes/no, or on/off). Failing rather than guessing, because a typo "
        f"that silently picks the wrong value would make an A/B measure "
        f"nothing -- see CHESSGPU_CORE's own check for the same reasoning."
    )


# The Python MCTS constructor takes none of these -- they're Rust-core-only
# search-quality/speed args (see the CHESSGPU_CORE branch below). Setting one
# without also setting CHESSGPU_CORE=rust was a silent no-op with nothing at
# runtime -- not boot telemetry, not the dashboard -- to show the mistake,
# the same invisible-failure shape as the root_q/top()/pv() bugs found
# elsewhere in this project the same night this was caught.
RUST_ONLY_FLAGS = ("CHESSGPU_DEDUP", "CHESSGPU_COMPILE",
                    "CHESSGPU_MATE_DISTANCE", "CHESSGPU_VLOSS_FIX")


def ignored_rust_flags(env: dict) -> list[str]:
    """Which of RUST_ONLY_FLAGS are set true in env but core is not rust.

    Pure over an explicit env mapping (not os.environ) so it's directly
    testable. Raises the same way env_flag does on an unrecognized spelling.
    """
    def flag(name: str) -> bool:
        raw = env.get(name, "0")
        norm = raw.strip().lower()
        if norm in _TRUE:
            return True
        if norm in _FALSE:
            return False
        raise ValueError(
            f"{name}={raw!r} is not a recognized boolean (use 1/0, true/false, "
            f"yes/no, or on/off)."
        )
    return [name for name in RUST_ONLY_FLAGS if flag(name)]


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
    # From the checkpoint's own config, never from a hardcoded preset name.
    # This said `build("9M", ...)` and the width was therefore assumed rather
    # than read: dropping any checkpoint that is not 9M into `runs/value.pt`
    # made `load_state_dict` raise on a shape mismatch, which kills the engine
    # at boot and leaves lichess-bot with nothing to play the game with. The
    # checkpoint swap is advertised all over CLAUDE.md as a file copy needing
    # no restart, and it would have been, right up until the first copy of a
    # differently-shaped net.
    # Filtered, because `ModelConfig(**cfg)` hard-binds every checkpoint ever
    # written to the current dataclass signature: rename or drop one field and
    # a TypeError kills the engine at boot for checkpoints that are otherwise
    # perfectly loadable. The hardcoded preset it replaced was wrong but at
    # least could not fail this way.
    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    vmodel = ChessTransformer(
        ModelConfig(**{k: v for k, v in ck["cfg"].items() if k in fields})
    )
    state = ck.get("ema") or ck["model"]
    vmodel.load_state_dict({k: v.float() for k, v in state.items()})
    value = ValuePolicy(vmodel, HLGauss(bins=bins), device="cuda:0")

    # High cap on purpose: the CLOCK should be what stops the search, not an
    # arbitrary simulation count. At ~0.5s per 800 simulations, a 12s budget is
    # worth roughly 20k, so a cap of 800 was throwing away 95% of the thinking
    # time it had been granted.
    sims = int(os.environ.get("CHESSGPU_SIMS", "100000"))
    batch = int(os.environ.get("CHESSGPU_BATCH", "64"))

    # Rust is the default core as of 2026-07-31 (`CHESSGPU_CORE=python` opts
    # back into the pre-port Python search). It produces byte-identical root
    # visit vectors -- verified with the real checkpoints in
    # `tests/identity_engine.py` -- so it plays the same moves, faster.
    # Measured 7.0x on a midgame position at 1600 simulations.
    #
    # Rollback is `CHESSGPU_CORE=python`. Nothing about the checkpoints, the
    # config or the unit changes, which is the point: an env var is a decision
    # a tired person can reverse at 3am.
    mcts_cls, core_name = select_mcts_class()
    if core_name == "rust":
        # Both speed flags default OFF, and that is a measured decision rather
        # than caution.
        #
        # `dedup` and `compile+pad` are each identity-preserving in the SEARCH --
        # verified byte-identical against sumofish.mcts with a deterministic mock.
        # Against the real networks they are not, and the reason is not a bug in
        # either: they change the number of rows in the forward pass, and the
        # network is not batch-shape invariant, because GPU float reductions
        # depend on shape. Measured at 400 simulations over 8 positions:
        #
        #   plain rust        8/8 byte-identical, 8/8 same move    3.6x
        #   dedup only        3/8 byte-identical, 8/8 same move    3.8x
        #   compile+pad only  1/8 byte-identical, 7/8 same move
        #   both              1/8 byte-identical, 7/8 same move    7.0x
        #
        # So plain rust is free: provably the same moves, 3.6x faster. The extra
        # 1.9x changes what the engine plays in roughly one position in eight, and
        # is therefore an Elo question, not a speed one. It is almost certainly
        # positive -- at a fixed clock it buys ~2x the simulations -- but "almost
        # certainly" is the reasoning this project's own philosophy forbids, and
        # the instrument to settle it now exists.
        #
        # `mate_distance` and `vloss_fix` are a different kind of flag: neither
        # changes the number of rows sent to the network, so neither is a speed
        # question at all. Both are search-QUALITY fixes (a mate-in-2 no longer
        # scores the same as a mate-in-14; virtual loss no longer corrupts Q) and
        # both are also off by default, for the same reason -- a defect fixed in
        # the tree still needs an Elo verdict from the pair-match harness before
        # it earns a default, exactly like the speed flags above.
        mcts = mcts_cls(
            value, policy=policy, simulations=sims, batch=batch,
            dedup=env_flag("CHESSGPU_DEDUP"),
            compile_nets=env_flag("CHESSGPU_COMPILE"),
            pad_batches=env_flag("CHESSGPU_COMPILE"),
            mate_distance=env_flag("CHESSGPU_MATE_DISTANCE"),
            vloss_fix=env_flag("CHESSGPU_VLOSS_FIX"),
        )
    else:
        # Warn on stderr, not stdout: this is a UCI engine and stdout is
        # protocol.
        ignored = ignored_rust_flags(os.environ)
        if ignored:
            print(
                f"WARNING: {', '.join(ignored)} set but CHESSGPU_CORE is not "
                f"'rust' -- these flags do nothing on the Python core and are "
                f"being silently ignored. Set CHESSGPU_CORE=rust or unset them.",
                file=sys.stderr,
            )
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
            "core": core_name,
            "params": pinfo["params"],
        },
        durable=True,
    )

    print(
        f"policy step={pinfo['step']} | value step={ck.get('step')} bins={bins} | "
        f"cap {sims} sims, batch {batch}, core={core_name} (clock-bound)",
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
        options=[("option name Simulations type string default 800", ""),
                 # Which game this process is playing. The engine does not use
                 # it and does not need to; it exists so the narration can say
                 # which of the two concurrent games each record came from,
                 # because both write to the same file. lichess-bot sends it
                 # from extra_game_handlers.py (patches/0003).
                 ("option name GameId type string default ", "")],
        on_option=_option_handler(mcts, tele),
    )


def _option_handler(mcts, tele):
    """UCI options this engine honours. Anything else is declined, loudly."""
    def handle(name: str, value: str) -> bool:
        key = name.lower()
        if key == "simulations" and value.isdigit():
            mcts.simulations = max(1, int(value))
            return True
        if key == "gameid":
            # Stamped on every record from here on. Empty means "no idea",
            # which is what a hand-run engine or an older lichess-bot gives.
            tele.game = value.strip() or None
            return True
        return False
    return handle


if __name__ == "__main__":
    main()
