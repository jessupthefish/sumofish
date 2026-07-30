#!/usr/bin/env python
"""Does this checkpoint actually work as the live engine, on a real clock?

    scripts/smoke.py runs/136M-sv/best.pt

Exit 0 if it is safe to deploy, non-zero with a reason if it is not.

A match tells you a checkpoint is *stronger*. It does not tell you the engine
boots with it, that its nodes-per-second survives the clock the bot plays on,
or that it never returns something illegal -- and a statistical gate cannot,
because every one of those failures happens outside the game record.

That gap is not theoretical. This session took the live bot down for ninety
minutes with a NameError in the value loader, and separately the loader
hardcoded the 9M preset, so the first non-9M checkpoint copied into
`runs/value.pt` would have killed the engine at boot on a rated account at 3am.
Both were invisible to every match ever played, and both are caught here in
about a minute.

Five checks, in the order a deployment fails:

  1. It loads at all, through the real engine entry point, not a test harness.
  2. It answers `uci` and reaches `uciok`.
  3. Its throughput at the deployed time budget clears a floor. A net that is
     better per node and half the speed can be worse on a clock, and the bot
     is clock-bound.
  4. Over a handful of real moves from real positions it returns only legal
     moves and never overruns its budget.
  5. `mcts.report()` -- the exact call `search_engine.py::choose()` makes to
     turn a finished search into a move -- runs clean on every position. Added
     2026-07-30: `root_q`/`top()`/`pv()` were all missing from the Rust
     binding this method reads, entirely invisible to checks 1-4 (which only
     ever called `.search()`, never `.report()`), and the live bot played
     dozens of rated games falling back to "the first legal move" before a
     human watching the dashboard caught it. This check exists so that
     specific failure mode cannot happen silently again.

Checks 1, 3, 4 and 5 all run under whatever `CHESSGPU_CORE`/`CHESSGPU_DEDUP`/
`CHESSGPU_COMPILE`/`CHESSGPU_MATE_DISTANCE`/`CHESSGPU_VLOSS_FIX` the live bot
unit (`systemd/chess-gpu-bot.service`) actually has set, read fresh from that
file rather than hardcoded here -- so this gate tracks whatever is really
deployed instead of quietly drifting out of sync with it the way the Python
core / `.search()`-only version did.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

UNIT = ROOT / "systemd" / "chess-gpu-bot.service"


def deployed_env() -> dict[str, str]:
    """The CHESSGPU_* flags the live bot unit actually sets, parsed from the
    unit file itself rather than duplicated here by hand -- the whole reason
    this gate used to test a different engine than production is that the
    Python-core defaults were hardcoded once and never touched again while
    the unit moved on. Missing file or no matches means "nothing set",
    which is also the live bot's own default posture if the unit vanished.
    """
    if not UNIT.exists():
        return {}
    env = {}
    for line in UNIT.read_text().splitlines():
        m = re.match(r"^Environment=(CHESSGPU_\w+)=(.*)$", line.strip())
        if m:
            env[m.group(1)] = m.group(2)
    return env

# Positions with different characters: opening, a sharp middlegame, an endgame.
# A checkpoint that only breaks in one phase is exactly the sort this catches.
POSITIONS = [
    chess.STARTING_FEN,
    "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9",
    "8/5pk1/6p1/8/4K3/6P1/5P2/8 w - - 0 40",
    "r2q1rk1/pp1nbppp/2p1pn2/3p4/2PP4/2N1PN2/PPQ1BPPP/R1B2RK1 b - - 0 9",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--seconds", type=float, default=3.0,
                    help="per-move budget to test at")
    ap.add_argument("--min-nps", type=float, default=300.0,
                    help="floor on unique nodes per second. The 9M does ~2000 "
                         "at the deployed budget; a candidate under this is "
                         "too slow to search meaningfully on a clock")
    args = ap.parse_args()

    path = Path(args.checkpoint)
    if not path.exists():
        print(f"FAIL: no checkpoint at {path}")
        return 1

    live = deployed_env()
    if live:
        print(f"  testing under the deployed config: {live}")
    else:
        print(f"  WARNING: {UNIT} not found or has no CHESSGPU_* lines; "
              f"testing under the Python core default, which is NOT "
              f"necessarily what's live")

    # 1 and 2, through the real entry point. Importing the module would miss
    # exactly the failure mode that took the bot down: an error on a code path
    # only main() reaches.
    print(f"  checkpoint {path}")
    proc = subprocess.run(
        [str(ROOT / ".venv/bin/python"), "-m", "chessgpu.engines.search_engine"],
        input="uci\nquit\n", capture_output=True, text=True, cwd=ROOT,
        env={**os.environ, **live, "CHESSGPU_VALUE": str(path),
             "CHESSGPU_TELEMETRY": ""},
        timeout=300,
    )
    if "uciok" not in proc.stdout:
        tail = (proc.stderr or "").strip().splitlines()[-4:]
        print("FAIL: the engine did not reach uciok with this checkpoint")
        for line in tail:
            print(f"    {line}")
        return 1
    print("  [ok] boots and answers uci")

    # 3, 4 and 5 in-process, because they need the search, not the protocol.
    # Apply the deployed env BEFORE importing anything that reads it at import
    # or construction time, so this process makes the same core/flag choice
    # `search_engine.py::main()` would -- `select_mcts_class()` reads
    # `CHESSGPU_CORE` from `os.environ` directly, not from an argument, so it
    # has to see it here.
    os.environ.update(live)

    import torch

    from chessgpu.engines.neural_engine import load_policy
    from chessgpu.engines.search_engine import env_flag
    from chessgpu.hlgauss import HLGauss
    from chessgpu.model import ChessTransformer, ModelConfig
    from chessgpu.rust_mcts import select_mcts_class
    from chessgpu.value_policy import ValuePolicy

    ck = torch.load(path, map_location="cuda:0", weights_only=False)
    fields = {f.name for f in __import__("dataclasses").fields(ModelConfig)}
    model = ChessTransformer(
        ModelConfig(**{k: v for k, v in ck["cfg"].items() if k in fields}))
    model.load_state_dict(
        {k: v.float() for k, v in (ck.get("ema") or ck["model"]).items()})
    value = ValuePolicy(model, HLGauss(bins=ck["cfg"]["output_size"]), device="cuda:0")
    policy, _ = load_policy(str(ROOT / "runs/policy.pt"))

    # The same branch `search_engine.py::main()` takes, on the same flags, so
    # this constructs whichever engine is actually deployed rather than
    # always the Python one. Duplicated rather than imported from
    # `search_engine.py` because that module's construction lives inline in
    # `main()`, not factored out -- refactoring the live bot's own entry
    # point is a bigger, riskier change than this gate warrants tonight.
    mcts_cls, core_name = select_mcts_class()
    if core_name == "rust":
        mcts = mcts_cls(
            value, policy=policy, simulations=10**9, batch=64,
            dedup=env_flag("CHESSGPU_DEDUP"),
            compile_nets=env_flag("CHESSGPU_COMPILE"),
            pad_batches=env_flag("CHESSGPU_COMPILE"),
            mate_distance=env_flag("CHESSGPU_MATE_DISTANCE"),
            vloss_fix=env_flag("CHESSGPU_VLOSS_FIX"),
        )
    else:
        mcts = mcts_cls(value, policy=policy, simulations=10**9, batch=64)
    print(f"  testing the {core_name} core (this is what {UNIT.name} actually runs)")

    failures = []
    rates = []
    for fen in POSITIONS:
        board = chess.Board(fen)
        mcts.reset()
        start = time.perf_counter()
        root, visits = mcts.search(board, deadline=start + args.seconds)
        elapsed = time.perf_counter() - start
        if not visits:
            failures.append(f"{fen}: no move returned")
            continue
        move = max(visits.items(), key=lambda kv: kv[1])[0]
        if move not in board.legal_moves:
            failures.append(f"{fen}: illegal move {move.uci()}")
        # Generous: the search checks the clock between batches, so overshoot
        # is bounded by one batch. Anything past 1.5x is a real problem.
        if elapsed > args.seconds * 1.5:
            failures.append(f"{fen}: overran its budget, {elapsed:.1f}s of {args.seconds}s")
        rates.append(mcts.evaluations / elapsed)

        # Check 5: the exact call `choose()` -> `snapshot()` makes to turn this
        # search into a move. `.search()` succeeding says nothing about this --
        # it's what stayed unreachable, and therefore untested, right up until
        # a live game hit it. A raise here is exactly tonight's failure mode.
        try:
            mcts.report(root, board, top_n=5, pv_max=12)
        except Exception as exc:  # noqa: BLE001 -- any exception here is the finding
            failures.append(f"{fen}: mcts.report() raised {exc!r}")

    nps = sum(rates) / len(rates) if rates else 0.0
    print(f"  [{'ok' if nps >= args.min_nps else 'FAIL'}] "
          f"{nps:.0f} nodes/s at a {args.seconds}s budget (floor {args.min_nps:.0f})")
    if nps < args.min_nps:
        failures.append(f"too slow to deploy: {nps:.0f} nps")
    if not failures:
        print(f"  [ok] {len(POSITIONS)} positions, all legal, none overran, "
              f"report() clean on every one")
        print(f"  {model.num_parameters():,} parameters, step {ck.get('step')}")
        print("PASS")
        return 0
    print("FAIL:")
    for f in failures:
        print(f"    {f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
