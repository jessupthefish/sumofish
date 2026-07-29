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

Four checks, in the order a deployment fails:

  1. It loads at all, through the real engine entry point, not a test harness.
  2. It answers `uci` and reaches `uciok`.
  3. Its throughput at the deployed time budget clears a floor. A net that is
     better per node and half the speed can be worse on a clock, and the bot
     is clock-bound.
  4. Over a handful of real moves from real positions it returns only legal
     moves and never overruns its budget.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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

    # 1 and 2, through the real entry point. Importing the module would miss
    # exactly the failure mode that took the bot down: an error on a code path
    # only main() reaches.
    print(f"  checkpoint {path}")
    proc = subprocess.run(
        [str(ROOT / ".venv/bin/python"), "-m", "chessgpu.engines.search_engine"],
        input="uci\nquit\n", capture_output=True, text=True, cwd=ROOT,
        env={**__import__("os").environ, "CHESSGPU_VALUE": str(path),
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

    # 3 and 4 in-process, because they need the search, not the protocol.
    import torch

    from chessgpu.engines.neural_engine import load_policy
    from chessgpu.hlgauss import HLGauss
    from chessgpu.mcts import MCTS
    from chessgpu.model import ChessTransformer, ModelConfig
    from chessgpu.value_policy import ValuePolicy

    ck = torch.load(path, map_location="cuda:0", weights_only=False)
    fields = {f.name for f in __import__("dataclasses").fields(ModelConfig)}
    model = ChessTransformer(
        ModelConfig(**{k: v for k, v in ck["cfg"].items() if k in fields}))
    model.load_state_dict(
        {k: v.float() for k, v in (ck.get("ema") or ck["model"]).items()})
    value = ValuePolicy(model, HLGauss(bins=ck["cfg"]["output_size"]), device="cuda:0")
    policy, _ = load_policy(str(ROOT / "runs/policy.pt"))
    mcts = MCTS(value, policy=policy, simulations=10**9, batch=64)

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

    nps = sum(rates) / len(rates) if rates else 0.0
    print(f"  [{'ok' if nps >= args.min_nps else 'FAIL'}] "
          f"{nps:.0f} nodes/s at a {args.seconds}s budget (floor {args.min_nps:.0f})")
    if nps < args.min_nps:
        failures.append(f"too slow to deploy: {nps:.0f} nps")
    if not failures:
        print(f"  [ok] {len(POSITIONS)} positions, all legal, none overran")
        print(f"  {model.num_parameters():,} parameters, step {ck.get('step')}")
        print("PASS")
        return 0
    print("FAIL:")
    for f in failures:
        print(f"    {f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
