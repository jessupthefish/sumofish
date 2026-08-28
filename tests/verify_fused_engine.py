#!/usr/bin/env python
"""Does the engine actually run a fused checkpoint, and run it correctly?

A fused net fails three ways that all LOOK like a working engine, so a smoke
test that only asks "does it move" passes every one of them. This asserts the
three directly, against the real checkpoint, with a hand-computed reference:

1. **The value head reads the value columns.** `HLGauss(bins=cfg.output_size)`
   is right for a value net and gives 2032 on a fused one. That one raises, so
   it is the harmless failure -- asserted anyway, because the fix for it is a
   slice and a slice can be silently dropped.

2. **The priors read the policy columns.** `NeuralPolicy` indexes move ids from
   column 0. On a fused net the moves start at column `bins`, so every prior
   would belong to the move 64 places earlier in the action space. The search
   still masks to legal moves and still returns one, so nothing anywhere
   reports this; it just plays worse.

3. **One module, therefore one forward.** `rust_mcts` collapses two forward
   passes into one only when the two wrappers are backed by the same object.
   Load the file twice and the engine pays the FULL fused-width forward twice
   per node, which is slower than the two-net engine the fusion exists to beat.
   The whole case for d=384 is 1.14x parameters at 1.15x speed; lose the
   collapse and the speed half of that case inverts.

Run: tests/verify_fused_engine.py [checkpoint]
"""

from __future__ import annotations

import sys
from pathlib import Path

import chess
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sumofish.engines.loader import build_model, fused_bins, load_nets, read  # noqa: E402
from sumofish.tokenizer import MOVE_TO_ACTION, NUM_ACTIONS, tokenize_board  # noqa: E402

# Real positions, not `torch.randint` token ids. The growth retraction of
# 2026-08-24 happened because a measurement was taken on boards that cannot
# exist, and a slicing bug is exactly the kind of thing random tokens hide.
POSITIONS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9",
    "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
]

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def main() -> int:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "runs/19M-fused/best.pt")
    if not Path(ckpt).exists():
        print(f"SKIP: no checkpoint at {ckpt}")
        return 0

    ck = read(ckpt, DEVICE)
    bins = fused_bins(ck)
    if bins is None:
        fail(f"{ckpt} is not a fused checkpoint (output_size "
             f"{ck['cfg']['output_size']}); nothing here applies to it")
    print(f"fused checkpoint: {bins} value bins + {NUM_ACTIONS} moves "
          f"= {ck['cfg']['output_size']} columns, step {ck.get('step')}")

    # An INDEPENDENT reference model, built from the same file by a different
    # path than the engine's, so a bug in `build_model` cannot cancel itself out.
    ref = build_model(ck, DEVICE)
    boards = [chess.Board(f) for f in POSITIONS]
    tokens = torch.from_numpy(
        np.stack([tokenize_board(b) for b in boards])).long().to(DEVICE)
    with torch.inference_mode(), torch.autocast(DEVICE.split(":")[0],
                                                dtype=torch.bfloat16):
        raw = ref(tokens).float()
    if raw.shape[1] != bins + NUM_ACTIONS:
        fail(f"reference forward is {raw.shape[1]} wide, expected "
             f"{bins + NUM_ACTIONS}")

    value, policy, info = load_nets(ckpt, None, device=DEVICE)

    # --- 3. one module ---
    if value.model is not policy.model:
        fail("the value and policy wrappers hold DIFFERENT module objects; "
             "rust_mcts will not collapse the two forward passes")
    print("one module backs both heads: ok")

    # --- 1. the value head reads columns [:bins] ---
    if value.hl.bins != bins:
        fail(f"HLGauss got {value.hl.bins} bins, the checkpoint has {bins}")
    want_v = value.hl.expectation(raw[:, :bins]).cpu().numpy()
    got_v, _ = value.evaluate(boards)
    if not np.allclose(got_v, want_v, atol=1e-5):
        fail(f"value head disagrees with a hand-sliced reference:\n"
             f"  engine    {got_v}\n  reference {want_v}")
    print(f"value head matches the [:{bins}] slice on {len(boards)} positions: "
          f"max|delta| {np.max(np.abs(got_v - want_v)):.3e}")

    # --- 2. the priors read columns [bins:] ---
    if getattr(policy, "offset", 0) != bins:
        fail(f"policy offset is {getattr(policy, 'offset', 0)}, expected {bins}")
    got_p = policy._logprobs(boards).cpu().numpy()
    want_p = raw[:, bins:].cpu().numpy()
    if got_p.shape != want_p.shape:
        fail(f"prior logits are {got_p.shape}, expected {want_p.shape}")
    if not np.allclose(got_p, want_p, atol=1e-5):
        fail("priors disagree with a hand-sliced reference")
    print(f"priors match the [{bins}:] slice: max|delta| "
          f"{np.max(np.abs(got_p - want_p)):.3e}")

    # The slice being off by `bins` is the failure that plays legal moves and
    # loses quietly, so assert the difference is DETECTABLE here: if reading
    # from column 0 gave the same best move, this test would prove nothing.
    disagreements = 0
    for i, b in enumerate(boards):
        legal = list(b.legal_moves)
        acts = [MOVE_TO_ACTION.get(m.uci(), -1) for m in legal]
        acts = [a for a in acts if a >= 0]
        if not acts:
            continue
        right = int(np.argmax([want_p[i][a] for a in acts]))
        wrong_row = raw[i, :NUM_ACTIONS].cpu().numpy()   # the off-by-bins read
        wrong = int(np.argmax([wrong_row[a] for a in acts]))
        disagreements += right != wrong
    if disagreements == 0:
        fail("reading from column 0 picks the same move everywhere, so this "
             "test cannot see the bug it exists to catch; add positions")
    print(f"the off-by-{bins} read picks a different move on "
          f"{disagreements}/{len(boards)} positions, so the check has teeth")

    # --- the engine takes the fused path, in the core that actually ships ---
    if DEVICE == "cpu":
        print("SKIP: no CUDA, not building the search")
        print("PASS (partial: no GPU)")
        return 0
    from sumofish.rust_mcts import RustMCTS

    mcts = RustMCTS(value, policy=policy, simulations=64, batch=16)
    if not mcts.fused:
        fail("RustMCTS did not take the fused path; it is running two "
             "full-width forwards per node")
    move = mcts.play(chess.Board())
    if move not in chess.Board().legal_moves:
        fail(f"fused search returned an illegal move: {move}")
    print(f"RustMCTS.fused = True, and it plays {move} from the start position")

    # A two-net engine must NOT claim fusion, or the flag means nothing.
    if (ROOT / "runs/value.pt").exists() and (ROOT / "runs/policy.pt").exists():
        v2, p2, i2 = load_nets(ROOT / "runs/value.pt", ROOT / "runs/policy.pt",
                               device=DEVICE)
        if i2["fused"]:
            fail("the deployed two-net pair reports itself fused")
        if RustMCTS(v2, policy=p2, simulations=8, batch=8).fused:
            fail("a two-net engine took the fused path")
        print("the deployed two-net pair still reports fused = False")

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
