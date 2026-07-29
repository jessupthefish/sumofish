#!/usr/bin/env python
"""Identity between the Python and Rust search using the REAL networks.

The port was verified with a deterministic mock evaluator, deliberately: with the
real net, float reductions can depend on batch shape, so a divergence would
confound "the search is wrong" with "the network is not bit-reproducible", and the
first is what the port had to get right.

That leaves one gap the mock cannot close: whether the Rust path calls torch the
same way the Python path does. Same models, same order, same `.float()`, same
tokens. This closes it, on the actual checkpoints the bot plays with.

Compared at fixed simulations, `dedup` and `mate_distance` OFF, which is the
configuration that is supposed to be byte-identical:

  * the root visit vector, move for move, in order
  * the move that would be played

Run with the GPU free. It loads both nets.

    tests/identity_engine.py --positions 12 --sims 400
"""
from __future__ import annotations
import argparse, dataclasses, random, sys, time
from pathlib import Path
import chess, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chessgpu.engines.neural_engine import load_policy      # noqa: E402
from chessgpu.hlgauss import HLGauss                        # noqa: E402
from chessgpu.mcts import MCTS                              # noqa: E402
from chessgpu.model import ChessTransformer, ModelConfig    # noqa: E402
from chessgpu.rust_mcts import RustMCTS                     # noqa: E402
from chessgpu.value_policy import ValuePolicy               # noqa: E402


def load_nets(device="cuda:0"):
    policy, pinfo = load_policy(str(ROOT / "runs/policy.pt"))
    ck = torch.load(str(ROOT / "runs/value.pt"), map_location=device, weights_only=False)
    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    vmodel = ChessTransformer(
        ModelConfig(**{k: v for k, v in ck["cfg"].items() if k in fields})
    )
    state = ck.get("ema") or ck["model"]
    vmodel.load_state_dict({k: v.float() for k, v in state.items()})
    value = ValuePolicy(vmodel, HLGauss(bins=ck["cfg"]["output_size"]), device=device)
    return policy, value, pinfo["step"], ck.get("step")


def corpus(n: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    out = [[]]
    while len(out) < n:
        b, moves = chess.Board(), []
        for _ in range(rng.randint(1, 40)):
            legal = list(b.legal_moves)
            if not legal:
                break
            mv = rng.choice(legal)
            b.push(mv); moves.append(mv.uci())
        if not b.is_game_over(claim_draw=False):
            out.append(moves)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", type=int, default=12)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    policy, value, pstep, vstep = load_nets()
    print(f"policy step {pstep}, value step {vstep}, {args.sims} sims, batch {args.batch}")

    games = corpus(args.positions, args.seed)
    bad = 0
    py_t = rs_t = 0.0
    for i, moves in enumerate(games):
        board = chess.Board()
        for u in moves:
            board.push(chess.Move.from_uci(u))

        py = MCTS(value, policy=policy, c_puct=2.0, c_puct_base=19652.0,
                  c_puct_init=1.25, simulations=args.sims, batch=args.batch,
                  fpu=-0.2, reuse=False)
        t0 = time.perf_counter()
        _r, pv = py.search(board.copy())
        py_t += time.perf_counter() - t0
        want = [(m.uci(), v) for m, v in pv.items()]

        rs = RustMCTS(value, policy=policy, simulations=args.sims,
                      batch=args.batch, reuse=False, dedup=False,
                      mate_distance=False)
        t0 = time.perf_counter()
        _rr, rvis = rs.search(board.copy())
        rs_t += time.perf_counter() - t0
        got = [(m.uci(), v) for m, v in rvis.items()]

        if got != want:
            bad += 1
            if bad <= 3:
                same = [m for m, _ in got] == [m for m, _ in want]
                print(f"\nFAIL [{i}] {board.fen()}")
                if not same:
                    print(f"  move sets/order differ")
                    print(f"    rust:   {' '.join(m for m,_ in got)}")
                    print(f"    python: {' '.join(m for m,_ in want)}")
                else:
                    d = [(m, g, w) for (m, g), (_, w) in zip(got, want) if g != w]
                    print(f"  {len(d)}/{len(got)} visit counts differ: {d[:6]}")
                    print(f"  totals rust={sum(v for _,v in got)} python={sum(v for _,v in want)}")
        elif i and i % 4 == 0:
            print(f"  {i} identical")

    n = len(games)
    print(f"\n{n - bad}/{n} byte-identical with the real networks")
    print(f"wall clock: python {py_t:.1f}s, rust {rs_t:.1f}s  ({py_t/max(rs_t,1e-9):.1f}x)")
    if bad:
        print("\nFAIL: the Rust path does not reproduce the Python path on real nets.")
        return 1
    print("\nOK: same moves, same visits. The switch is provably behaviour-neutral.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
