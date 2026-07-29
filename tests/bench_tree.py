"""Tree-only stopwatch: a trivial evaluator on both sides, so the ratio is the tree.

Not identity-tested and does not need to be -- identity is proved by
identity_search.py with the real mock. This exists purely to answer "how fast is
the tree machinery", which the identity mock is too expensive to reveal.

Python gets `policy=None`, which makes `_priors_batch` produce uniform priors
from its OWN move generation -- correctly charging Python for movegen, which is
part of what the port replaces. Rust does its movegen in Rust and returns the
same uniform priors. Values are constant on both sides.
"""
import sys, time, chess
sys.path.insert(0, '.'); sys.path.insert(0, '/home/nomad/chess-gpu')
from chessgpu.mcts import MCTS
import sumofish_core as core

class TrivialValue:
    def value_of(self, board): return 0.5
    def evaluate(self, boards): return [0.5] * len(boards), None

def rust_trivial(fens, actions):
    return [[1.0 / len(a)] * len(a) for a in actions], [0.5] * len(fens)

CASES = [("startpos", []),
         ("open middlegame", ["e2e4","e7e5","g1f3","b8c6","f1b5","a7a6"]),
         ("sharp middlegame", ["d2d4","g8f6","c2c4","e7e6","g1f3","d7d5","b1c3","c7c5","c4d5","f6d5"])]
SIMS, BATCH = 4000, 64
print(f"{SIMS} simulations, batch {BATCH}, trivial evaluator (tree only)\n")
ratios = []
for name, moves in CASES:
    b = chess.Board()
    for u in moves: b.push(chess.Move.from_uci(u))
    m = MCTS(TrivialValue(), policy=None, c_puct=2.0, c_puct_base=19652.0,
             c_puct_init=1.25, simulations=SIMS, batch=BATCH, fpu=-0.2, reuse=False)
    t = time.perf_counter(); m.search(b); py = time.perf_counter() - t

    p = core.Position()
    for u in moves: p.push_uci(u)
    rm = core.Mcts(c_puct=2.0, c_puct_base=19652.0, c_puct_init=1.25, fpu=-0.2, batch=BATCH)
    t = time.perf_counter(); rm.search(p, SIMS, rust_trivial); rs = time.perf_counter() - t

    ratios.append(py / rs)
    print(f"  {name:18s} python {py:7.3f}s ({SIMS/py:8.0f} sim/s) | "
          f"rust {rs:7.3f}s ({SIMS/rs:9.0f} sim/s) | {py/rs:6.0f}x")
tree = sum(ratios) / len(ratios)
print(f"\n  tree-only speedup: {tree:.0f}x")
print()
print("  Engine extrapolation. Python profile: network 9%, everything else 91%.")
print("  Under this port the network stays in Python, and so does the prior")
print("  softmax (~2%, forced by numpy's summation order). The rest moves.")
overall = 1.0 / (0.09 + 0.02 + 0.89 / tree)
print(f"    1/(0.09 + 0.02 + 0.89/{tree:.0f}) = {overall:.1f}x overall")
print()
print("  NOT Elo. The exchange rate was retracted 2026-07-29.")
