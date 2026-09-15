#!/usr/bin/env python
"""`--cpuct` does nothing in the shipped configuration. Keep that impossible to
forget, and keep `--cpuct-init` wired.

No GPU and no engine boot: `c_puct_at()` is a pure function of three attributes,
so this asserts against them directly.

The history. AlphaZero's schedule is
`c(N) = ln((1 + N + base) / base) + c_puct_init`, and the constant `c_puct` is
read ONLY when `c_puct_base is None`, i.e. only under `--fixed-cpuct`. The
default ships the schedule. So for the whole life of this project:

  * `--cpuct` was a silent no-op in every match that did not also pass
    `--fixed-cpuct`, while `config.json` faithfully recorded whatever was
    passed, which made the no-op invisible in the archive;
  * nothing anywhere could vary `c_puct_init`, so every match ever run used the
    hardcoded 1.25; and
  * a c_puct sweep on 2026-08-09 returned FIVE IDENTICAL ARMS (values 1.0
    through 4.5, all 2W 1D 1L, byte-identical move hashes) which reads exactly
    like "this parameter does not matter" rather than "this parameter is not
    connected".

That last failure mode is the reason this file exists. A tuning result of "no
effect" and a plumbing bug are indistinguishable from the outside, and the
cheap ones to catch are the plumbing bugs.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sumofish.mcts import MCTS  # noqa: E402


def c_puct_at(c_puct: float, base: float | None, init: float, visits: int) -> float:
    """`MCTS.c_puct_at` with the three attributes injected.

    Constructed via `object.__new__` because the real `__init__` wants a loaded
    value net and this is a pure-arithmetic assertion. If `c_puct_at` ever grows
    a dependency on anything else, this raises AttributeError and the test fails
    loudly rather than silently testing the wrong thing.
    """
    m = object.__new__(MCTS)
    m.c_puct = c_puct
    m.c_puct_base = base
    m.c_puct_init = init
    return MCTS.c_puct_at(m, visits)


def main() -> int:
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    # 1. Under the schedule, c_puct is dead. This is the whole point.
    for visits in (0, 1, 100, 60_000):
        lo = c_puct_at(1.0, 19652.0, 1.25, visits)
        hi = c_puct_at(4.5, 19652.0, 1.25, visits)
        check(lo == hi,
              f"c_puct changed the schedule at visits={visits}: {lo} vs {hi}. "
              "If this now binds, --cpuct is no longer a no-op and the "
              "docstrings added 2026-08-09 are wrong.")

    # 2. Under the schedule, c_puct_init is what binds.
    a = c_puct_at(2.0, 19652.0, 0.5, 100)
    b = c_puct_at(2.0, 19652.0, 3.0, 100)
    check(a != b, "c_puct_init did NOT change the schedule; the only tunable "
                  "exploration term in the shipped configuration is inert")
    check(abs((b - a) - 2.5) < 1e-9,
          f"c_puct_init should shift the schedule additively by exactly its "
          f"own delta; got {b - a} for a delta of 2.5")

    # 3. Under --fixed-cpuct, c_puct binds and c_puct_init does not.
    check(c_puct_at(3.3, None, 1.25, 100) == 3.3,
          "with c_puct_base=None the constant c_puct must be returned verbatim")
    check(c_puct_at(3.3, None, 9.9, 100) == 3.3,
          "c_puct_init must be ignored under --fixed-cpuct")

    # 4. BEHAVIOURAL, not structural. Checks 4 and 5 used to be
    #    `inspect.signature(...).parameters` and `__dataclass_fields__`
    #    membership, i.e. name checks. Both would still have passed with the
    #    forwarding line in `match.py` deleted, or with `rust_mcts.py` no longer
    #    handing `c_puct_init` to `core.Mcts` -- and that second one is the exact
    #    layer the original bug lived at, one level BELOW what a signature check
    #    can see. Worse, the CLI default equals the shipped constructor default,
    #    so a broken forward is invisible until someone passes --a-cpuct-init.
    #
    #    So: drive the Rust core with two different values and require the
    #    search to actually come out different. Priors are deliberately
    #    NON-UNIFORM -- with a flat prior and a constant value the PUCT term is
    #    symmetric across children and the visit vector can come out identical
    #    for any exploration constant, which would make this test pass while
    #    proving nothing.
    def _asym_evaluator(fens, actions):
        priors, values = [], []
        for fen, acts in zip(fens, actions):
            n = max(1, len(acts))
            w = [1.0 / (i + 1) ** 2 for i in range(n)]   # sharp, rank-ordered
            s = sum(w)
            priors.append([x / s for x in w])
            values.append(((hash(fen) % 1000) / 1000.0) * 0.6 + 0.2)
        return priors, values

    class _StubNet:
        """Enough for `make_evaluator` to BUILD. It is never called: the
        evaluator it returns is replaced immediately after construction.

        Going through the REAL `RustMCTS.__init__` is the whole point: an
        earlier draft of this check built `core.Mcts(...)` directly and
        therefore passed with `rust_mcts.py`'s forwarding line deleted, which
        is precisely the bug it is supposed to catch. Verified by inducing that
        deletion; see `docs/induced-failures.md`.
        """
        device = "cpu"
        dtype = None
        model = None
        # `make_evaluator` reads `value_policy.hl.bins` to detect a fused net
        # (2026-08-14). Without it the build raised and this check reported
        # FAIL for a month on a binding that was fine. None = not fused.
        hl = None

    def _visits(c_init: float):
        import chess
        import torch
        from sumofish.rust_mcts import RustMCTS
        net = _StubNet()
        net.dtype = torch.float32
        m = RustMCTS(net, policy=net, c_puct_init=c_init, fpu=-0.05,
                     simulations=300, batch=8, reuse=False)
        m._evaluate = _asym_evaluator   # swap AFTER __init__ built the core
        board = chess.Board(
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        # `search()` returns (root, visits) and `root` is a fresh object every
        # call, so comparing the RETURN VALUES compares object identity and is
        # unequal no matter what the search did. An earlier draft did exactly
        # that and could not have failed. Compare the visit vector only.
        _root, visits = m.search(board)
        return {mv.uci(): n for mv, n in visits.items()}

    try:
        lo, hi = _visits(0.25), _visits(4.0)
        check(lo != hi,
              "c_puct_init=0.25 and c_puct_init=4.0 produced IDENTICAL root "
              "visits, so the value is not reaching the Rust tree. This is the "
              "2026-08-09 bug: five sweep arms came back byte-identical.")
    except Exception as exc:
        failures.append(f"could not drive the Rust core to check binding: {exc}")

    # 5. `match.py` must PASS it at every construction site, not merely have a
    #    field for it. Checked on the AST, because the failure being guarded
    #    against is a deleted keyword argument at a call site.
    try:
        import ast
        import match  # scripts/match.py
        check("c_puct_init" in match.Spec.__dataclass_fields__,
              "match.Spec has no c_puct_init field, so --cpuct-init cannot "
              "reach either engine no matter what the parser accepts")
        tree = ast.parse((ROOT / "scripts" / "match.py").read_text())
        sites = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name)
                 and n.func.id in ("MCTS", "RustMCTS")]
        check(len(sites) >= 2,
              f"expected both engines to be constructed in match.py, found "
              f"{len(sites)} construction site(s)")
        for site in sites:
            kw = {k.arg for k in site.keywords}
            for param in ("c_puct_init", "fpu"):
                check(param in kw,
                      f"match.py constructs {site.func.id}() without passing "
                      f"{param}=, so a --{param.replace('_', '-')} on the CLI "
                      f"is silently replaced by the constructor default")
    except Exception as exc:
        failures.append(f"could not check match.py construction sites: {exc}")

    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("verify_cpuct_binding: OK "
          "(--cpuct inert under the schedule; --cpuct-init measurably changes "
          "the Rust search; match.py passes c_puct_init and fpu at every "
          "engine construction site)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
