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

    # 4. Both engines must ACCEPT the parameter. A flag that parses and is then
    #    dropped one layer down is the same silent no-op wearing a new hat.
    for mod, cls in (("sumofish.mcts", "MCTS"), ("sumofish.rust_mcts", "RustMCTS")):
        try:
            m = __import__(mod, fromlist=[cls])
            sig = inspect.signature(getattr(m, cls).__init__)
            check("c_puct_init" in sig.parameters,
                  f"{mod}.{cls}.__init__ does not accept c_puct_init")
        except Exception as exc:  # import of the rust binding can fail w/o build
            print(f"  (skipped {mod}.{cls}: {exc})")

    # 5. match.py must carry it on the Spec, or no CLI flag can reach an engine.
    try:
        import match  # scripts/match.py
        check("c_puct_init" in match.Spec.__dataclass_fields__,
              "match.Spec has no c_puct_init field, so --cpuct-init cannot "
              "reach either engine no matter what the parser accepts")
    except Exception as exc:
        failures.append(f"could not import scripts/match.py to check Spec: {exc}")

    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("verify_cpuct_binding: OK "
          "(--cpuct inert under the schedule, --cpuct-init binds, both engines "
          "and match.Spec carry it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
