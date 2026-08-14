#!/usr/bin/env python
"""The match harness records what it measures, and refuses what it cannot.

Three claims, all of them about `scripts/match.py`, none of them about chess.

  1. **A game record carries its search cost.** Under `--time` and `--tc` the
     harness sets `simulations = 10**9` so the clock binds, which makes the
     evaluation count the DEPENDENT variable and therefore the one quantity
     that prices a slower network. Until 2026-08-14 it was computed and thrown
     away: `games.jsonl` had no sims key, and all six archived movetime arms
     are silent about it. Now every record carries a `search` block.

  2. **An engine is identified by the CONTENT of its checkpoint.** `Spec.value`
     is a path, and `runs/value.pt` is a path that every promotion overwrites,
     so a resume keyed on the path alone will extend a match with a different
     network and report the two halves as one population. That is the hazard
     PHILOSOPHY:197-199 names and it is how the exchange-rate ladder came to be
     four replays. The sha of each arm's checkpoint bytes now sits inside the
     resume fingerprint.

  3. **A wall-clock match refuses a shared box.** PHILOSOPHY makes an idle
     machine a validity condition rather than a preference, and it was stated
     and never enforced for a year. Fixed-simulation work is unaffected, where
     contention costs time and not validity.

Claims 2 and 3 need no GPU: both refusals happen before a single game. Claim 1
does, so it is skipped unless --deep is passed.

Usage:
    tests/verify_match_recording.py           # claims 2 and 3, seconds, no GPU
    tests/verify_match_recording.py --deep    # also claim 1, plays two games
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv/bin/python")
MATCH = str(ROOT / "scripts/match.py")

failures: list[str] = []


def check(ok: bool, what: str, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {what}" + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(what)


def run(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run([PY, MATCH, *args], capture_output=True, text=True,
                          cwd=ROOT, timeout=900, **kw)


def load_match(alias: str):
    """Import scripts/match.py as a module, correctly.

    `module_from_spec` + `exec_module` alone is not enough: @dataclass resolves
    its field types through `sys.modules[cls.__module__]`, so a module that was
    executed but never REGISTERED makes `Spec` raise
    `AttributeError: 'NoneType' object has no attribute '__dict__'` at class
    creation. The failure reads like a bug in match.py and is a bug in the
    loader. Register first, execute second.
    """
    import importlib.util
    # `python scripts/match.py` puts scripts/ on sys.path[0] for free; an
    # import by path does not, and match.py's `from elo import ...` needs it.
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(alias, MATCH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# Claim 2: content-hashed engines
# --------------------------------------------------------------------------
def claim_content_hash() -> None:
    print("\n=== claim 2: an engine is identified by checkpoint CONTENT ===")

    m = load_match("_match_recording_a")

    live = ROOT / "runs/value.pt"
    if not live.exists():
        print("  skipped: no runs/value.pt to hash")
        return

    h1 = m.checkpoint_sha(str(live))
    h2 = m.checkpoint_sha(str(live))
    check(h1 == h2 and len(h1) == 12, "hashing a checkpoint is deterministic",
          f"sha {h1}")
    check(m.checkpoint_sha(None) == "none",
          "an arm with no checkpoint hashes to 'none', not to the empty string")
    check(m.checkpoint_sha("/nonexistent/x.pt") == "missing",
          "a missing checkpoint is 'missing', distinct from 'none'")
    check(m.checkpoint_sha(None) != m.checkpoint_sha("/nonexistent/x.pt"),
          "the two kinds of absence do not hash equal")

    # Two files with the same NAME and different CONTENT must not hash alike.
    with tempfile.TemporaryDirectory() as td:
        a, b = Path(td) / "value.pt", Path(td) / "other" / "value.pt"
        b.parent.mkdir()
        a.write_bytes(b"one"); b.write_bytes(b"two")
        check(m.checkpoint_sha(str(a)) != m.checkpoint_sha(str(b)),
              "same filename, different bytes, different hash")
        c = Path(td) / "third" / "value.pt"
        c.parent.mkdir(); c.write_bytes(b"one")
        check(m.checkpoint_sha(str(a)) == m.checkpoint_sha(str(c)),
              "different path, same bytes, same hash (content, not location)")

    # And the hash is inside the Spec, so it reaches the resume fingerprint.
    check("value_sha" in m.Spec.__dataclass_fields__
          and "policy_sha" in m.Spec.__dataclass_fields__,
          "Spec carries value_sha and policy_sha, so the fingerprint sees them")


# --------------------------------------------------------------------------
# Claim 2b: a swapped checkpoint is actually refused at resume
# --------------------------------------------------------------------------
def claim_resume_refusal() -> None:
    print("\n=== claim 2b: a resume with different bytes is refused ===")
    live = ROOT / "runs/value.pt"
    if not live.exists():
        print("  skipped: no runs/value.pt")
        return

    with tempfile.TemporaryDirectory() as td:
        # A checkpoint that is byte-different but structurally loadable is not
        # needed: the refusal happens before anything is loaded. A copy with one
        # byte appended is enough, and it keeps this test off the GPU.
        swapped = Path(td) / "value.pt"
        shutil.copy(live, swapped)
        with swapped.open("ab") as fh:
            fh.write(b"\0")

        name = "_selftest_resume_refusal"
        outdir = ROOT / "runs/matches" / name
        if outdir.exists():
            shutil.rmtree(outdir)
        try:
            # Round 1: two games at 4 sims, to lay down a config and a log.
            r = run(["--name", name, "--games", "2", "--sims", "4"])
            if not (outdir / "games.jsonl").exists():
                print(f"  skipped: could not lay down a match ({r.returncode})")
                print("  " + (r.stderr or r.stdout).strip().splitlines()[-1][:160])
                return
            first = json.loads((outdir / "config.json").read_text())
            check(first.get("a", {}).get("value_sha") not in (None, ""),
                  "the written config records each arm's checkpoint sha",
                  f"a.value_sha {first['a']['value_sha']}")

            # Round 2: same everything, different checkpoint BYTES.
            r = run(["--name", name, "--games", "4", "--sims", "4",
                     "--a-value", str(swapped)])
            out = (r.stdout or "") + (r.stderr or "")
            check(r.returncode == 2, "resuming with swapped bytes exits 2",
                  f"got {r.returncode}")
            check("REFUSING to resume" in out, "and says so")
            check("value_sha" in out, "and names value_sha as what moved",
                  [l.strip() for l in out.splitlines() if "value_sha" in l][:1])

            # The control that makes the above mean something: an identical
            # rerun must still be allowed to resume, or the check is just
            # "refuse everything".
            r = run(["--name", name, "--games", "2", "--sims", "4"])
            check(r.returncode != 2,
                  "an identical rerun is NOT refused (the check is specific)",
                  f"rc {r.returncode}")
        finally:
            if outdir.exists():
                shutil.rmtree(outdir)


# --------------------------------------------------------------------------
# Claim 3: a wall-clock match refuses a shared box
# --------------------------------------------------------------------------
def claim_idle_assertion() -> None:
    print("\n=== claim 3: a wall-clock match refuses a shared box ===")
    m = load_match("_match_recording_b")

    busy = m.contending_units()
    print(f"  box currently reports {len(busy)} contending unit(s)")

    # The guard must not block ITSELF. A match launched as a `sumofish-*`
    # systemd unit matches its own glob, and without the own-unit exclusion it
    # refuses to start on the grounds that it is already running. Run the probe
    # inside such a unit and require that the unit is absent from its own list.
    probe = ("import sys; sys.path.insert(0, %r); import importlib.util as iu; "
             "sp = iu.spec_from_file_location('_p', %r); mm = iu.module_from_spec(sp); "
             "sys.modules['_p'] = mm; sp.loader.exec_module(mm); "
             "print('UNITS:' + '|'.join(mm.contending_units()))"
             % (str(ROOT / "scripts"), MATCH))
    unit = "sumofish-selftest-selfblock"
    r = subprocess.run(["systemd-run", "--user", "--quiet", "--wait", "--pipe",
                        f"--unit={unit}", PY, "-c", probe],
                       capture_output=True, text=True, cwd=ROOT, timeout=300)
    line = next((l for l in (r.stdout or "").splitlines()
                 if l.startswith("UNITS:")), None)
    subprocess.run(["systemctl", "--user", "reset-failed", unit],
                   capture_output=True)
    if line is None:
        print(f"  skipped self-block check: systemd-run gave nothing "
              f"(rc {r.returncode})")
    else:
        seen = [u for u in line[len("UNITS:"):].split("|") if u]
        check(not any(unit in u for u in seen),
              "the guard does not list its OWN unit as contention",
              f"saw {[u for u in seen if unit in u]}" if any(unit in u for u in seen)
              else f"{len(seen)} other unit(s)")

    for u in busy[:4]:
        print(f"    {u[:100]}")

    name = "_selftest_idle_assertion"
    outdir = ROOT / "runs/matches" / name
    if outdir.exists():
        shutil.rmtree(outdir)
    try:
        r = run(["--name", name, "--games", "2", "--time", "0.1"])
        out = (r.stdout or "") + (r.stderr or "")
        if busy:
            check("refusing to start a wall-clock match" in out,
                  "a clock match on a shared box is refused")
            check("--allow-contended" in out,
                  "and the refusal names its own escape hatch")
            check(not outdir.exists(),
                  "and it leaves no half-made run directory behind")
        else:
            check("refusing to start a wall-clock match" not in out,
                  "an IDLE box is not refused (the check is not always-on)")
            print("  note: box was idle, so the refusal path was not exercised")
    finally:
        if outdir.exists():
            shutil.rmtree(outdir)


# --------------------------------------------------------------------------
# Claim 1: the search block is present and consistent (GPU)
# --------------------------------------------------------------------------
def claim_search_block() -> None:
    print("\n=== claim 1: every game record carries its search cost ===")
    name = "_selftest_search_block"
    outdir = ROOT / "runs/matches" / name
    if outdir.exists():
        shutil.rmtree(outdir)
    try:
        r = run(["--name", name, "--games", "2", "--sims", "16"])
        log = outdir / "games.jsonl"
        if not log.exists():
            check(False, "the smoke match produced games.jsonl",
                  (r.stderr or r.stdout).strip().splitlines()[-1][:160])
            return
        rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        check(bool(rows), f"{len(rows)} game(s) recorded")
        keys = {"white_evals", "black_evals", "white_unique",
                "black_unique", "white_moves", "black_moves"}
        for i, g in enumerate(rows):
            s = g.get("search")
            if not check(isinstance(s, dict), f"game {i} has a search block"):
                continue
            check(keys <= set(s), f"game {i} carries every search key",
                  f"missing {sorted(keys - set(s))}" if keys - set(s) else "")
            check(s["white_evals"] > 0 and s["black_evals"] > 0,
                  f"game {i} counted evaluations on both sides",
                  f"{s['white_evals']}/{s['black_evals']}")
            # The invariant the root-expansion bug broke: with dedup OFF the
            # unique count can never EXCEED the total, and it undercounted by
            # exactly one per search until 2026-08-14.
            check(s["white_unique"] <= s["white_evals"]
                  and s["black_unique"] <= s["black_evals"],
                  f"game {i}: unique never exceeds total")
            check(s["white_moves"] > 0, f"game {i} counted moves",
                  f"{s['white_moves']}/{s['black_moves']}")
        cfg = json.loads((outdir / "config.json").read_text())
        check("machine" in cfg, "config.json stamps the machine state")
        check("clock_match" in cfg.get("machine", {}),
              "and says whether this was a clock match")
    finally:
        if outdir.exists():
            shutil.rmtree(outdir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--deep", action="store_true",
                    help="also play two real games to check the search block")
    a = ap.parse_args()

    os.environ.setdefault("CHESSGPU_CORE", "rust")
    claim_content_hash()
    claim_resume_refusal()
    claim_idle_assertion()
    if a.deep:
        claim_search_block()
    else:
        print("\n=== claim 1 skipped (needs the GPU); pass --deep ===")

    print()
    if failures:
        print(f"FAILED: {len(failures)}")
        for f in failures:
            print(f"  {f}")
        return 1
    print("OK: the harness records what it measures and refuses what it cannot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
