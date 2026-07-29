#!/usr/bin/env python
"""Swap the live value net, by hand, the same way the lab does it automatically.

Rewritten 2026-07-29. The previous version was broken three ways, and all three
are worth recording, because this is the path `CLAUDE.md` pointed a human at:

1. **It wrote `runs/current.pt`, which nothing reads.** The searching engine loads
   `runs/value.pt`. A "successful" promotion therefore changed nothing at all, and
   the operator had every reason to believe otherwise.

2. **It gated on puzzle accuracy.** Sigma is +-1.5% at n=1000, and both
   PHILOSOPHY and the Lab Notes say never to select on it -- it has already
   promoted the marginally worse of two checkpoints once. It also ran that
   evaluation through `load_policy`, the POLICY loader, against a state-value
   checkpoint.

3. **It restarted the bot.** The swap needs no restart: lichess-bot spawns a fresh
   engine per game, so the next game picks the new file up. Restarting costs
   whatever game is in progress, and has done.

What replaces it does what `lab.py::decide_promote` does, minus the match that
decides: keep the outgoing net, stage beside the target, rename atomically, write
the provenance sidecar, and gate on the smoke test rather than on a noisy metric.

    scripts/promote.py runs/9M-sv-warm-full/best.pt
    scripts/promote.py --status
    scripts/promote.py --rollback
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIVE = ROOT / "runs" / "value.pt"
PREVIOUS = ROOT / "runs" / "value.pt.previous"
STAGED = ROOT / "runs" / "value.pt.staged"
SIDECAR = ROOT / "runs" / "value.pt.json"


def smoke(ckpt: Path) -> tuple[bool, str]:
    """Does this checkpoint actually work? `scripts/smoke.py` is the gate.

    A match says "stronger"; this says "loads, boots, moves inside the budget,
    never plays an illegal move". Different questions, and the second is the one
    that has actually bitten: a checkpoint that fails to load takes the engine
    down at boot, and lichess-bot's start limit turns that into a unit in
    `failed`.
    """
    r = subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/smoke.py"),
         "--value", str(ckpt)],
        capture_output=True, text=True, check=False,
    )
    out = r.stdout + r.stderr
    if r.returncode != 0:
        fail = next((ln for ln in out.splitlines() if ln.startswith("FAIL")),
                    "smoke test failed")
        return False, fail
    return True, "smoke passed"


def status() -> int:
    if not LIVE.exists():
        print("no runs/value.pt -- nothing is live")
        return 1
    print(f"live: {LIVE}  ({LIVE.stat().st_size / 1e6:.0f} MB)")
    if SIDECAR.exists():
        print(json.dumps(json.loads(SIDECAR.read_text()), indent=2))
    else:
        print("no provenance sidecar: this net was put here by something that did "
              "not record why, so it cannot be attributed.")
    print(f"rollback available: {PREVIOUS.exists()}")
    return 0


def rollback() -> int:
    if not PREVIOUS.exists():
        print("no runs/value.pt.previous to roll back to", file=sys.stderr)
        return 1
    shutil.copyfile(PREVIOUS, STAGED)
    os.replace(STAGED, LIVE)
    print(f"rolled back: {PREVIOUS} -> {LIVE}")
    print("the next game picks it up; no restart needed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", nargs="?")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--rollback", action="store_true")
    ap.add_argument(
        "--skip-smoke", action="store_true",
        help="promote without checking the checkpoint boots. There is no good "
             "reason; it exists so that skipping it is a visible choice.",
    )
    ap.add_argument("--note", default="", help="why this is being promoted")
    args = ap.parse_args()

    if args.status:
        return status()
    if args.rollback:
        return rollback()
    if not args.checkpoint:
        ap.error("give a checkpoint, or --status / --rollback")

    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.exists():
        print(f"no such checkpoint: {ckpt}", file=sys.stderr)
        return 1

    if not args.skip_smoke:
        print(f"smoke testing {ckpt.name} ...")
        ok, msg = smoke(ckpt)
        print(f"  {msg}")
        if not ok:
            print("refusing to promote a checkpoint that does not work",
                  file=sys.stderr)
            return 1
    else:
        print("WARNING: --skip-smoke, promoting an unverified checkpoint")

    # Keep the outgoing net. One deep is thin, but a rollback that exists beats
    # one that does not.
    if LIVE.exists():
        shutil.copyfile(LIVE, PREVIOUS)

    # Stage beside the target and rename. Atomic, because the engine loads this
    # file per game and a game starting mid-copy would read a torn one.
    shutil.copyfile(ckpt, STAGED)
    os.replace(STAGED, LIVE)

    SIDECAR.write_text(json.dumps({
        "source": str(ckpt),
        "promoted_at": time.time(),
        "promoted_by": "scripts/promote.py (manual)",
        "note": args.note,
        # Deliberately null: a manual promotion has not been measured, and writing
        # a number here that no match produced is how a provenance sidecar becomes
        # fiction. The lab writes a real one because a match gave it one.
        "elo": None,
        "smoke": "skipped" if args.skip_smoke else "passed",
        "rollback": "scripts/promote.py --rollback",
    }, indent=2))

    print(f"promoted -> {LIVE}")
    print("the next game picks it up; NO restart, because lichess-bot spawns a "
          "fresh engine per game and restarting costs the game in progress")
    print("rollback: scripts/promote.py --rollback")
    print()
    print("This promotion is UNMEASURED. If it is meant to be a version, cut one "
          "deliberately with scripts/release.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
