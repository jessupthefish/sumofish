#!/usr/bin/env python
"""Swap a live net, by hand, the same way the lab does it automatically.

Handles BOTH deployed nets as of 2026-08-14. It previously handled `value.pt`
alone, which meant the policy net -- half the deployed model -- was swapped
with `cp` and had no tooled rollback. `runs/policy.pt.json` still carries the
scar: `"promoted_by": "manual swap (scripts/promote.py handles runs/value.pt
only)"`.

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
    scripts/promote.py --net policy runs/9M-bc-2026-08-01/best.pt
    scripts/promote.py --status                 # both nets
    scripts/promote.py --net policy --rollback

A FUSED checkpoint (one trunk, both heads, `output_size == bins + NUM_ACTIONS`)
is promoted into the `value` slot and is detected from the file, never from a
flag. `runs/policy.pt` is left in place untouched so that `--rollback` on the
value slot restores the two-net engine exactly; the live engine ignores it
while the value slot holds a fused net and says so at boot. `--net policy` on
a fused checkpoint is refused: there is no separate prior to swap.
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

# Both nets, not just the value net. Until 2026-08-14 this script handled
# `value.pt` alone, so the policy net -- half the deployed model -- was
# promoted by a hand-typed `cp` and its only rollback instruction lived in a
# sidecar that `.gitignore` excluded. `runs/policy.pt.json` still records
# `"promoted_by": "manual swap (scripts/promote.py handles runs/value.pt
# only)"`, which is the artefact of exactly that gap.
NETS = ("value", "policy")


def is_fused(ckpt: Path) -> bool:
    """From the checkpoint's own shape, via the loader the engine uses."""
    sys.path.insert(0, str(ROOT))
    import torch  # deferred: --status and --rollback should not need the GPU stack

    from sumofish.engines.loader import fused_bins

    return fused_bins(
        torch.load(str(ckpt), map_location="cpu", weights_only=False)) is not None


def paths(net: str) -> dict[str, Path]:
    """Live, previous, staged and sidecar paths for one net."""
    base = ROOT / "runs" / f"{net}.pt"
    return {
        "live": base,
        "previous": base.with_suffix(".pt.previous"),
        "staged": base.with_suffix(".pt.staged"),
        "sidecar": base.with_suffix(".pt.json"),
    }


def smoke(ckpt: Path, net: str) -> tuple[bool, str]:
    """Does this checkpoint actually work? `scripts/smoke.py` is the gate.

    A match says "stronger"; this says "loads, boots, moves inside the budget,
    never plays an illegal move". Different questions, and the second is the one
    that has actually bitten: a checkpoint that fails to load takes the engine
    down at boot, and lichess-bot's start limit turns that into a unit in
    `failed`.
    """
    r = subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/smoke.py"),
         str(ckpt), "--net", net],
        capture_output=True, text=True, check=False,
    )
    out = r.stdout + r.stderr
    if r.returncode != 0:
        fail = next((ln for ln in out.splitlines() if ln.startswith("FAIL")),
                    "smoke test failed")
        return False, fail
    return True, "smoke passed"


def status(nets: tuple[str, ...] = NETS) -> int:
    """Both nets by default: the deployed engine is the PAIR, and reporting on
    one of them was how the policy net's provenance went unexamined."""
    rc = 0
    for net in nets:
        p = paths(net)
        print(f"=== {net} ===")
        if not p["live"].exists():
            print(f"  no {p['live'].relative_to(ROOT)} -- nothing is live")
            rc = 1
            continue
        print(f"  live: {p['live'].relative_to(ROOT)}  "
              f"({p['live'].stat().st_size / 1e6:.0f} MB)")
        if p["sidecar"].exists():
            body = json.dumps(json.loads(p["sidecar"].read_text()), indent=2)
            print("  " + body.replace("\n", "\n  "))
        else:
            print("  no provenance sidecar: this net was put here by something "
                  "that did not record why, so it cannot be attributed.")
        print(f"  rollback available: {p['previous'].exists()}")
    vs = paths("value")["sidecar"]
    if vs.exists() and json.loads(vs.read_text()).get("fused"):
        print("=== fused ===\n  the value slot holds a fused net; the policy "
              "slot above is not what plays")
    return rc


def rollback(net: str) -> int:
    p = paths(net)
    if not p["previous"].exists():
        print(f"no {p['previous'].relative_to(ROOT)} to roll back to",
              file=sys.stderr)
        return 1
    shutil.copyfile(p["previous"], p["staged"])
    os.replace(p["staged"], p["live"])
    print(f"rolled back: {p['previous'].name} -> {p['live'].name}")
    print("the next game picks it up; no restart needed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", nargs="?")
    ap.add_argument("--net", choices=NETS, default="value",
                    help="which net this checkpoint is. Default value, because "
                         "that is what the lab promotes; pass --net policy for "
                         "the prior. Getting this wrong is caught by the smoke "
                         "test, which loads the candidate into the named slot.")
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
        return rollback(args.net)
    if not args.checkpoint:
        ap.error("give a checkpoint, or --status / --rollback")

    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.exists():
        print(f"no such checkpoint: {ckpt}", file=sys.stderr)
        return 1

    fused = is_fused(ckpt)
    if fused and args.net == "policy":
        print("a fused checkpoint carries both heads; promote it into the "
              "value slot (the default), not --net policy", file=sys.stderr)
        return 1
    p = paths(args.net)

    if not args.skip_smoke:
        print(f"smoke testing {ckpt.name} as the {args.net} net ...")
        ok, msg = smoke(ckpt, args.net)
        print(f"  {msg}")
        if not ok:
            print("refusing to promote a checkpoint that does not work",
                  file=sys.stderr)
            return 1
    else:
        print("WARNING: --skip-smoke, promoting an unverified checkpoint")

    # Keep the outgoing net. One deep is thin, but a rollback that exists beats
    # one that does not.
    if p["live"].exists():
        shutil.copyfile(p["live"], p["previous"])

    # Stage beside the target and rename. Atomic, because the engine loads this
    # file per game and a game starting mid-copy would read a torn one.
    shutil.copyfile(ckpt, p["staged"])
    os.replace(p["staged"], p["live"])

    p["sidecar"].write_text(json.dumps({
        "net": args.net,
        "fused": fused,
        "policy_slot": ("unused while a fused net is live; runs/policy.pt left "
                        "in place so --rollback restores the two-net engine"
                        if fused else "runs/policy.pt"),
        "source": str(ckpt),
        "promoted_at": time.time(),
        "promoted_by": "scripts/promote.py (manual)",
        "note": args.note,
        # Deliberately null: a manual promotion has not been measured, and writing
        # a number here that no match produced is how a provenance sidecar becomes
        # fiction. The lab writes a real one because a match gave it one.
        "elo": None,
        "smoke": "skipped" if args.skip_smoke else "passed",
        "rollback": f"scripts/promote.py --net {args.net} --rollback",
    }, indent=2))

    print(f"promoted -> {p['live']}{' (fused: both heads)' if fused else ''}")
    if fused:
        print("runs/policy.pt is now IGNORED by the engine and left in place "
              "for rollback")
    print("the next game picks it up; NO restart, because lichess-bot spawns a "
          "fresh engine per game and restarting costs the game in progress")
    print(f"rollback: scripts/promote.py --net {args.net} --rollback")
    print()
    print("This promotion is UNMEASURED. If it is meant to be a version, cut one "
          "deliberately with scripts/release.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
