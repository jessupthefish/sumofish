#!/usr/bin/env python
"""Price a sequence of checkpoints against one reference, and plot the curve.

    scripts/ladder.py --reference runs/value.pt \
        --checkpoints runs/9M-sv-continue/step*.pt

This is an ORCHESTRATOR, not a second harness. Every game is played by
`scripts/match.py`, so the fingerprinting, the resume refusal, the Stockfish
arbiter and the SPRT all come along unchanged. Reimplementing any of that here
would create a second path to a published number, and the archive already shows
what happens when a number can be produced by something other than playing the
games: 4 of the first 8 matches in this project were replays.

# What it is for

The project has no training-Elo curve. Not because nobody wanted one, but
because `train.py` used to write only `latest.pt` and `best.pt`, so a finished
run left two checkpoints and the only comparable pair in the archive was 10k
against 280k, from two different runs. With `--keep-every` writing step-tagged
rungs, the curve becomes buildable, and this walks it.

The question it answers is the one that gates everything else on the roadmap:
is more training still buying strength, or has this model stopped moving? Loss
cannot answer it -- loss falls while puzzle accuracy sits flat, and that
disagreement was resolved the wrong way once already, in a direction that
steered a 37-hour run.

# Read the output as a curve, not as a row of verdicts

Adjacent rungs differ by a small amount of training, and a 200-game match sees
about +-25 Elo at best. Individual rungs will therefore be individually
inconclusive, and that is expected. The signal is the TREND across rungs: a
curve that climbs and flattens says the model converged, one that climbs
throughout says it was stopped early, and one that is flat from the first rung
says the warm restart did nothing and the schedule is the problem.

Do not read a single non-significant rung as "no improvement". That is the
error the +-1.5% puzzle metric invited for the whole first half of this project.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# `[ 200|100pr] W60 D80 L60  score  50.0%  elo   +0.0 +- 24.3  LOS  50.0% ...`
LINE = re.compile(
    r"^\[\s*(\d+)\|.*?score\s+([\d.]+)%\s+elo\s+([+-][\d.]+)\s+\+-\s*([\d.]+|inf)"
)


def step_of(path: Path) -> int | None:
    """The training step a rung represents, from `step0320000.pt`."""
    m = re.search(r"step(\d+)", path.stem)
    return int(m.group(1)) if m else None


def slug(p: Path) -> str:
    """A name unique across runs.

    `p.stem` alone is not: two runs both contain `best.pt`, so a ladder mixing
    them would build the same match directory twice. match.py's fingerprint
    guard would catch that and REFUSE, which is correct but turns a measurable
    rung into a gap in the curve. Qualify with the run directory.
    """
    return f"{p.parent.name}-{p.stem}"


def run_one(ckpt: Path, ref: Path, args) -> dict:
    """One match. Returns the parsed result, or the refusal that stopped it."""
    name = f"ladder-{slug(ref)}-vs-{slug(ckpt)}"
    cmd = [
        str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/match.py"),
        "--a-value", str(ref), "--a-label", f"ref:{slug(ref)}",
        "--b-value", str(ckpt), "--b-label", f"rung:{slug(ckpt)}",
        "--policy", str(args.policy),
        "--core", args.core, "--sims", str(args.sims),
        "--games", str(args.games), "--name", name,
    ]
    if args.extra:
        cmd += args.extra.split()

    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out = proc.stdout + proc.stderr

    # Exit 2 is match.py refusing to resume across a spec change. That is the
    # guard working, not a crash, and it must never be silently skipped -- a
    # skipped rung leaves a gap in the curve that looks like a measurement.
    if proc.returncode == 2:
        return {"checkpoint": str(ckpt), "step": step_of(ckpt),
                "status": "REFUSED", "detail": out.strip().splitlines()[-1:],
                "seconds": round(time.time() - started, 1)}

    last = None
    for ln in out.splitlines():
        m = LINE.match(ln.strip())
        if m:
            last = m
    if last is None:
        return {"checkpoint": str(ckpt), "step": step_of(ckpt),
                "status": "NO RESULT", "returncode": proc.returncode,
                "detail": out.strip().splitlines()[-3:],
                "seconds": round(time.time() - started, 1)}

    games, score, elo, err = last.groups()
    # match.py reports from A's perspective, and A is the reference. The rung's
    # Elo is therefore the negation: positive means the rung beat the incumbent.
    return {
        "checkpoint": str(ckpt),
        "step": step_of(ckpt),
        "status": "ok",
        "games": int(games),
        "rung_elo": -float(elo),
        "err": None if err == "inf" else float(err),
        "sprt": "concluded" if "SPRT concluded" in out else "cap reached",
        "seconds": round(time.time() - started, 1),
        "match_dir": f"runs/matches/{name}",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", default=str(ROOT / "runs/value.pt"),
                    help="the incumbent every rung is measured against")
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--policy", default=str(ROOT / "runs/policy.pt"),
                    help="shared by both arms, so the value net is isolated")
    ap.add_argument("--sims", type=int, default=400,
                    help="FIXED SIMS, deliberately. A fixed-clock ladder would "
                         "be invalid under any contention, and this is meant to "
                         "be runnable while something else uses the GPU.")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--core", default="rust", choices=("python", "rust"))
    ap.add_argument("--out", default=str(ROOT / "runs/ladder"))
    ap.add_argument("--extra", default="", help="passed through to match.py")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ref = Path(args.reference).resolve()
    if not ref.exists():
        print(f"no reference checkpoint: {ref}", file=sys.stderr)
        return 1

    rungs = sorted({Path(c).resolve() for c in args.checkpoints if Path(c).exists()},
                   key=lambda p: (step_of(p) is None, step_of(p) or 0, p.name))
    if not rungs:
        print("no checkpoints matched. If the run is young, its first "
              "step-tagged rung has not been written yet.", file=sys.stderr)
        return 1

    print(f"reference : {ref}")
    print(f"rungs     : {len(rungs)}")
    for r in rungs:
        print(f"    step {step_of(r) or '?':>8}  {slug(r)}")
    print(f"each rung : {args.games} games at {args.sims} sims, core={args.core}")
    print()
    if args.dry_run:
        print("--dry-run, stopping here")
        return 0

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    curve = outdir / f"{ref.stem}-curve.jsonl"

    results = []
    with open(curve, "a") as fh:
        for i, rung in enumerate(rungs, 1):
            print(f"[{i}/{len(rungs)}] step {step_of(rung)} ...", flush=True)
            res = run_one(rung, ref, args)
            results.append(res)
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            if res["status"] == "ok":
                e, err = res["rung_elo"], res["err"]
                band = f"+-{err:.1f}" if err is not None else "+-inf"
                print(f"        {e:+7.1f} {band}  over {res['games']} games "
                      f"({res['seconds'] / 60:.0f} min, {res['sprt']})")
            else:
                print(f"        {res['status']}: {res.get('detail')}")

    ok = [r for r in results if r["status"] == "ok"]
    print(f"\ncurve written to {curve}")
    if not ok:
        print("no rung produced a result; nothing to plot")
        return 1

    print(f"\n{'step':>9}  {'Elo vs reference':>18}  {'games':>6}")
    for r in ok:
        err = f"+-{r['err']:.1f}" if r["err"] is not None else "+-inf"
        print(f"{r['step']:>9,}  {r['rung_elo']:>+11.1f} {err:>6}  {r['games']:>6}")

    best = max(ok, key=lambda r: r["rung_elo"])
    print()
    print(f"best rung: step {best['step']:,} at {best['rung_elo']:+.1f} Elo")
    print()
    print("Read the TREND, not the individual rungs. Adjacent rungs differ by a "
          "little training and a 200-game match sees about +-25 Elo, so single "
          "rungs are expected to be inconclusive. Climbing then flat means "
          "converged; climbing throughout means it was stopped early; flat from "
          "the first rung means the restart did nothing.")
    print()
    print("Promoting the best rung is NOT automatic: best-of-N on a noisy metric "
          "is biased upward by roughly two sigma, which already promoted the "
          "worse of two checkpoints once here. Confirm the winner against the "
          "incumbent in a FRESH match with a different seed before promoting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
