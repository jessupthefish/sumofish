#!/usr/bin/env python
"""Audit every match in the archive for the replay that produced the ladder.

On 2026-07-29 all four rungs of the exchange-rate ladder -- the number
`CLAUDE.md` says "prices the rest of the project", and which `README.md`
publishes -- were found to be REPLAYS. `match.py` resume keyed on `rec["game"]`
alone, so a job with different code, a different checkpoint and a different
budget landed on an existing directory, skipped every game as already played, and
reported the old numbers as its own.

**The replay is invisible in `games.jsonl`.** The per-game timings are organic,
because the games were really played -- just not by the job credited with them.
It shows up in exactly one place: an elapsed time that could not have produced
that many games.

    sum(game.seconds) <= job.seconds

That inequality is physically impossible to violate legitimately. It has no
tuning constant, no duration model, and no false positives, and it catches all
four rungs. A ratio test with a threshold was the first design and it was worse:
it needs a model of how long a game "should" take, it fires on legitimate resumes,
and anything that fires on legitimate work gets bypassed.

Also reports provenance, since a match with no `code` fingerprint cannot be
attributed even in principle: three of the four rungs carry one on 0 of 300 games.

**Fails closed, since 2026-08-14, and the reason is the whole point.** The
verdict used to start at "trusted" and downgrade only on positive evidence, so a
match whose central check could not RUN was reported identically to one that ran
and passed. `credited` comes only from `runs/lab/state.json`, and the lab has
driven nothing since 2026-08-02, so the inequality was ABSTAINING on 100 of 104
directories while the tool printed "99 trusted". A verdict of `unverifiable` now
says so, and `--check` refuses it.

The honest coverage today is 4 of 104, and those four are the `sims-*` rungs the
tool was written to catch. Nothing else in the archive has a credited wall clock
to compare against. That is a statement about coverage, not about the other
hundred matches being dirty, and the fix is for `match.py` to record its own
start and end time rather than for this script to guess.

    scripts/verify_replays.py                 # audit and print
    scripts/verify_replays.py --json          # machine-readable
    scripts/verify_replays.py --check         # exit 1 unless every match is verified
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_job_seconds(root: Path) -> dict[str, float]:
    """What the lab credits each job, from `runs/lab/state.json`."""
    out: dict[str, float] = {}
    state = root / "runs/lab/state.json"
    if not state.exists():
        return out
    try:
        done = json.loads(state.read_text()).get("done", {})
    except Exception:
        return out
    for job, rec in done.items():
        secs = rec.get("seconds")
        if isinstance(secs, (int, float)):
            out[job] = float(secs)
    return out


def audit_match(d: Path, job_seconds: dict[str, float]) -> dict:
    log = d / "games.jsonl"
    games, total_seconds, codes, adjudicated, arbitered = 0, 0.0, set(), 0, 0
    for line in log.read_text().splitlines() if log.exists() else []:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        games += 1
        total_seconds += float(rec.get("seconds") or 0.0)
        codes.add(rec.get("code"))
        reason = rec.get("reason") or ""
        if reason.startswith("adjudicated"):
            adjudicated += 1
            if rec.get("adjudicated_by"):
                arbitered += 1

    cfg = {}
    cfg_path = d / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            cfg = {}

    # The lab's job id is not the match name; match on the name appearing in it,
    # which is how the two are related in practice (`sims-800-400` <-> the match
    # `sims-800-vs-400`).
    key = d.name.replace("-vs-", "-")
    credited = job_seconds.get(key)

    # Verdict by PRECEDENCE, worst first, rather than by a chain of
    # `if verdict == "trusted"` guards. The old form started at "trusted" and
    # downgraded only on positive evidence, which meant a match whose central
    # check could not RUN was reported identically to one that ran and passed.
    # On 2026-08-14 that was 100 of 104 directories graded "trusted" on a check
    # that never executed. A pass must assert that something was measured; it
    # must never be the mere absence of a failure.
    why: list[str] = []
    replayed = credited is not None and total_seconds > credited
    if replayed:
        why.append(
            f"credited {credited:.0f}s for {total_seconds:.0f}s of logged play "
            f"({total_seconds / max(credited, 1e-9):.0f}x impossible)"
        )

    unprovenanced = False
    if codes == {None}:
        unprovenanced = True
        why.append(f"no code fingerprint on any of {games} games")
    elif len(codes) > 1:
        unprovenanced = True
        why.append(f"{len(codes)} different code fingerprints in one match")
    if not cfg.get("fingerprint"):
        unprovenanced = True
        why.append("config.json has no spec fingerprint")

    # The inequality PHILOSOPHY calls "physically impossible to violate
    # legitimately" needs a credited wall clock to compare against, and that
    # comes only from `runs/lab/state.json`. Matches driven by a shell script
    # rather than the lab have none, so the check ABSTAINS. Abstention is not
    # a pass.
    unverifiable = credited is None
    if unverifiable and games:
        why.append(
            "no credited job wall clock: the sum(game.seconds) <= job.seconds "
            "check could not run, so this match is UNCHECKED, not clean"
        )

    if games == 0:
        verdict = "empty"
        why = ["no games"]
    elif replayed:
        verdict = "REPLAYED"
    elif unprovenanced:
        verdict = "unprovenanced"
    elif unverifiable:
        verdict = "unverifiable"
    else:
        verdict = "trusted"

    return {
        "name": d.name,
        "games": games,
        "game_seconds": round(total_seconds, 1),
        "job_seconds": credited,
        "codes": sorted(str(c) for c in codes),
        "adjudicated": adjudicated,
        "adjudicated_by_arbiter": arbitered,
        "verdict": verdict,
        "why": why,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any match is not trusted")
    args = ap.parse_args()

    mdir = args.root / "runs/matches"
    if not mdir.exists():
        print(f"no matches under {mdir}")
        return 0

    job_seconds = load_job_seconds(args.root)
    rows = [audit_match(d, job_seconds) for d in sorted(mdir.iterdir()) if d.is_dir()]

    out_path = args.root / "runs/TRUST.jsonl"
    out_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(f"{'match':26s} {'games':>6} {'game_s':>9} {'job_s':>8} {'adj':>5} "
              f"{'arb':>4}  verdict")
        for r in rows:
            js = "-" if r["job_seconds"] is None else f"{r['job_seconds']:.0f}"
            print(f"{r['name']:26s} {r['games']:>6} {r['game_seconds']:>9.0f} "
                  f"{js:>8} {r['adjudicated']:>5} {r['adjudicated_by_arbiter']:>4}  "
                  f"{r['verdict']}")
        for r in rows:
            for w in r["why"]:
                print(f"  {r['name']}: {w}")
        print(f"\nwrote {out_path}")

    from collections import Counter
    tally = Counter(r["verdict"] for r in rows)
    bad = [r for r in rows if r["verdict"] not in ("trusted", "empty")]
    print(f"\n{len(rows)} matches: " + ", ".join(
        f"{tally[v]} {v}" for v in
        ("trusted", "unverifiable", "unprovenanced", "REPLAYED", "empty")
        if tally[v]))
    if tally["unverifiable"]:
        print(f"\n{tally['unverifiable']} matches could not be checked at all. "
              "The wall-clock inequality needs a credited job time from "
              "runs/lab/state.json, and only lab-driven jobs have one. This is "
              "a statement about coverage, not about those matches being dirty.")
    if args.check and bad:
        print("FAIL: numbers from these matches may not be cited.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
