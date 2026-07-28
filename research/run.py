#!/usr/bin/env python
"""Experiment harness. Runs research/train.py, scores it, keeps or reverts.

    python research/run.py            # run the current train.py as one experiment
    python research/run.py --status   # leaderboard
    python research/run.py --restore  # copy best.py back over train.py

Modelled on karpathy/autoresearch. The load-bearing idea is the fixed wall-clock
budget: every experiment gets the same number of seconds, so a change that wins
by training faster counts as a real win, exactly like a change that wins by
learning more per step. Comparing at fixed step count would hide that.

What this harness guarantees, so the agent editing train.py cannot fool itself:
  * train.py is snapshotted to attempts/ before every run, so nothing is lost.
  * The metric is parsed from the child process, not computed here.
  * A run that crashes, times out, or prints no RESULT is recorded as a failure
    and reverts, rather than silently keeping a broken file.
  * best.py is only overwritten on a genuine improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRAIN = HERE / "train.py"
BEST = HERE / "best.py"
RESULTS = HERE / "results.jsonl"
ATTEMPTS = HERE / "attempts"
PYTHON = ROOT / ".venv/bin/python"

# Wall-clock ceiling for the child: the training budget plus room for import,
# torch.compile, and the held-out eval. Generous, because killing a run that was
# about to report is worse than waiting.
OVERHEAD_S = 420


def budget_from_train_py() -> int:
    m = re.search(r"^TIME_BUDGET_S\s*=\s*(\d+)", TRAIN.read_text(), re.M)
    return int(m.group(1)) if m else 300


def load_results() -> list[dict]:
    if not RESULTS.exists():
        return []
    return [json.loads(line) for line in RESULTS.read_text().splitlines() if line.strip()]


def best_so_far(results: list[dict]) -> dict | None:
    ok = [r for r in results if r.get("ok")]
    return min(ok, key=lambda r: r["bpm"]) if ok else None


def run_once(note: str = "") -> dict:
    ATTEMPTS.mkdir(exist_ok=True)
    source = TRAIN.read_text()
    digest = hashlib.sha256(source.encode()).hexdigest()[:12]
    results = load_results()
    exp_id = len(results) + 1

    snapshot = ATTEMPTS / f"{exp_id:04d}-{digest}.py"
    snapshot.write_text(source)

    budget = budget_from_train_py()
    print(f"=== experiment {exp_id} ({digest}) budget {budget}s ===")
    if note:
        print(f"    note: {note}")

    started = time.perf_counter()
    record: dict = {
        "id": exp_id,
        "sha": digest,
        "note": note,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "snapshot": snapshot.name,
    }

    try:
        proc = subprocess.run(
            [str(PYTHON), str(TRAIN)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=budget + OVERHEAD_S,
        )
    except subprocess.TimeoutExpired:
        record |= {"ok": False, "error": "timeout", "wall": round(time.perf_counter() - started, 1)}
        print(f"    FAILED: exceeded {budget + OVERHEAD_S}s")
        _append(record)
        return record

    wall = round(time.perf_counter() - started, 1)
    stdout = proc.stdout

    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-6:]
        record |= {"ok": False, "error": f"exit {proc.returncode}", "stderr": "\n".join(tail), "wall": wall}
        print(f"    FAILED: exit {proc.returncode}")
        for line in tail:
            print(f"      {line}")
        _append(record)
        return record

    match = re.search(r"^RESULT (\{.*\})\s*$", stdout, re.M)
    if not match:
        record |= {"ok": False, "error": "no RESULT line", "wall": wall}
        print("    FAILED: train.py printed no RESULT line")
        _append(record)
        return record

    payload = json.loads(match.group(1))
    record |= {"ok": True, "wall": wall, **payload}

    prior = best_so_far(results)
    if prior is None:
        verdict, delta = "FIRST", None
    else:
        delta = payload["bpm"] - prior["bpm"]
        verdict = "KEEP" if delta < 0 else "REVERT"

    record["verdict"] = verdict
    record["delta"] = round(delta, 5) if delta is not None else None
    _append(record)

    print(
        f"    bpm {payload['bpm']:.5f}  params {payload['params']:,}  "
        f"steps {payload['steps']:,}  ({payload['positions']:,} positions)"
    )
    if delta is not None:
        print(f"    vs best {prior['bpm']:.5f} (exp {prior['id']}): {delta:+.5f}  -> {verdict}")

    if verdict in ("FIRST", "KEEP"):
        shutil.copy2(TRAIN, BEST)
        print(f"    new best, promoted to best.py")
    else:
        shutil.copy2(BEST, TRAIN)
        print(f"    reverted train.py to best.py")

    return record


def _append(record: dict) -> None:
    with RESULTS.open("a") as f:
        f.write(json.dumps(record) + "\n")


def status() -> None:
    results = load_results()
    if not results:
        print("no experiments yet")
        return
    ok = [r for r in results if r.get("ok")]
    print(f"{len(results)} experiments, {len(ok)} successful, {len(results)-len(ok)} failed")
    if not ok:
        return
    print()
    print(f"{'rank':>4} {'exp':>4} {'bpm':>9} {'params':>12} {'steps':>8}  note")
    for rank, r in enumerate(sorted(ok, key=lambda r: r["bpm"])[:15], 1):
        print(
            f"{rank:>4} {r['id']:>4} {r['bpm']:>9.5f} {r['params']:>12,} "
            f"{r['steps']:>8,}  {r.get('note','')[:50]}"
        )
    print()
    baseline = ok[0]["bpm"]
    best = min(r["bpm"] for r in ok)
    print(f"first experiment {baseline:.5f} -> best {best:.5f}  ({best-baseline:+.5f} bits/move)")
    print(f"random baseline is {10.94270:.5f} bits/move (log2 of 1968 actions)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--note", default="", help="one line describing the hypothesis")
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    if args.status:
        status()
        return
    if args.restore:
        if not BEST.exists():
            sys.exit("no best.py yet")
        shutil.copy2(BEST, TRAIN)
        print("restored train.py from best.py")
        return

    for i in range(args.repeat):
        if args.repeat > 1:
            print(f"\n--- repeat {i+1}/{args.repeat} ---")
        run_once(note=args.note)


if __name__ == "__main__":
    main()
