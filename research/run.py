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

# Measured 2026-07-28 from three identical baseline runs: spread 0.00961,
# stdev 0.00483 bits. A result must beat the incumbent by more than this to be
# believed. Re-measure if the hardware or the time budget changes.
NOISE_FLOOR = 0.015


# The FROZEN block of train.py, byte for byte. `program.md` rule 3 forbids
# touching it and nothing enforced that, which made the rule a suggestion: an
# agent could lengthen TIME_BUDGET_S, widen the eval, or repoint the bags, and
# every number afterwards would be incomparable to every number before with no
# record that the yardstick had moved. Moving the yardstick to make the metric
# improve is the exact failure this whole harness exists to prevent, so it is
# checked rather than asked for.
FROZEN_MARKER = "# EDITABLE. Everything below here is fair game."


def frozen_region(text: str) -> str:
    """Everything above the EDITABLE marker."""
    head, sep, _tail = text.partition(FROZEN_MARKER)
    if not sep:
        raise SystemExit("research/train.py has lost its EDITABLE marker; refusing to run")
    return head


def check_frozen() -> str:
    """Abort unless train.py's frozen half matches best.py's.

    Compared against `best.py` rather than a hardcoded hash, because best.py is
    the last thing the harness itself promoted and therefore the last frozen
    block it has already vouched for. That makes the check self-maintaining: a
    deliberate change to the frozen block is made once, promoted once, and then
    becomes the new reference, while an undeclared edit between runs is caught.
    """
    current = hashlib.sha256(frozen_region(TRAIN.read_text()).encode()).hexdigest()[:12]
    if not BEST.exists():
        return current
    reference = hashlib.sha256(frozen_region(BEST.read_text()).encode()).hexdigest()[:12]
    if current != reference:
        raise SystemExit(
            "research/train.py's FROZEN block differs from best.py's.\n"
            f"  train.py {current}   best.py {reference}\n"
            "Rule 3: TIME_BUDGET_S, SEED, EVAL_BATCHES, evaluate() and the bag\n"
            "paths are the yardstick. Changing them makes every past number\n"
            "incomparable. Revert them, or if the change is deliberate, say so\n"
            "in a note and reset the leaderboard on purpose."
        )
    return current


def budget_from_train_py() -> int:
    m = re.search(r"^TIME_BUDGET_S\s*=\s*(\d+)", TRAIN.read_text(), re.M)
    return int(m.group(1)) if m else 300


def load_results() -> list[dict]:
    if not RESULTS.exists():
        return []
    return [json.loads(line) for line in RESULTS.read_text().splitlines() if line.strip()]


def check_editable_region(text: str) -> list[str]:
    """Reject edits that move the yardstick even though they are "below the line".

    `check_frozen` hashes the region above the EDITABLE marker, which covers the
    constants and the import of the metric. It cannot see two moves that defeat
    the metric from inside the editable region:

    * `def evaluate(...)` -- Python lets a later definition shadow the imported
      name, so redefining it here silently replaces the yardstick.
    * rebinding a frozen constant -- `EVAL_BATCHES = 4` below the marker shadows
      the frozen one at runtime, and averaging over a tenth of the held-out set
      is not the same measurement.

    Both are cheap to detect syntactically and neither has a legitimate use.
    """
    import ast as _ast

    marker = text.find(FROZEN_MARKER)
    if marker < 0:
        return ["the EDITABLE marker is missing entirely"]
    # Line number where the editable region starts.
    start_line = text[:marker].count("\n") + 1

    frozen_names = {
        "TIME_BUDGET_S", "SEED", "EVAL_BATCHES", "EVAL_BATCH_SIZE",
        "TRAIN_BAG", "VAL_BAG",
    }
    problems: list[str] = []
    try:
        tree = _ast.parse(text)
    except SyntaxError as exc:
        return [f"train.py does not parse: {exc}"]

    for node in _ast.walk(tree):
        lineno = getattr(node, "lineno", 0)
        if lineno <= start_line:
            continue
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            if node.name == "evaluate" and not _is_shim(node):
                problems.append(
                    f"line {lineno}: `def evaluate` in the editable region. The "
                    f"metric lives in research/harness_eval.py and redefining it "
                    f"here would shadow the import and move the yardstick."
                )
        if isinstance(node, _ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, _ast.Name) and tgt.id in frozen_names:
                    problems.append(
                        f"line {lineno}: rebinds the frozen constant {tgt.id}. The "
                        f"value above the marker is checked; this shadows it."
                    )
    return problems


def _is_shim(node) -> bool:
    """True for the one-line delegation to harness_eval, which is allowed."""
    for sub in node.body:
        if isinstance(sub, __import__("ast").Return):
            src = __import__("ast").dump(sub)
            if "harness_eval" in src:
                return True
    return False


def check_incumbent(results: list[dict]) -> list[str]:
    """Does `best.py` match the run the ledger says is the incumbent?

    `best_so_far` reads the last FIRST/KEEP from `results.jsonl` and uses its
    score as the bar. That is only meaningful if the FILE on disk is the file that
    run produced. On 2026-07-29 they disagreed: the last KEEP was exp 5 (bpm
    4.71635, a 0.00026 delta -- noise), while `best.py` was byte-identical to the
    baseline attempt 0001. So the bar sat about a sigma BELOW the incumbent's true
    mean, and clearing it required roughly 4 sigma. Nothing could be promoted, and
    the leaderboard read as a run of failed hypotheses rather than a locked
    harness.

    Verdicts recorded under an older rule cannot be trusted, so this refuses to
    guess and asks for a deliberate reset instead.
    """
    inc = best_so_far(results)
    if inc is None:
        return []
    # The field is `snapshot`, e.g. "0005-e62086c33030.py". It is the copy of
    # train.py as that run saw it, which is exactly what best.py should equal if
    # that run really is the incumbent.
    attempt = inc.get("snapshot")
    if not attempt:
        return ["the incumbent record names no snapshot, so it cannot be checked"]
    apath = ATTEMPTS / attempt if not str(attempt).startswith("/") else Path(attempt)
    if not apath.exists():
        return [f"the incumbent's attempt file is missing: {apath}"]
    if _digest(BEST.read_bytes()) != _digest(apath.read_bytes()):
        return [
            f"best.py does NOT match the ledger's incumbent ({attempt}).\n"
            f"    The bar and the file on disk disagree, so no promotion is "
            f"possible and the leaderboard is misleading.\n"
            f"    Fix deliberately: either restore best.py from {attempt}, or "
            f"re-baseline by recording a fresh FIRST for what is actually on disk."
        ]
    return []


def _digest(b: bytes) -> str:
    import hashlib as _h

    return _h.sha256(b).hexdigest()


def best_so_far(results: list[dict]) -> dict | None:
    """The score of whatever is actually in `best.py`, not the global minimum.

    This returned `min(ok, key=bpm)` over every run ever made, which is a
    different thing and a self-sealing one. A run that scores below the
    incumbent but not by more than NOISE_FLOOR is correctly NOT promoted -- the
    file on disk does not change -- but under the old rule its bpm still became
    the bar. So the bar drifts down to a number no promoted file ever achieved,
    and after a few lucky rolls nothing can be promoted again, forever. The
    leaderboard then shows a long tail of REVERT verdicts that look like
    hypotheses failing when they are the harness having locked itself.

    The incumbent is the last run whose verdict actually moved `best.py`.
    """
    promoted = [r for r in results
                if r.get("ok") and r.get("verdict") in ("FIRST", "KEEP")]
    return promoted[-1] if promoted else None


def run_once(note: str = "") -> dict:
    ATTEMPTS.mkdir(exist_ok=True)
    check_frozen()
    source = TRAIN.read_text()

    # Two guards beyond the frozen-block hash. Both refuse rather than warn: a
    # harness that reports a number it cannot stand behind is worse than one that
    # stops, and every failure this project has had came from something that
    # carried on.
    problems = check_editable_region(source)
    if problems:
        for prob in problems:
            print(f"REFUSING: {prob}", file=sys.stderr)
        raise SystemExit(2)

    stale = check_incumbent(load_results())
    if stale:
        for prob in stale:
            print(f"REFUSING: {prob}", file=sys.stderr)
        raise SystemExit(2)

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
        # Must beat the incumbent by more than the measured noise floor, not
        # merely by any amount. Identical code re-run scores +-0.005 bits, so
        # `delta < 0` promotes coin flips. Over a night of experiments that
        # ratchets on randomness: best.py accumulates meaningless edits, the
        # leaderboard shows steady progress, and the model is no better.
        # This already happened once (exp 5 promoted on a 0.00026 delta).
        if delta < -NOISE_FLOOR:
            verdict = "KEEP"
        elif delta < 0:
            verdict = "NOISE"
        else:
            verdict = "REVERT"

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
        print("    new best, promoted to best.py")
    else:
        shutil.copy2(BEST, TRAIN)
        if verdict == "NOISE":
            print(f"    better but within noise (|{delta:.5f}| < {NOISE_FLOOR}); reverted")
        else:
            print("    reverted train.py to best.py")

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
