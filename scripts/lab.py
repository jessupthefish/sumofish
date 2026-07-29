#!/usr/bin/env python
"""The lab. An experiment queue that runs the plan without being watched.

    systemctl --user start chess-gpu-lab      # run the plan
    sumofish-lab                              # look at it
    sumofish-lab watch                        # look at it, live

Everything below the search work needs the GPU, needs hours, and needs to
happen in a specific order because later steps read earlier results. Doing that
by hand means being present at every handoff, at 3am, for two days. This runs it
instead.

## What it is

An ordered list of jobs and a runner that walks it. Two kinds of job:

    command   a subprocess -- a training run, a match. Owns the GPU while it
              runs, gets a wall-clock timeout, and is interrupted with SIGTERM
              because `train.py` checkpoints on SIGTERM and killing it outright
              throws away hours.
    decision  a Python function that reads earlier results and returns facts
              the later jobs interpolate. This is where "which arm won" and
              "is that difference real" live, and it is why the plan can branch
              without anybody being awake.

State is in `runs/lab/state.json`: which jobs finished, what they concluded,
and the facts so far. Restarting the unit resumes from there rather than
starting over, so a reboot in the middle of a 40-hour training run costs the
run, not the plan.

## What it is allowed to do on its own

It measures freely and it changes exactly one thing: `runs/value.pt`, the
checkpoint the live bot plays, and only when a match of at least
`PROMOTE_MIN_GAMES` says the candidate is better with `PROMOTE_MIN_LOS`
confidence AND the Elo interval excludes zero. The previous checkpoint is kept
beside it. That is a file copy the bot picks up on its next game, and it is
reversible with another file copy.

It does not edit code, does not touch systemd units, and does not change the
engine's own defaults. A match that says "batch 256 is free" writes that
conclusion into the report; applying it is a code change and code changes are
not something to do unsupervised.

## The one rule that is not obvious

Nothing may edit `chessgpu/` while this is running. The matches import from the
working tree and spawn a process per game, so an edit half way through a match
means the first half of the games played a different engine than the second and
nothing in the log says so. The runner refuses to start a job while a foreign
`train.py` or `match.py` is alive, which covers accidental overlap; it cannot
protect you from an editor.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from elo import score_stats, tally  # noqa: E402

LAB = ROOT / "runs" / "lab"
STATE = LAB / "state.json"
EVENTS = LAB / "log.jsonl"

# Promotion gate. 300 games at these bounds resolves roughly a 25-Elo
# difference, which is about the smallest change worth deploying.
PROMOTE_MIN_GAMES = 300
PROMOTE_MIN_LOS = 0.95

HOUR = 3600


# ---------------------------------------------------------------------------
# reading other people's logs


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A record torn by a concurrent write. The next read gets it.
                pass
    return out


def train_progress(run: str) -> str:
    """Last step, loss and held-out loss of a training run."""
    rows = jsonl(ROOT / "runs" / run / "log.jsonl")
    steps = [r for r in rows if "loss" in r]
    evals = [r for r in rows if "puzzle_acc" in r]
    if not steps:
        return "starting"
    last = steps[-1]
    out = f"step {last['step']:,}  loss {last['loss']:.4f}"
    if evals:
        e = evals[-1]
        out += f"  puzzles {e['puzzle_acc']:.3f}"
        if "val_loss" in e:
            out += f"  val {e['val_loss']:.4f}"
    return out


def match_progress(name: str) -> str:
    """Score and Elo interval of a match, live."""
    games = jsonl(ROOT / "runs" / "matches" / name / "games.jsonl")
    if not games:
        return "starting"
    st = score_stats(*tally(games))
    return (
        f"{st['n']} games  {st['score'] * 100:.1f}%  "
        f"elo {st['elo']:+.1f} +-{st['err']:.1f}  LOS {st['los'] * 100:.0f}%"
    )


def match_verdict(name: str) -> dict:
    """Everything a decision needs to know about a finished match."""
    games = jsonl(ROOT / "runs" / "matches" / name / "games.jsonl")
    w, d, l = tally(games)
    st = score_stats(w, d, l)
    st["decisive"] = (
        st["n"] >= PROMOTE_MIN_GAMES
        and st["los"] >= PROMOTE_MIN_LOS
        and st["elo"] - st["err"] > 0
    )
    return st


# ---------------------------------------------------------------------------
# jobs


@dataclass
class Job:
    id: str
    what: str
    # A command job builds argv from the facts so far. A decision job returns
    # facts instead. Exactly one of these is set.
    argv: Callable[[dict], list[str]] | None = None
    decide: Callable[[dict], dict] | None = None
    timeout: float = 6 * HOUR
    probe: Callable[[], str] | None = None
    needs: list[str] = field(default_factory=list)


def python(*args: str) -> list[str]:
    return [str(ROOT / ".venv/bin/python"), *args]


def match_argv(name: str, *extra: str, games: int = 300, sims: int = 400) -> list[str]:
    return python(
        str(ROOT / "scripts/match.py"),
        "--name", name,
        "--games", str(games),
        "--sims", str(sims),
        "--elo0", "0", "--elo1", "25",
        *extra,
    )


# --- decisions -------------------------------------------------------------


def decide_causal(facts: dict) -> dict:
    """Which attention mask the 136M run should use.

    Judged on held-out loss rather than puzzle accuracy. At 20k steps puzzles
    are still mostly noise (+-1.5% at n=1000) while the held-out loss separates
    two architectures cleanly, and loss is what the arms were trained on.
    """
    arms = {}
    for causal in ("1", "0"):
        rows = jsonl(ROOT / f"runs/9M-sv-causal{causal}-ablation/log.jsonl")
        evals = [r for r in rows if "val_loss" in r]
        losses = [r for r in rows if "loss" in r]
        arms[causal] = {
            "val_loss": evals[-1]["val_loss"] if evals else None,
            "puzzle_acc": evals[-1]["puzzle_acc"] if evals else None,
            "train_loss": losses[-1]["loss"] if losses else None,
            "steps": losses[-1]["step"] if losses else 0,
        }

    usable = {c: a for c, a in arms.items() if a["val_loss"] is not None}
    if len(usable) < 2:
        # An arm that did not produce a held-out number cannot be compared, and
        # guessing here would silently pick an architecture for a 40-hour run.
        # Upstream's setting is the safe default and the one the pretrained
        # checkpoints use.
        return {"causal": "1", "causal_why": "an arm produced no val_loss; kept upstream's default",
                "causal_arms": arms}

    winner = min(usable, key=lambda c: usable[c]["val_loss"])
    gap = abs(usable["1"]["val_loss"] - usable["0"]["val_loss"])
    return {
        "causal": winner,
        "causal_why": (
            f"causal={winner} won on held-out loss "
            f"({usable[winner]['val_loss']:.4f} vs "
            f"{usable['0' if winner == '1' else '1']['val_loss']:.4f}, gap {gap:.4f})"
        ),
        "causal_arms": arms,
    }


def decide_promote(facts: dict) -> dict:
    """Swap the live checkpoint, but only if the match actually said so."""
    verdict = match_verdict("lab-136m-vs-current")
    candidate = ROOT / "runs/136M-sv/best.pt"

    if not verdict["decisive"]:
        return {"promoted": False, "promote_why": (
            f"not promoted: {verdict['n']} games, elo {verdict['elo']:+.1f} "
            f"+-{verdict['err']:.1f}, LOS {verdict['los'] * 100:.1f}% "
            f"(needs >={PROMOTE_MIN_GAMES} games, LOS >={PROMOTE_MIN_LOS * 100:.0f}%, "
            f"interval clear of zero)"), "promote_verdict": verdict}

    if not candidate.exists():
        return {"promoted": False, "promote_why": f"no candidate at {candidate}"}

    live = ROOT / "runs/value.pt"
    if live.exists():
        shutil.copyfile(live, ROOT / "runs/value.pt.previous")
    # Atomic, for the same reason promote.py is: a game starting mid-copy would
    # otherwise read a torn file.
    staged = ROOT / "runs/value.pt.staged"
    shutil.copyfile(candidate, staged)
    os.replace(staged, live)
    return {"promoted": True, "promote_why": (
        f"promoted 136M: elo {verdict['elo']:+.1f} +-{verdict['err']:.1f}, "
        f"LOS {verdict['los'] * 100:.1f}% over {verdict['n']} games. "
        f"Previous checkpoint kept at runs/value.pt.previous"), "promote_verdict": verdict}


def write_report(facts: dict) -> dict:
    """A single page saying what the lab concluded, for a human to read."""
    lines = ["# Lab report", ""]
    for name, title in (
        ("lab-batch", "Batch 256 vs 64, equal simulations"),
        ("lab-cpuct", "c_puct 3.0 vs 2.0"),
        ("lab-fpu", "fpu -0.5 vs -0.2"),
        ("lab-136m-vs-current", "136M vs the live 9M"),
    ):
        st = match_verdict(name)
        if not st["n"]:
            continue
        lines += [
            f"## {title}",
            "",
            f"- {st['n']} games, score {st['score'] * 100:.1f}%",
            f"- **elo {st['elo']:+.1f} +-{st['err']:.1f}**, LOS {st['los'] * 100:.1f}%",
            f"- {'A is better' if st['decisive'] else 'not resolved: treat as no difference'}",
            "",
        ]
    lines += ["## Decisions", ""]
    for key in ("causal_why", "promote_why"):
        if key in facts:
            lines.append(f"- {facts[key]}")
    lines += [
        "",
        "## Not applied automatically",
        "",
        "Match results that imply a *code* change are reported, never applied.",
        "If the batch match is flat, `CHESSGPU_BATCH` can go to 256 for ~36% more",
        "nps at no measured cost; that is an edit to `search_engine.py` and a",
        "human should make it.",
        "",
    ]
    (LAB / "report.md").write_text("\n".join(lines))
    return {"report": str(LAB / "report.md")}


# --- the plan --------------------------------------------------------------
#
# Ordered deliberately: the three cheap matches answer live-configuration
# questions in about an hour each, and they run BEFORE the long training job
# rather than behind it, so a two-day run does not sit in front of an answer
# that takes an hour to get.

PLAN: list[Job] = [
    Job(
        id="causal",
        what="pick the attention mask for the 136M run, on held-out loss",
        decide=decide_causal,
    ),
    Job(
        id="match-batch",
        what="batch 256 vs 64 at equal simulations: is a bigger batch free?",
        argv=lambda f: match_argv(
            "lab-batch", "--a-batch", "256", "--b-batch", "64",
            "--a-label", "batch256", "--b-label", "batch64",
        ),
        probe=lambda: match_progress("lab-batch"),
        timeout=4 * HOUR,
    ),
    Job(
        id="match-cpuct",
        what="c_puct 3.0 vs 2.0, never measured",
        argv=lambda f: match_argv(
            "lab-cpuct", "--a-cpuct", "3.0", "--b-cpuct", "2.0",
            "--a-label", "cpuct3", "--b-label", "cpuct2",
        ),
        probe=lambda: match_progress("lab-cpuct"),
        timeout=4 * HOUR,
    ),
    Job(
        id="match-fpu",
        what="first-play urgency -0.5 vs -0.2, never measured",
        argv=lambda f: match_argv(
            "lab-fpu", "--a-fpu", "-0.5", "--b-fpu", "-0.2",
            "--a-label", "fpu05", "--b-label", "fpu02",
        ),
        probe=lambda: match_progress("lab-fpu"),
        timeout=4 * HOUR,
    ),
    Job(
        id="train-136m",
        what="the 136M state-value run, ~35h at 821 samples/s",
        argv=lambda f: python(
            str(ROOT / "train.py"),
            "--target", "state_value", "--value-bins", "64",
            "--preset", "136M",
            # Batch 256 because 512 does not fit in 16GB, and because
            # throughput is flat across batch size anyway (8128/s at 256 vs
            # 7807/s at 1024 for the 9M), so the ceiling costs nothing.
            "--steps", "400000", "--batch-size", "256",
            "--workers", "8", "--lr", "3e-4", "--warmup", "2000",
            "--causal", f.get("causal", "1"),
            "--log-every", "500", "--eval-every", "20000",
            "--eval-puzzles", "1000", "--ckpt-every", "10000",
            "--val-batches", "32", "--compile", "1",
            "--run", "136M-sv", "--auto-resume",
        ),
        probe=lambda: train_progress("136M-sv"),
        # Generous: the estimate is 35h and contention with the live bot is
        # real. SIGTERM at the deadline makes train.py checkpoint and exit
        # cleanly, so hitting this is a stop, not a loss.
        timeout=50 * HOUR,
        needs=["causal"],
    ),
    Job(
        id="match-136m",
        what="the trained 136M against the live 9M, equal simulations",
        argv=lambda f: match_argv(
            "lab-136m-vs-current",
            "--a-value", str(ROOT / "runs/136M-sv/best.pt"),
            "--b-value", str(ROOT / "runs/value.pt"),
            "--a-label", "136M", "--b-label", "live-9M",
            games=400,
        ),
        probe=lambda: match_progress("lab-136m-vs-current"),
        # A 136M forward pass is roughly an order of magnitude dearer than a
        # 9M one and one side of every game pays it, so this is much slower per
        # game than the tuning matches. Generous on purpose: the SPRT stops it
        # early if the answer is clear, and a match cut short at the deadline
        # still leaves a valid, smaller sample in the log.
        timeout=20 * HOUR,
        needs=["train-136m"],
    ),
    Job(
        id="promote",
        what="swap the live checkpoint if, and only if, the match said so",
        decide=decide_promote,
        needs=["match-136m"],
    ),
    Job(
        id="report",
        what="write runs/lab/report.md",
        decide=write_report,
    ),
]


# ---------------------------------------------------------------------------
# state


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"done": {}, "facts": {}, "current": None, "started": None}


def save_state(state: dict) -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, STATE)


def event(kind: str, **fields) -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a") as fh:
        fh.write(json.dumps({"t": time.time(), "ev": kind, **fields}) + "\n")


# ---------------------------------------------------------------------------
# running


def foreign_gpu_jobs(own: set[int]) -> list[str]:
    """Training or match processes this runner did not start.

    Deliberately not `pgrep -f`: this file's own command line contains the
    strings being searched for, so pgrep matches itself, which is a trap this
    project has already fallen into twice. Reading /proc and excluding our own
    pids cannot make that mistake.
    """
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in own:
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        cmd = cmd.replace("\0", " ").strip()
        if not cmd:
            continue
        if "train.py" in cmd or "scripts/match.py" in cmd:
            found.append(cmd[:120])
    return found


def wait_for_quiet(poll: float = 60.0) -> None:
    """Hold until nothing else is using the card for an experiment.

    The live bot is not counted: it plays whether or not the lab is busy, that
    is its job, and every match here is fixed-simulation so contention costs
    wall clock and cannot bias a result.
    """
    own = {os.getpid()}
    announced = False
    while (busy := foreign_gpu_jobs(own)):
        if not announced:
            print(f"[lab] waiting for {len(busy)} foreign job(s): {busy[0]}", flush=True)
            event("waiting", jobs=busy)
            announced = True
        time.sleep(poll)
    if announced:
        print("[lab] card is free, continuing", flush=True)


def run_command(job: Job, facts: dict) -> tuple[bool, str]:
    argv = job.argv(facts)
    log_path = LAB / f"{job.id}.log"
    LAB.mkdir(parents=True, exist_ok=True)
    print(f"[lab] {job.id}: {' '.join(argv)}", flush=True)

    with log_path.open("a") as fh:
        fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(argv)} =====\n")
        fh.flush()
        proc = subprocess.Popen(
            argv, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
            # Its own process group, so the SIGTERM below reaches the whole
            # job -- train.py's dataloader workers included -- instead of just
            # the parent, which would leave orphans holding the GPU.
            start_new_session=True,
        )
        deadline = time.time() + job.timeout
        while proc.poll() is None:
            if time.time() > deadline:
                print(f"[lab] {job.id}: deadline reached, SIGTERM", flush=True)
                # SIGTERM, not SIGKILL: train.py finishes its step, runs a
                # final eval and checkpoints. That takes minutes and is the
                # difference between stopping a run and losing it.
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                break
            time.sleep(5)

    code = proc.returncode
    # A job stopped at its deadline exits non-zero and has still done its work.
    ok = code in (0, -signal.SIGTERM, 128 + signal.SIGTERM)
    return ok, f"exit {code}"


def run(only: str | None = None) -> int:
    state = load_state()
    state["started"] = state.get("started") or time.time()
    state["pid"] = os.getpid()
    save_state(state)
    event("lab-start", pid=os.getpid())

    for job in PLAN:
        if only and job.id != only:
            continue
        if job.id in state["done"]:
            continue
        missing = [n for n in job.needs if n not in state["done"]]
        if missing:
            print(f"[lab] {job.id}: skipped, needs {missing}", flush=True)
            continue

        state["current"] = job.id
        save_state(state)
        event("job-start", job=job.id, what=job.what)
        started = time.time()
        print(f"\n[lab] === {job.id}: {job.what} ===", flush=True)

        # Before decisions too, not just commands. A decision reads the logs of
        # the job before it, and a log being read while it is still being
        # written is a decision made on half the evidence -- which is exactly
        # how the first version of this would have picked an attention mask for
        # a 35-hour run from a 4,000-step sample.
        wait_for_quiet()

        if job.decide is not None:
            facts = job.decide(state["facts"])
            state["facts"].update(facts)
            ok, detail = True, "; ".join(
                str(v) for k, v in facts.items() if k.endswith("_why")
            ) or "done"
        else:
            ok, detail = run_command(job, state["facts"])

        state["done"][job.id] = {
            "ok": ok,
            "detail": detail,
            "seconds": round(time.time() - started),
            "finished": time.time(),
            "summary": job.probe() if job.probe else detail,
        }
        state["current"] = None
        save_state(state)
        event("job-end", job=job.id, ok=ok, detail=detail,
              seconds=state["done"][job.id]["seconds"])
        print(f"[lab] {job.id}: {'ok' if ok else 'FAILED'} -- {detail}", flush=True)

        if not ok:
            # Stop rather than run the rest against a broken premise. The queue
            # resumes from here once the cause is fixed.
            print(f"[lab] halting: {job.id} failed", flush=True)
            event("lab-halt", job=job.id)
            return 1

    event("lab-done")
    print("\n[lab] plan complete", flush=True)
    return 0


# ---------------------------------------------------------------------------
# looking at it


def render() -> str:
    state = load_state()
    lines = []
    total = len(PLAN)
    done = sum(1 for j in PLAN if j.id in state["done"])
    header = f"SumoFish lab   {done}/{total} jobs"
    if state.get("started"):
        header += f"   running {(time.time() - state['started']) / HOUR:.1f}h"
    lines += [header, "=" * max(60, len(header)), ""]

    for job in PLAN:
        record = state["done"].get(job.id)
        if record:
            mark = "done " if record["ok"] else "FAIL "
            detail = record.get("summary") or record["detail"]
            took = f"{record['seconds'] / 60:.0f}m" if record["seconds"] < HOUR \
                else f"{record['seconds'] / HOUR:.1f}h"
            lines.append(f"  [{mark}] {job.id:<16} {took:>6}  {detail}")
        elif state.get("current") == job.id:
            live = job.probe() if job.probe else "running"
            lines.append(f"  [ >>> ] {job.id:<16} {'':>6}  {live}")
        else:
            lines.append(f"  [     ] {job.id:<16} {'':>6}  {job.what}")

    facts = state.get("facts", {})
    notes = [v for k, v in facts.items() if k.endswith("_why")]
    if notes:
        lines += ["", "Decisions:"]
        lines += [f"  - {n}" for n in notes]

    report = LAB / "report.md"
    if report.exists():
        lines += ["", f"Report: {report}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", nargs="?", default="status",
                    choices=["run", "status", "watch", "reset"])
    ap.add_argument("--only", help="run a single job by id")
    ap.add_argument("--job", help="with reset: forget one job instead of all")
    args = ap.parse_args()

    if args.action == "run":
        sys.exit(run(args.only))
    if args.action == "watch":
        try:
            while True:
                print("\033[2J\033[H" + render(), end="", flush=True)
                time.sleep(5)
        except KeyboardInterrupt:
            return
    if args.action == "reset":
        state = load_state()
        if args.job:
            state["done"].pop(args.job, None)
            print(f"forgot {args.job}; it will run again")
        else:
            state = {"done": {}, "facts": {}, "current": None, "started": None}
            print("plan reset")
        save_state(state)
        return
    print(render(), end="")


if __name__ == "__main__":
    main()
