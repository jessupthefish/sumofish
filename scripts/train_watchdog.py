#!/usr/bin/env python
"""Detect a training run that is alive but no longer making progress.

systemd restarts the unit if the process *dies*. It cannot see the worse
failure: the process still running, the GPU still hot, and the step counter
frozen. That happens on a wedged CUDA kernel or a deadlocked dataloader worker,
and over an eight-hour unattended run it is the failure that costs the whole
night, because everything looks fine from the outside.

This compares the last logged step against the previous check. No progress in
STALL_MINUTES while the unit is active means restart; auto-resume picks up from
the last checkpoint.

Run from a timer. Read-only apart from its own state file.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# NEVER hardcode a run directory here. This was `runs/9M-causal/log.jsonl`, so
# the first run started under a different --run name would have read a stale,
# frozen log, concluded the step counter was not moving, and restarted a
# perfectly healthy process every 30 minutes forever -- while the journal
# showed a run being dutifully supervised. train.py publishes runs/active.json
# at startup; that is the only source of truth for which run is live.
ACTIVE = ROOT / "runs" / "active.json"

# Training has no unit of its own. It is a SUBPROCESS of chess-gpu-lab.service,
# which owns the GPU roadmap and starts training runs itself.
#
# This said `chess-gpu-train.service` until 2026-07-29, which stopped existing
# when the lab took over. The result was worse than a no-op: `is-active` returned
# non-zero, so main() logged "not active; nothing to watch" and returned, every
# five minutes, forever. The journal showed a watchdog dutifully running and it
# had never once looked at a training run -- including a 37-hour one, the exact
# case it was written for. A green timer is indistinguishable from a working
# watchdog, which is why this one now watches the PROCESS.
LAB_UNIT = "chess-gpu-lab.service"
STATE = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "chess-gpu/train_watchdog.json"

# Generous: a puzzle eval plus a checkpoint write plus a torch.compile after a
# restart can legitimately take several minutes with no new step logged.
STALL_MINUTES = 20
RESTART_COOLDOWN = 1800


def log(msg: str) -> None:
    print(f"[train-watchdog] {msg}", flush=True)


def training_pid() -> int | None:
    """PID of the live training process, or None if there is not one.

    Reads `runs/active.json`, which `train.py` publishes at startup, then checks
    /proc directly. Two guards, both learned the hard way:

    * `/proc/<pid>/cmdline` must actually contain `train.py`. A PID is reused, and
      a watchdog that restarts the lab because some unrelated process inherited
      the number would be worse than no watchdog.
    * Never compare this against the unit's MainPID. ExecStart runs python under
      `systemd-inhibit`, so MainPID is the wrapper and never matches -- a check
      that once disabled the supervision it was added to provide.
    """
    try:
        info = json.loads(ACTIVE.read_text())
        pid = int(info["pid"])
    except Exception:
        return None
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
    except OSError:
        return None            # process is gone
    if "train.py" not in cmdline:
        return None            # PID reused by something else
    return pid


def active_log() -> Path | None:
    """Log file of the currently-live run, or None if we cannot be sure.

    Returning None makes the caller do nothing. That is the correct bias: a
    watchdog that is unsure must never act, because its only action is
    destructive.
    """
    try:
        info = json.loads(ACTIVE.read_text())
    except (OSError, ValueError):
        return None
    log = Path(info.get("log", ""))
    if not log.exists():
        return None
    # Guard against a stale active.json left by a previous run: the pid it
    # names must still be alive.
    #
    # Do NOT compare against the unit's MainPID. ExecStart runs python under
    # systemd-inhibit, so MainPID is the WRAPPER and never equals the trainer's
    # own getpid(). Comparing them made this return None on every check, which
    # disables the watchdog entirely -- a fix that quietly removed the
    # supervision it was meant to repair.
    pid = info.get("pid")
    if not isinstance(pid, int) or not Path(f"/proc/{pid}").exists():
        print(f"[train-watchdog] active.json names dead pid {pid}; declining to act", flush=True)
        return None
    return log


def last_step() -> int | None:
    LOG = active_log()
    if LOG is None or not LOG.exists():
        return None
    step = None
    with LOG.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if "step" in rec:
                step = rec["step"]
    return step


def main() -> int:
    pid = training_pid()
    if pid is None:
        log("no live training process; nothing to watch")
        return 0

    now = time.time()
    step = last_step()
    if step is None:
        log("no log yet")
        return 0

    try:
        state = json.loads(STATE.read_text())
    except (OSError, ValueError):
        state = {}

    prev_step = state.get("step")
    prev_time = state.get("time", now)

    if prev_step is None or step != prev_step:
        state.update({"step": step, "time": now})
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(state))
        log(f"progressing: step {step:,}")
        return 0

    stalled_for = now - prev_time
    if stalled_for < STALL_MINUTES * 60:
        log(f"step {step:,} unchanged for {stalled_for/60:.1f} min (limit {STALL_MINUTES})")
        return 0

    last_restart = state.get("last_restart", 0)
    if now - last_restart < RESTART_COOLDOWN:
        log(f"stalled but restarted {int(now-last_restart)}s ago; holding off")
        return 0

    # Restart the LAB, not the training process directly.
    #
    # SIGTERMing train.py looks more surgical and is wrong: train.py checkpoints
    # and exits 0 on SIGTERM, so lab.py would see a clean exit, mark the job
    # COMPLETED, and move to the next one -- silently abandoning the rest of the
    # run. Restarting the lab makes it resume its plan from runs/lab/state.json
    # and re-run the job, and train.py's --auto-resume picks up from the last
    # checkpoint. That is the recovery the two were designed for.
    log(f"STALLED at step {step:,} for {stalled_for/60:.1f} min "
        f"(pid {pid}); restarting {LAB_UNIT}")
    subprocess.run(["systemctl", "--user", "restart", LAB_UNIT], check=False)
    state.update({"last_restart": now, "time": now})
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
