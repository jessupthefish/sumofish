#!/usr/bin/env python
"""Serialise the GPU between the lichess bot and every lab/match/training job.

    scripts/gpu_lock.py status
    scripts/gpu_lock.py run --drain-bot -- scripts/match.py --name x --time 0.5 ...
    scripts/gpu_lock.py run -- train.py --target state_value ...

Why this exists. `rust/src/tree.rs` checks the search deadline BETWEEN batches,
never mid-batch, so if a GPU callback stalls the check never runs. Under
contention that is not theoretical: deduped across `logs/games/` and
`logs/games-superseded/`, SumoFish has lost 10 rated games on time at 900+10 and
abandoned 23 more without a first move, and every 08-07 and 08-08 forfeit falls
inside a concurrent lab job. One game shows a 290-second move against a
~30-second budget. Separately, PHILOSOPHY makes an idle machine a *validity*
requirement for any wall-clock match, not a nicety.

Two guarantees:

  1. Two jobs holding this lock cannot run at once (flock, so it survives kill -9
     and is released by the kernel when the holder dies).
  2. With `--drain-bot`, the bot is down for the duration and comes back after,
     including if the wrapped command crashes, raises, or is signalled.

**Draining the bot is two commands and the order matters.** LAB-NOTES 2026-08-01:
`systemctl kill --kill-who=main --signal=SIGINT` drains gracefully (lichess-bot
installs a handler for SIGINT only, and `stop` signals the whole cgroup, taking
the engine subprocess out mid-game and abandoning a live rated game) but does NOT
mark the unit stopped, so `Restart=always` puts it back ten seconds later. The
correct sequence is drain, wait for the main pid to exit, THEN `stop`. Checking
`is-active` immediately after the drain is not enough; the restart lands later.

**Never `systemctl --user disable`.** Every unit here is a symlink from
`~/.config/systemd/user/` into the repo's `systemd/`, and `disable` removes all
symlinks pointing at the unit, deleting the unit file.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "sumofish"
LOCK = STATE_DIR / "gpu.lock"
LEDGER = STATE_DIR / "gpu-lock-ledger.jsonl"
BOT = "sumofish-bot.service"


def sysctl(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["systemctl", "--user", *args],
                          capture_output=True, text=True, check=check)


def bot_active() -> bool:
    return sysctl("is-active", BOT).stdout.strip() == "active"


def bot_main_pid() -> int:
    out = sysctl("show", BOT, "-p", "MainPID", "--value").stdout.strip()
    try:
        return int(out)
    except ValueError:
        return 0


def live_games() -> list[str]:
    """Game ids the bot currently has in flight, best effort from its log."""
    log = ROOT / "lichess-bot/lichess_bot_auto_logs/lichess-bot.log"
    if not log.exists():
        return []
    import re
    started, finished = [], set()
    try:
        tail = log.read_text(errors="replace")[-400_000:]
    except OSError:
        return []
    for m in re.finditer(r"\+\+\+ https://lichess\.org/(\w+)", tail):
        started.append(m.group(1))
    for m in re.finditer(r"--- https://lichess\.org/(\w+)", tail):
        finished.add(m.group(1))
    return [g for g in started if g not in finished]


def drain_bot(timeout: float) -> dict:
    """Drain, wait, then stop. Returns what happened, for the ledger."""
    rec = {"was_active": bot_active(), "games_at_drain": [], "drained": False,
           "abandoned": False, "at": time.time()}
    if not rec["was_active"]:
        print("  bot: already down")
        return rec

    rec["games_at_drain"] = live_games()
    if rec["games_at_drain"]:
        print(f"  bot: {len(rec['games_at_drain'])} game(s) in flight: "
              f"{', '.join(rec['games_at_drain'])}")
    print(f"  bot: draining (SIGINT to main pid), up to {timeout:.0f}s ...")
    sysctl("kill", "--kill-who=main", "--signal=SIGINT", BOT)

    deadline = time.time() + timeout
    while time.time() < deadline:
        if bot_main_pid() == 0:
            rec["drained"] = True
            break
        time.sleep(2)

    if not rec["drained"]:
        # It did not finish its games in time. Stopping now takes the engine out
        # mid-game and concedes, which is a real rating cost and must be recorded
        # rather than absorbed: a conceded loss is indistinguishable in
        # rating.jsonl from one the engine earned.
        rec["abandoned"] = True
        print(f"  bot: did NOT drain within {timeout:.0f}s; stopping anyway and "
              f"recording the abandoned games")

    # `stop` is what makes the unit STAY down. `kill` alone leaves Restart=always
    # to bring it back in ~10s, which is the 2026-08-01 trap.
    sysctl("stop", BOT)
    for _ in range(15):
        if not bot_active():
            break
        time.sleep(1)
    print(f"  bot: {'down' if not bot_active() else 'STILL ACTIVE (check manually)'}")
    return rec


def restore_bot(rec: dict) -> None:
    if not rec.get("was_active"):
        print("  bot: was down before, leaving it down")
        return
    print("  bot: restarting ...")
    sysctl("start", BOT)
    time.sleep(2)
    print(f"  bot: {'active' if bot_active() else 'FAILED TO START (check journal)'}")


def note(entry: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


def status() -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    holder = None
    with LOCK.open("a+") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fh, fcntl.LOCK_UN)
        except BlockingIOError:
            fh.seek(0)
            holder = fh.read().strip() or "(unknown)"
    print(f"lock: {'HELD by ' + holder if holder else 'free'}  ({LOCK})")
    print(f"bot:  {'active' if bot_active() else 'inactive'}  (pid {bot_main_pid()})")
    games = live_games()
    if games:
        print(f"      {len(games)} game(s) in flight: {', '.join(games)}")
    # Anything else on the card is a contention risk this lock does not cover,
    # e.g. a transient systemd-run job started outside the wrapper.
    smi = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader"], capture_output=True, text=True)
    if smi.returncode == 0 and smi.stdout.strip():
        print("gpu processes:")
        for line in smi.stdout.strip().splitlines():
            print(f"      {line}")
    return 0


def run(cmd: list[str], drain: bool, wait: float, drain_timeout: float) -> int:
    if not cmd:
        print("nothing to run", file=sys.stderr)
        return 2
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    fh = LOCK.open("a+")

    deadline = time.time() + wait
    while True:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            fh.seek(0)
            who = fh.read().strip() or "(unknown)"
            if time.time() >= deadline:
                print(f"gpu lock held by {who}; waited {wait:.0f}s, giving up",
                      file=sys.stderr)
                return 2
            print(f"  waiting for gpu lock (held by {who}) ...")
            time.sleep(5)

    fh.seek(0); fh.truncate()
    fh.write(f"pid {os.getpid()}: {' '.join(cmd)}\n"); fh.flush()

    drained: dict = {}
    rc = 1
    try:
        if drain:
            drained = drain_bot(drain_timeout)
        print(f"  running: {' '.join(cmd)}")
        rc = subprocess.run(cmd, cwd=ROOT).returncode
    except KeyboardInterrupt:
        print("\n  interrupted", file=sys.stderr)
        rc = 130
    finally:
        # Restore FIRST, then release, so the next waiter never sees a free lock
        # with the bot still down.
        if drain:
            restore_bot(drained)
        note({"at": started, "finished": time.time(),
              "seconds": round(time.time() - started, 1),
              "cmd": cmd, "rc": rc, "drained_bot": bool(drain),
              "bot_was_active": drained.get("was_active"),
              "games_at_drain": drained.get("games_at_drain", []),
              "games_abandoned": drained.get("abandoned", False)})
        fh.seek(0); fh.truncate(); fh.close()
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("status")
    r = sub.add_parser("run")
    r.add_argument("--drain-bot", action="store_true",
                   help="take the bot down for the duration and restore it "
                        "after. REQUIRED for any wall-clock match: PHILOSOPHY "
                        "makes an idle machine a validity condition, not a "
                        "preference. Not needed for fixed-simulation work, "
                        "where contention costs time and not validity.")
    r.add_argument("--wait", type=float, default=0.0,
                   help="seconds to wait for the lock before giving up")
    r.add_argument("--drain-timeout", type=float, default=900.0,
                   help="seconds to let the bot finish its games before "
                        "stopping it anyway and recording the abandonment")
    r.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()

    if a.mode == "status":
        return status()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    # A bare `scripts/match.py` runs under the script's shebang, which is the
    # SYSTEM python and has no torch. The docstring's own examples are written
    # that way, and the first fused gate died on `No module named 'torch'`
    # with the bot already drained. Route a .py through this interpreter.
    if cmd and cmd[0].endswith(".py"):
        cmd = [sys.executable, *cmd]
    return run(cmd, a.drain_bot, a.wait, a.drain_timeout)


if __name__ == "__main__":
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    raise SystemExit(main())
