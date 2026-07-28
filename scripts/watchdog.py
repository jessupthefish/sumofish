#!/usr/bin/env python3
"""Watchdog for the chess-gpu lichess bot.

Works around two lichess-bot bugs that are both closed without a fix:

  #1184 - a transient stream drop makes the bot exit the game loop while the
          game is still live. The bot process stays up and looks healthy; it
          has simply stopped playing, and flags on time.
  #1101 - after a restart, not all in-progress games are picked back up.

Neither is visible from the process state, so the only reliable signal is
lichess's own view: ask whether it is our turn, and whether it has been our
turn for implausibly long. If so, restart the unit.

Run from a systemd timer. Reads LICHESS_BOT_TOKEN from the environment and
never logs it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://lichess.org/api/account/playing"
UNIT = "chess-gpu-bot.service"
STATE = Path(
    os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")
) / "chess-gpu/watchdog.json"

# How long it may plausibly be our turn before we call the bot stuck. The
# engine answers in well under a second, so anything past this is a hang, not
# thinking. Kept above lichess's own 30s abort window so we don't fight it.
STUCK_SECONDS = 120

# Don't restart more than once per this interval, so a genuinely broken bot
# doesn't turn into a restart loop that also loses every game it picks up.
RESTART_COOLDOWN = 600


def log(msg: str) -> None:
    print(f"[watchdog] {msg}", flush=True)


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state))


def fetch_playing(token: str) -> list[dict] | None:
    req = urllib.request.Request(API, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.load(resp).get("nowPlaying", [])
    except urllib.error.HTTPError as e:
        # Deliberately does not echo the response body, which can contain the
        # request that carried the token.
        log(f"HTTP {e.code} from lichess; skipping this cycle")
        return None
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        log(f"could not reach lichess ({type(e).__name__}); skipping this cycle")
        return None


def unit_active() -> bool:
    r = subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", UNIT], check=False
    )
    return r.returncode == 0


def restart() -> None:
    """Hard restart.

    The unit now stops with SIGINT and a 180s drain so ordinary restarts do not
    abandon live games. That is exactly wrong here: this path only runs when the
    game loop is already dead, so a graceful drain would block for the full
    timeout while the clock runs. Kill it and let Restart=always bring it back.
    """
    log(f"restarting {UNIT} (SIGKILL: the game loop is already dead)")
    subprocess.run(["systemctl", "--user", "kill", "-s", "SIGKILL", UNIT], check=False)
    subprocess.run(["systemctl", "--user", "restart", UNIT], check=False)


def main() -> int:
    token = os.environ.get("LICHESS_BOT_TOKEN")
    if not token:
        log("LICHESS_BOT_TOKEN is not set; nothing to do")
        return 1

    if not unit_active():
        # systemd's own Restart=always owns this case. Restarting here would
        # just race it.
        log(f"{UNIT} is not active; leaving it to systemd")
        return 0

    games = fetch_playing(token)
    if games is None:
        return 0

    now = time.time()
    state = load_state()
    seen: dict[str, float] = state.get("our_turn_since", {})
    fresh: dict[str, float] = {}

    stuck = []
    for game in games:
        gid = game.get("gameId")
        if not gid or not game.get("isMyTurn"):
            continue
        since = seen.get(gid, now)
        fresh[gid] = since
        waited = now - since
        if waited > STUCK_SECONDS:
            stuck.append((gid, waited))

    state["our_turn_since"] = fresh

    if stuck:
        last = state.get("last_restart", 0)
        if now - last < RESTART_COOLDOWN:
            log(
                f"{len(stuck)} game(s) stuck but restarted "
                f"{int(now - last)}s ago; holding off"
            )
        else:
            for gid, waited in stuck:
                log(f"game {gid}: our turn for {int(waited)}s")
            restart()
            state["last_restart"] = now
            state["our_turn_since"] = {}

    save_state(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
