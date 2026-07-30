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

# How long it may plausibly be our turn, WITHOUT THE POSITION CHANGING, before we
# call the bot stuck.
#
# Measured 2026-07-29 over 3,191 consecutive moves at 15+10 with concurrency 2:
# median 12s between our moves, p99 54s, and exactly one interval above 120s. So
# 120 is a comfortable threshold -- but only if it measures ONE turn. See the
# `last_move` tracking below for why it did not.
#
# (The comment this replaces said "the engine answers in well under a second",
# which was true at 1+0 and has not been true since the bot moved to 15+10 and
# ~30s of thinking per move.)
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


def unit_state() -> str:
    """`systemctl is-active` output: active, inactive, failed, activating...

    Reported rather than reduced to a boolean, because `failed` and `inactive`
    need opposite responses and the old code collapsed them into "not active".
    """
    r = subprocess.run(
        ["systemctl", "--user", "is-active", UNIT],
        capture_output=True, text=True, check=False,
    )
    return (r.stdout or r.stderr).strip() or "unknown"


def main() -> int:
    token = os.environ.get("LICHESS_BOT_TOKEN")
    if not token:
        log("LICHESS_BOT_TOKEN is not set; nothing to do")
        return 1

    if not unit_active():
        # `Restart=always` does NOT own every inactive state, and believing it did
        # is what cost ninety minutes of rated downtime once already.
        #
        # The unit carries StartLimitBurst=10 in StartLimitIntervalSec=300. A
        # crash loop -- a NameError at import, say, which has happened here --
        # burns through those ten restarts in five minutes and systemd gives up,
        # leaving the unit `failed`. That is not active, so the old code logged
        # "leaving it to systemd" and returned, deferring to the one mechanism
        # that had already stopped trying. Nothing else watched unit state.
        #
        # So: distinguish the states. `failed` needs reset-failed to clear the
        # rate limit before start will do anything; anything else inactive is
        # either deliberate (someone stopped it) or transient (activating), and
        # neither is ours to touch.
        state = unit_state()
        if state != "failed":
            log(f"{UNIT} is {state}; not ours to restart")
            return 0
        log(f"{UNIT} has FAILED -- systemd hit its start limit and gave up. "
            f"Clearing the limit and starting it.")
        subprocess.run(["systemctl", "--user", "reset-failed", UNIT], check=False)
        subprocess.run(["systemctl", "--user", "start", UNIT], check=False)
        return 0

    games = fetch_playing(token)
    if games is None:
        return 0

    now = time.time()
    state = load_state()
    seen: dict = state.get("our_turn_since", {})
    fresh: dict = {}

    stuck = []
    for game in games:
        gid = game.get("gameId")
        if not gid or not game.get("isMyTurn"):
            continue
        # The clock must measure ONE turn, and the only way to know a turn ended is
        # that the position changed. Timing "how long has isMyTurn been true across
        # polls" is a different and much larger quantity: it is our turn about half
        # the time, this runs once a minute, so three consecutive polls landing on
        # three DIFFERENT turns read as one 120-second turn. That bug killed 69
        # healthy games in 36 hours on 2026-07-29, every one of them mid-move, and
        # `systemctl kill` meant systemd's NRestarts stayed at 0 the whole time so
        # nothing ever showed it.
        marker = game.get("lastMove") or game.get("fen") or ""
        prev = seen.get(gid)
        if isinstance(prev, dict) and prev.get("move") == marker:
            since = prev.get("since", now)
        else:
            # A new turn (or the first sighting). Start the clock now.
            since = now
        fresh[gid] = {"move": marker, "since": since}
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
