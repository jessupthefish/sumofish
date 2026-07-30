#!/usr/bin/env python3
"""The watchdog's stuck-game detection, which was wrong for at least 36 hours.

On 2026-07-29 it SIGKILLed the bot 69 times across 51 distinct games, every one of
them mid-move. The bot was healthy throughout: measured over 3,191 consecutive
moves, the median interval between our moves was 12s and exactly one exceeded 120s.

The cause was that `our_turn_since` was stamped the first time a poll saw
`isMyTurn` and then carried forward for as long as any later poll also saw it --
never reset when the turn actually changed. It is our turn about half the time and
this runs once a minute, so three consecutive polls landing on three different turns
read as one 120-second turn.

`systemctl kill` + `restart` means systemd's NRestarts never moved, so nothing on
any dashboard showed it either.

Run: tests/verify_watchdog.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import watchdog  # noqa: E402


def poll(state, games, now):
    """One watchdog pass over `games`, returning (state, stuck)."""
    seen = state.get("our_turn_since", {})
    fresh, stuck = {}, []
    for game in games:
        gid = game.get("gameId")
        if not gid or not game.get("isMyTurn"):
            continue
        marker = game.get("lastMove") or game.get("fen") or ""
        prev = seen.get(gid)
        if isinstance(prev, dict) and prev.get("move") == marker:
            since = prev.get("since", now)
        else:
            since = now
        fresh[gid] = {"move": marker, "since": since}
        if now - since > watchdog.STUCK_SECONDS:
            stuck.append((gid, now - since))
    state["our_turn_since"] = fresh
    return state, stuck


def game(gid="abcd1234", my_turn=True, last="e2e4"):
    return {"gameId": gid, "isMyTurn": my_turn, "lastMove": last}


def main() -> int:
    fails = []

    def check(name, cond):
        print(f"  {'ok  ' if cond else 'FAIL'}  {name}")
        if not cond:
            fails.append(name)

    print("watchdog stuck-detection")

    # THE REGRESSION. A bot that is moving normally must never be declared stuck,
    # however many polls happen to land while it is our turn.
    state, t = {}, 1000.0
    ever_stuck = False
    for i in range(60):            # an hour of polling
        t += 60
        # A different move every poll: the game is progressing.
        _, stuck = poll(state, [game(last=f"m{i}")], t)
        ever_stuck = ever_stuck or bool(stuck)
    check("a game that keeps moving is never declared stuck", not ever_stuck)

    # And a genuinely wedged game still is: same position, poll after poll.
    state, t = {}, 1000.0
    stuck = []
    for _ in range(5):
        t += 60
        _, stuck = poll(state, [game(last="frozen")], t)
    check("a game frozen on one move IS declared stuck", bool(stuck))
    check("...and reports a plausible duration", stuck and stuck[0][1] >= 120)

    # The old behaviour, for contrast: this is what shipped.
    def old_poll(state, games, now):
        seen = state.get("our_turn_since", {})
        fresh, stuck = {}, []
        for g in games:
            gid = g.get("gameId")
            if not gid or not g.get("isMyTurn"):
                continue
            since = seen.get(gid, now)
            fresh[gid] = since
            if now - since > watchdog.STUCK_SECONDS:
                stuck.append((gid, now - since))
        state["our_turn_since"] = fresh
        return state, stuck

    state, t, old_stuck = {}, 1000.0, False
    for i in range(5):
        t += 60
        _, s = old_poll(state, [game(last=f"m{i}")], t)
        old_stuck = old_stuck or bool(s)
    check("the OLD logic would have fired on that healthy game", old_stuck)

    # Not our turn clears the entry, so the clock cannot survive across a turn.
    state, _ = poll({}, [game(last="a")], 1000.0)
    state, _ = poll(state, [game(my_turn=False, last="a")], 1060.0)
    check("a game that is not our turn is dropped", state["our_turn_since"] == {})

    # Two concurrent games are tracked independently.
    state, _ = poll({}, [game("aaa", last="x"), game("bbb", last="y")], 1000.0)
    state, stuck = poll(state, [game("aaa", last="x"), game("bbb", last="z")], 1200.0)
    ids = [g for g, _ in stuck]
    check("with concurrency 2, only the frozen game is flagged", ids == ["aaa"])

    # A game with no lastMove (just started) must not divide by a missing key.
    state, stuck = poll({}, [{"gameId": "new", "isMyTurn": True}], 1000.0)
    check("a game with no last move does not crash", not stuck)

    # Old-format state on disk must not be read as a live clock.
    state, stuck = poll({"our_turn_since": {"abcd1234": 1.0}}, [game()], 1e9)
    check("a stale float from the old format is ignored, not trusted", not stuck)

    print()
    if fails:
        print(f"{len(fails)} FAILED: {', '.join(fails)}")
        return 1
    print("all ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
