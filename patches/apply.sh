#!/usr/bin/env bash
# Reapply local patches to the lichess-bot clone. Run after any git pull there.
# Idempotent: skips patches that are already applied.
set -euo pipefail
BOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../lichess-bot" && pwd)"
cd "$BOT"

if grep -q "PATCHED (see patches/0001" lib/lichess_bot.py; then
  echo "0001 (#1184 stream drop): already applied"
else
  python3 - <<'PY'
import pathlib, sys
p = pathlib.Path("lib/lichess_bot.py"); t = p.read_text()
old = """                    stopped = isinstance(e, StopIteration)
                    stay_in_game = not stopped and (move_attempted or game_is_active(li, game.id))"""
new = """                    # PATCHED (see patches/0001-lichess-bot-1184-stream-drop.patch):
                    # upstream short-circuits game_is_active() away on
                    # StopIteration and abandons live games on a network blip.
                    # Always ask lichess whether the game is really over.
                    stay_in_game = move_attempted or game_is_active(li, game.id)"""
if old not in t:
    sys.exit("0001: anchor not found -- upstream changed, re-derive the patch")
p.write_text(t.replace(old, new))
print("0001 (#1184 stream drop): applied")
PY
fi

if grep -q "PATCHED (see patches/0002" lib/matchmaking.py; then
  echo "0002 (matchmaking blocking check): already applied"
else
  python3 - <<'PY2'
import pathlib
p = pathlib.Path("lib/matchmaking.py"); t = p.read_text()
old = """            bot_profile = self.li.get_public_data(bot["username"])
            if bot_profile.get("blocking"):"""
new = """            # PATCHED (see patches/0002-matchmaking-blocking-check.patch):
            # a 429 on this courtesy lookup must not abort the challenge.
            try:
                bot_profile = self.li.get_public_data(bot["username"])
            except Exception:
                bot_profile = {}
            if bot_profile.get("blocking"):"""
assert old in t, "0002: anchor not found"
p.write_text(t.replace(old, new))
PY2
  echo "0002 (matchmaking blocking check): applied"
fi
