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
