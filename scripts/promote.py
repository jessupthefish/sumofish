#!/usr/bin/env python
"""Promote a checkpoint to the live bot.

    scripts/promote.py runs/9M-causal/best.pt
    scripts/promote.py runs/9M-causal/best.pt --min-accuracy 0.30
    scripts/promote.py --status

Copies the checkpoint to runs/current.pt, points lichess-bot's config at the
neural engine, and restarts the unit. Refuses to promote a checkpoint that
scores worse on puzzles than whatever is currently live, so an automated caller
cannot make the bot worse.

The copy is deliberate rather than a symlink: training keeps writing best.pt,
and a symlink would mean the engine's weights change under it mid-game.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CURRENT = ROOT / "runs" / "current.pt"
MANIFEST = ROOT / "runs" / "current.json"
CONFIG = ROOT / "config" / "lichess-bot.yml"
UNIT = "chess-gpu-bot.service"
EVAL_PUZZLES = 1000


def puzzle_accuracy(ckpt: Path, limit: int = EVAL_PUZZLES) -> float:
    from chessgpu.engines.neural_engine import load_policy
    from chessgpu.evaluate import evaluate_puzzles, load_puzzles

    policy, info = load_policy(ckpt)
    result = evaluate_puzzles(policy, load_puzzles(ROOT / "data/puzzles.csv", limit=limit))
    print(f"  {ckpt.name}: step {info['step']:,}  puzzles {result}")
    return result.accuracy


def set_engine(name: str) -> None:
    text = CONFIG.read_text()
    out, changed = [], False
    for line in text.splitlines():
        if line.strip().startswith("name:") and not changed:
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f'{indent}name: "{name}"')
            changed = True
        else:
            out.append(line)
    if not changed:
        sys.exit("could not find the engine name field in config")
    CONFIG.write_text("\n".join(out) + "\n")


def status() -> None:
    if not MANIFEST.exists():
        print("nothing promoted yet; bot is running the random-move engine")
        return
    print(json.dumps(json.loads(MANIFEST.read_text()), indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--min-accuracy", type=float, default=0.0,
                    help="refuse to promote below this puzzle accuracy")
    ap.add_argument("--force", action="store_true",
                    help="skip the comparison against what is currently live")
    ap.add_argument("--no-restart", action="store_true")
    args = ap.parse_args()

    if args.status:
        status()
        return
    if not args.checkpoint:
        sys.exit("give a checkpoint path, or --status")

    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.exists():
        sys.exit(f"no such checkpoint: {ckpt}")

    print("evaluating candidate")
    candidate = puzzle_accuracy(ckpt)

    if candidate < args.min_accuracy:
        sys.exit(f"refusing: {candidate:.4f} is below --min-accuracy {args.min_accuracy}")

    if CURRENT.exists() and not args.force:
        print("evaluating what is currently live")
        live = puzzle_accuracy(CURRENT)
        if candidate <= live:
            print(f"refusing: candidate {candidate:.4f} does not beat live {live:.4f}")
            print("pass --force to promote anyway")
            return

    CURRENT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt, CURRENT)
    MANIFEST.write_text(json.dumps({
        "source": str(ckpt),
        "puzzle_accuracy": round(candidate, 4),
        "puzzles_evaluated": EVAL_PUZZLES,
    }, indent=2))
    print(f"copied -> {CURRENT}")

    set_engine("sumofish")
    print("config points at bin/sumofish")

    if not args.no_restart:
        subprocess.run(["systemctl", "--user", "restart", UNIT], check=False)
        print(f"restarted {UNIT}")


if __name__ == "__main__":
    main()
