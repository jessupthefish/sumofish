#!/usr/bin/env python
"""Desktop alarm for when a training run stops, however it stops.

    scripts/train_done_notify.py --unit sumofish-train-fused.service \
                                 --run 19M-fused --steps 1200000

Run from a timer. Fires ONCE: it writes a marker beside the run and declines
afterwards, so a five-minute timer does not toast every five minutes for the
rest of the week.

**It notifies on failure too, and that is the point.** A watcher that only
knows the happy path is silent through a crash, an OOM and a wedged kernel
alike, and silence is indistinguishable from "still training". The unit going
inactive is the trigger; whether it reached the step count is a detail of the
message.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def unit_state(unit: str) -> tuple[str, str]:
    out = subprocess.run(
        ["systemctl", "--user", "show", "-p", "ActiveState", "-p", "Result",
         "--value", unit], capture_output=True, text=True).stdout.split()
    return (out + ["unknown", "unknown"])[0], (out + ["unknown", "unknown"])[1]


def last(run: str) -> dict:
    log = ROOT / "runs" / run / "log.jsonl"
    step, val, pol = 0, None, None
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            step = max(step, r.get("step", 0))
            if "val_loss" in r:
                val, pol = r["val_loss"], r.get("val_loss_policy")
    return {"step": step, "val": val, "policy": pol}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, required=True)
    a = ap.parse_args()

    marker = ROOT / "runs" / a.run / ".notified"
    if marker.exists():
        return 0
    state, result = unit_state(a.unit)
    if state in ("active", "activating", "reloading"):
        return 0
    if state == "inactive" and not (ROOT / "runs" / a.run).exists():
        return 0                      # never started; nothing to report

    r = last(a.run)
    done = r["step"] >= a.steps
    icon = "dialog-information" if done and result == "success" else "dialog-error"
    title = (f"SumoFish {a.run}: training finished"
             if done else f"SumoFish {a.run}: training STOPPED early")
    body = (f"step {r['step']:,} of {a.steps:,}  ({result})\n"
            f"held-out  value {r['val']}  policy {r['policy']}\n"
            f"targets   value 2.0674  policy 1.59138")
    subprocess.run(["notify-send", "-u", "critical", "-i", icon, "-t", "0",
                    title, body], check=False)
    marker.write_text(json.dumps({"state": state, "result": result, **r}))
    print(f"{title} | {body}")
    subprocess.run(["systemctl", "--user", "stop",
                    "sumofish-train-done.timer"], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
