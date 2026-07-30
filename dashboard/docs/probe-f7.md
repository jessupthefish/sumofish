## F7 — nvml and zbus instead of subprocesses

### nvml-wrapper

- driver: `610.43.03`
- NVML: `13.610.43.03`
- devices: 1

| field | value | today |
|---|---|---|
| name | NVIDIA GeForce RTX 5070 Ti | polled |
| gpu util % | 99 | polled |
| mem util % | 68 | NEW |
| memory | 12766 / 16303 MiB | polled |
| temp C | 77 | polled |
| power W | 280.9 | polled |
| power limit W | 300.0 | NEW |
| fan % | 86 | NEW |
| SM clock MHz | 2752 | NEW |
| mem clock MHz | 13801 | NEW |
| pstate | One | NEW |
| throttle reasons | ThrottleReasons(SW_POWER_CAP) | NEW |

**Per-process VRAM** (the R9 question):

| pid | used VRAM | cmdline |
|---|---|---|
| 591098 | 499 MiB | `/usr/bin/kwin_wayland --wayland-fd 7 --socket wayland-0 --xw…` |
| 1194446 | 107 MiB | `/usr/lib/claude-desktop/claude-desktop --type=gpu-process --…` |
| 1345757 | 11160 MiB | `/home/nomad/dev/active/chess-gpu/.venv/bin/python /home/nomad/dev/active/chess-gpu…` |

verdict: **PASS — per-process VRAM is readable; `machine` panel can answer the starving question directly**

### zbus + zbus_systemd on the USER bus

- `XDG_RUNTIME_DIR=/run/user/1000`
- `DBUS_SESSION_BUS_ADDRESS=(set)`

`ListUnitsByNames` returned 6 entries in one round trip.

| unit | load | active | sub | description |
|---|---|---|---|---|
| `chess-gpu-bot.service` | loaded | **active** | running | SumoFish lichess bot |
| `chess-gpu-lab.service` | loaded | **active** | running | SumoFish lab (autonomous experiment queue) |
| `chess-gpu-train.service` | not-found | **inactive** | dead | chess-gpu-train.service |
| `chess-gpu-watchdog.timer` | loaded | **active** | waiting | Run the chess-gpu bot watchdog every minute |
| `chess-gpu-train-watchdog.timer` | loaded | **active** | waiting | Check every 5 minutes that training is still progressing |
| `chess-gpu-rating.timer` | loaded | **active** | waiting | Sample the lichess rating every 15 minutes |

**Service properties** (note NRestarts vs the watchdog's own count):

| unit | MainPID | NRestarts | Result | started |
|---|---|---|---|---|
| `chess-gpu-bot.service` | 1424492 | 0 | success | 1785360623954099 |
| `chess-gpu-lab.service` | 1339507 | 0 | success | 1785346263166597 |
| `chess-gpu-train.service` | 0 | 0 | success | 1785269546958640 |

**Timers.** `NextElapseUSecMonotonic` is CLOCK_MONOTONIC microseconds, not wall time;
mixing it with `SystemTime::now()` is the classic bug. `ExecMainStartTimestamp` is
CLOCK_REALTIME microseconds. Different conversions.

| timer | next elapse (monotonic us) | last trigger |
|---|---|---|
| `chess-gpu-watchdog.timer` | 1181145983756 | 1785364171060789 |
| `chess-gpu-train-watchdog.timer` | 1181364839209 | 1785364149916305 |
| `chess-gpu-rating.timer` | 1181264344248 | 1785363449421027 |

