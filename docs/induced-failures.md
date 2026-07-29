# Induced failures, recorded

Per the council's condition: a supervisor is not installed until it has been
*observed refusing*. An untested supervisor is not deployed, it is scheduled --
and this project already carries two watchdogs that supervise nothing.

## `scripts/match.py` fingerprinted resume — 2026-07-29

Three refusals induced deliberately, all before any checkpoint is loaded, so they
are testable without a GPU:

| induced condition | expected | observed |
|---|---|---|
| existing games, different spec fingerprint | refuse, exit 2 | refused, exit 2 |
| existing games, no fingerprint at all | refuse, exit 2 | refused, exit 2 |
| fresh `--name` | proceed, write fingerprint | proceeded, wrote `6aab50fb...` |

**It caught a real bug in itself.** The first version returned 2 from `main()`,
but the script ended in `main()` rather than `sys.exit(main())`, so the return
value was discarded and a refusal exited **0** — which `lab.py` gates on as
success. A match that refused to run would have been marked completed. Found only
by inducing the refusal and checking `$?`; no amount of reading would have shown
it.

A second bug came from the same exercise: the check originally required
`config.json` to exist, so a directory with games and no config — precisely the
unprovenanced state — was waved through. Now keyed on the games.

## `scripts/verify_replays.py` — 2026-07-29

Run against the real archive on first use: **0 of 8 matches trusted, 4 REPLAYED,
4 unprovenanced**, with impossibility factors of 490x, 783x, 1902x and 19x. It
found the damage on the first run rather than sitting green, which is the only
evidence that a detector detects.

`--check` returns 1 while anything is untrustworthy. Verified.
