# Deploying and rolling back

> Written 2026-08-14, because the only rollback instruction for the live policy net
> was a `cp` command inside `runs/policy.pt.json`, and `.gitignore` excluded
> `runs/`. A fresh clone of this repository could not have recovered the deployed
> engine. That is now fixed in two places: `runs/*.pt.json` is un-ignored, and
> `scripts/promote.py` handles both nets.

## The deployed engine is a PAIR of nets

`sumofish/rust_mcts.py` runs both on every evaluation call, on the same tokenised
tensor:

| file | role | promoted by |
|---|---|---|
| `runs/value.pt` | scores positions, backed up as Q by MCTS | `scripts/promote.py` |
| `runs/policy.pt` | supplies the prior for every search | `scripts/promote.py --net policy` |

Each carries a sidecar (`runs/value.pt.json`, `runs/policy.pt.json`) recording where
it came from and why, and a one-deep rollback (`*.pt.previous`).

## Promote

```sh
scripts/promote.py runs/9M-sv-long/best.pt --note "why"
scripts/promote.py --net policy runs/9M-bc-2026-08-01/best.pt --note "why"
```

Both stage beside the target and rename atomically, because lichess-bot spawns a
fresh engine per game and a game starting mid-copy would read a torn file. **No
restart is needed and none should be done**: the next game picks the file up, and
restarting costs whatever game is in progress.

Both gate on `scripts/smoke.py`, which loads the candidate into the named slot and
fills the other slot from the live file. A policy candidate is therefore exercised
by a real search with the deployed value net beside it, which is the only
arrangement in which its priors do anything.

## Roll back

```sh
scripts/promote.py --rollback                 # value net
scripts/promote.py --net policy --rollback    # policy net
```

## Check what is live

```sh
scripts/promote.py --status                   # both nets, with provenance
systemctl --user show sumofish-bot.service -p Environment
```

**The running unit is the only authority on flags.** Not `systemd/sumofish-bot.service`
on disk, not the copy in `~/.config/systemd/user`, and not any claim in `STATE.md`.
After any `systemctl --user edit` the file and the process disagree until a reload.

## What a promotion does NOT do

- It does not cut a version. Use `scripts/release.py` deliberately.
- It does not measure anything. `"elo": null` in a manual sidecar is honest and
  should stay null unless a match produced a number.
- It does not re-tune the search. The v6 constants (`c_puct_init=0.875`,
  `fpu=-0.05`, worth +63.5 Elo) were settled against the *current* policy prior.
  `runs/policy.pt.json` records the rule in its own caveat field: change the prior
  and those constants must be re-settled against the new one before any strength
  claim is made.
