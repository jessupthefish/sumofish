#!/usr/bin/env python
"""Cut a version of SumoFish, and start its win/loss record from zero.

    scripts/release.py "search is 2.6x faster and it plays 15+10 now"
    scripts/release.py --list

A version is a claim: *this* is the bot now. The lifetime record cannot express
that. SumoFish is at 24% lifetime and 52% since yesterday, and the difference is
not variance, it is that the thing playing today is a different engine at a
different time control. Averaging them describes nothing that ever existed.

So every release records a boundary, and the record shown everywhere -- the
dashboard header, `sumofish-games` -- is the record *since that boundary*. The
games themselves are never touched: lichess-bot's PGNs are the archive, this
just says where to start counting. `--all` anywhere shows the whole history.

## What a version pins

Not just the code. A rating is produced by the code AND the checkpoint AND the
time control, and this project has changed all three in a day. Each entry
records the git SHA, the deployed checkpoint's hash and training step, and the
time control being accepted, so "v2 scored 52%" can be traced to exactly what
was playing.

## What it writes

  VERSIONS.jsonl           the registry, tracked in git
  a git tag                v1, v2, ... on the released commit
  an Obsidian note         10 Steven/13 Efforts/Software/SumoFish Releases/

The Obsidian note is written twice over in the same file: a technical section
listing what actually changed, and a plain-English one that does not assume you
remember what a value net is. Six months from now the second is the one worth
having, and it is the one nobody writes because it feels redundant on the day.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "VERSIONS.jsonl"
VAULT = Path.home() / "Documents" / "Uno" / "10 Steven" / "13 Efforts" / "Software" / "SumoFish Releases"


def git(*args: str, check: bool = True) -> str:
    """Run git and RAISE on failure.

    This used to be `check=False` with stderr discarded, so a tag collision, a
    detached HEAD or a missing git binary all reported success: `main()` prints
    "tagged vN" unconditionally. Worse, a failed `rev-parse` returned "", which
    renders an empty backtick pair in the note and then breaks the NEXT
    release's `commits_since()`, because it builds "<sha>..HEAD".
    """
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({r.returncode}): "
            f"{(r.stderr or r.stdout).strip()}")
    return r.stdout.strip()


def load() -> list[dict]:
    if not REGISTRY.exists():
        return []
    return [json.loads(line) for line in REGISTRY.read_text().splitlines()
            if line.strip()]


def checkpoint_facts() -> dict:
    """What is actually deployed, not what is in the repo.

    BOTH nets, because the engine is both. Recording only the value net was a
    real hole and it opened: `runs/policy.pt` was replaced on 2026-08-01 and
    nothing in the registry could see it, so v3's record went on accumulating
    across a player that had changed. A version is a claim that *this* is the bot
    now, and half of "this" is the prior that shapes every search.
    """
    facts: dict = {}
    for name, keys in (
        ("value", ("checkpoint_sha", "step", "width", "layers")),
        ("policy", ("policy_sha", "policy_step", None, None)),
    ):
        live = ROOT / "runs" / f"{name}.pt"
        if not live.exists():
            continue
        sha_key, step_key, width_key, layers_key = keys
        facts[sha_key] = hashlib.sha256(live.read_bytes()).hexdigest()[:12]
        try:
            import torch
            ck = torch.load(live, map_location="cpu", weights_only=False)
        except Exception:                                # noqa: BLE001
            continue
        facts[step_key] = ck.get("step")
        # Shape is recorded once. The two nets are the same architecture today
        # and a version that changes only one of their widths would be a
        # different kind of release than this file has ever cut; write the pair
        # out separately when that happens rather than guessing at it now.
        if width_key:
            facts[width_key] = ck.get("cfg", {}).get("embedding_dim")
            facts[layers_key] = ck.get("cfg", {}).get("num_layers")
    return facts


def time_control() -> str:
    try:
        import yaml
        cfg = yaml.safe_load((ROOT / "config" / "lichess-bot.yml").read_text())
        mm = cfg.get("matchmaking", {})
        base = (mm.get("challenge_initial_time") or [None])[0]
        inc = (mm.get("challenge_increment") or [None])[0]
        return f"{base // 60}+{inc}" if base is not None else "?"
    except (OSError, ImportError, ValueError, AttributeError, TypeError):
        return "?"


def ratings() -> dict:
    log = ROOT / "logs" / "rating.jsonl"
    if not log.exists():
        return {}
    rows = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    return rows[-1].get("ratings", {}) if rows else {}


def commits_since(previous: dict | None) -> list[str]:
    span = f"{previous['sha']}..HEAD" if previous else "HEAD"
    out = git("log", "--no-merges", "--pretty=format:%s", span)
    return [line for line in out.splitlines() if line.strip()]


def record_for(entry: dict) -> str:
    """W/D/L since this version started, read from the game archive."""
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from games import infer_me, load_games, outcome
    except ImportError:
        return ""
    games = load_games()
    if not games:
        return ""
    me = infer_me(games)
    start = datetime.fromtimestamp(entry["ts"], timezone.utc)
    mine = [g for g in games if g["when"] and g["when"] >= start]
    w = sum(1 for g in mine if outcome(g, me)[0] == "win")
    d = sum(1 for g in mine if outcome(g, me)[0] == "draw")
    lo = sum(1 for g in mine if outcome(g, me)[0] == "loss")
    n = w + d + lo
    return f"{w}W {d}D {lo}L over {n} games" + (
        f", scoring {(w + 0.5 * d) / n * 100:.0f}%" if n else "")


# ---------------------------------------------------------------------------
# the Obsidian note


def note_body(entry: dict, previous: dict | None, subjects: list[str]) -> str:
    when = datetime.fromtimestamp(entry["ts"]).strftime("%Y-%m-%d")
    prev_line = ""
    if previous:
        prev_line = (f"Previous version **{previous['version']}** finished on "
                     f"{record_for(previous) or 'no recorded games'}.\n\n")
    rated = entry.get("ratings", {})
    rating_line = "  \n".join(
        f"- {k} **{v['rating']}**" + (" (provisional)" if v.get("prov") else "")
        for k, v in rated.items() if v.get("games"))

    technical = "\n".join(f"- {s}" for s in subjects) or "- no commits recorded"

    # Vault schema, per 90 Meta/94 Docs/conventions.md:
    #   * `type` is a PROPERTY and never also a tag. `tags: [project, ...]` was
    #     a hard linter error and made the v6 note one of only two in the whole
    #     vault that tripped it, and the only one a script kept regenerating.
    #   * tags are block style, because Obsidian rewrites flow style on first
    #     touch and that guarantees a spurious diff on a synced file.
    #   * `status` comes from a closed set of six; a shipped release is `done`.
    #   * `created` is expected on every note.
    return f"""---
type: project
status: done
version: {entry['version']}
released: {when}
created: {when}
tags:
  - chess/engines
  - source/claude
---

# SumoFish {entry['version']}, {entry['title']}

Released {when}. Part of [[SumoFish]].

{prev_line}## In plain English

{entry.get('layman', '_not written_')}

## What changed

{technical}

## What was playing

- time control **{entry.get('time_control', '?')}**
- value net at step **{entry.get('step', '?')}**, {entry.get('width', '?')}-wide,
  {entry.get('layers', '?')} layers (`{entry.get('checkpoint_sha', '?')}`)
- policy net at step **{entry.get('policy_step', '?')}**
  (`{entry.get('policy_sha', '?')}`)
- commit `{entry['sha'][:12]}`

Rating at release:

{rating_line or '- none recorded'}

## Record

The record for this version starts at zero on release and is counted from the
game archive, not from lichess's lifetime totals. `sumofish-games` shows it;
`sumofish-games --all` shows every version.
"""


def obsidian_running() -> bool:
    """Is the app holding this vault in memory right now?

    Bracketed so the pattern cannot match this process's own command line,
    which is a documented trap on this machine. Also note `pgrep obsidian`
    is a FALSE NEGATIVE here: the Arch package runs as
    `electron /usr/lib/obsidian/app.asar`, so the process name never contains
    "obsidian".
    """
    r = subprocess.run(["pgrep", "-f", "obsidian/app[.]asar"],
                       capture_output=True, text=True)
    return r.returncode == 0


def write_note(entry: dict, previous: dict | None, subjects: list[str],
               force: bool = False) -> Path | None:
    if not VAULT.parent.exists():
        print(f"  vault not found at {VAULT.parent}; skipping the note")
        return None
    VAULT.mkdir(parents=True, exist_ok=True)
    safe = "".join(c for c in entry["title"] if c.isalnum() or c in " -_").strip()
    # Comma, not an em dash: matches v1-v5, and em dashes are out house-wide.
    path = VAULT / f"{entry['version']}, {safe}.md"

    # A release note is written ONCE. Re-running for an existing version used to
    # silently overwrite whatever was there, including anything added by hand.
    if path.exists() and not force:
        print(f"  note already exists, not overwriting: {path}")
        print("  (pass --force-note if you really mean to replace it)")
        return None

    body = note_body(entry, previous, subjects)

    # THE VAULT CONTRACT, rule 1. Obsidian keeps an in-memory copy of any open
    # file and flushes it over an external write when the tab loses focus, so a
    # plain write_text() into a running vault can vanish with no error. Go
    # through the CLI, which uses Obsidian's own Vault API.
    if obsidian_running():
        rel = path.relative_to(Path.home() / "Documents" / "Uno")
        cmd = ["obsidian", "create", f"path={rel}", f"content={body}"]
        if force:
            cmd.append("overwrite")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not path.exists():
            raise RuntimeError(
                "obsidian create failed and a raw write is not safe while the "
                f"app is running: {(r.stderr or r.stdout).strip()}")
        return path

    path.write_text(body)
    return path


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("title", nargs="?", help="one line: what this version is")
    ap.add_argument("--layman", default="",
                    help="the plain-English paragraph. Prompted for if omitted")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-tag", action="store_true")
    ap.add_argument("--force-note", action="store_true",
                    help="replace an existing vault note for this version")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be recorded, touch nothing")
    args = ap.parse_args()

    history = load()

    if args.list or not args.title:
        if not history:
            print("no versions yet. scripts/release.py \"what this version is\"")
            return
        for entry in history:
            when = datetime.fromtimestamp(entry["ts"]).strftime("%Y-%m-%d %H:%M")
            print(f"  {entry['version']:>4}  {when}  {entry['title']}")
            print(f"        {record_for(entry) or 'no games'}")
        return

    previous = history[-1] if history else None
    number = int(previous["version"].lstrip("v")) + 1 if previous else 1
    subjects = commits_since(previous)

    layman = args.layman
    if not layman:
        print("One paragraph, plain English, for someone who does not know what")
        print("a value net is. Blank line to finish.")
        lines = []
        try:
            while (line := input()) != "":
                lines.append(line)
        except EOFError:
            pass
        layman = " ".join(lines).strip()

    entry = {
        "version": f"v{number}",
        "title": args.title,
        "layman": layman,
        "ts": time.time(),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "commits": len(subjects),
        "time_control": time_control(),
        "ratings": ratings(),
        **checkpoint_facts(),
    }

    # PRE-FLIGHT, before anything is written. The old order appended to the
    # registry and created the tag first, so a note that could not be written
    # left a tagged, registered version behind with one line of output that
    # scrolls past. Check the conflicting cases while both are still undone.
    safe = "".join(c for c in entry["title"] if c.isalnum() or c in " -_").strip()
    planned = VAULT / f"{entry['version']}, {safe}.md"
    if planned.exists() and not args.force_note:
        print(f"a note for {entry['version']} already exists:\n  {planned}")
        print("nothing written. Pass --force-note to replace it.")
        return 1
    existing_tags = git("tag", "--list", entry["version"])
    if existing_tags and not args.no_tag:
        print(f"tag {entry['version']} already exists; nothing written.")
        return 1

    if args.dry_run:
        print(f"\n--dry-run, nothing written. Would record:\n")
        print(json.dumps(entry, indent=2))
        print(f"\n  tag:  {entry['version']}")
        print(f"  note: {planned}")
        return 0

    with REGISTRY.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    print(f"  {entry['version']} recorded in VERSIONS.jsonl")

    if not args.no_tag:
        git("tag", "-a", entry["version"], "-m", f"{entry['version']}: {args.title}")
        print(f"  tagged {entry['version']} (git push --tags to publish it)")

    note = write_note(entry, previous, subjects, force=args.force_note)
    if note:
        print(f"  wrote {note}")

    if previous:
        print(f"\n  {previous['version']} finished on {record_for(previous) or 'no games'}")
    print(f"  {entry['version']} starts at 0W 0D 0L, counting from now")


if __name__ == "__main__":
    main()
