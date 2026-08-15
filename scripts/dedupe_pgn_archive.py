#!/usr/bin/env python
"""One record per rated game, across every PGN this project has written.

    scripts/dedupe_pgn_archive.py --check      # report, write nothing
    scripts/dedupe_pgn_archive.py --apply

Plan item 0.8's other half. The reliability ledger it belongs to was rebuilt
from these files, and the files disagreed with themselves.

**What is wrong.** `logs/games/SumoFish games.pgn` is the master and it holds
duplicate records: the same lichess game appended more than once, up to five
times, and the most duplicated games are the two most argued-about ones in the
project's history, which is precisely how a ledger built by counting records
overstates. Meanwhile 35 of the 42 individually-saved PGNs under `logs/` are
ABSENT from the master, so counting the master alone understates. Both errors
at once, in opposite directions, in the file every forfeit-and-abandon count
has been derived from.

**The key is `Site`.** It is the lichess game URL and it is unique per game.
Not the date, not the players, not the moves: two games between the same bots
on the same day are different games, and the same game re-saved twice is not.

**Which copy wins when a Site appears more than once.** The most COMPLETE one,
scored in this order: a decisive `Result` beats `*`, then more moves, then more
tags, then longer text. An abandoned game saved before its result was known and
then re-saved after must resolve to the version that knows the result, or the
abandon ledger loses exactly the games it exists to count.

Non-destructive by default. `--apply` writes a timestamped backup first and
then verifies that every Site present before is present after, refusing to
leave the archive smaller in games than it found it.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "logs/games/SumoFish games.pgn"
SITE = re.compile(r'^\[Site "([^"]+)"\]', re.M)
TAG = re.compile(r'^\[[A-Za-z0-9_]+ "', re.M)
RESULT = re.compile(r'^\[Result "([^"]+)"\]', re.M)


def split_records(text: str) -> list[str]:
    """PGN records, split on the tag pair that starts each one.

    A record begins at `[Event `, which is the first tag of every PGN export.
    Splitting on blank lines instead would break every game in two, because a
    PGN has one between its tag pair and its movetext.
    """
    parts = re.split(r'(?m)^(?=\[Event ")', text)
    return [p for p in parts if p.strip()]


def completeness(rec: str) -> tuple[int, int, int, int]:
    """Sort key: decisive result, then moves, then tags, then length."""
    res = RESULT.search(rec)
    decisive = 1 if (res and res.group(1) != "*") else 0
    moves = len(re.findall(r"\b\d+\.", rec))
    return (decisive, moves, len(TAG.findall(rec)), len(rec))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not a.apply:
        a.check = True

    sources = sorted(p for p in (ROOT / "logs").rglob("*.pgn"))
    if MASTER not in sources:
        sys.exit(f"no master at {MASTER}")
    print(f"{len(sources)} pgn file(s) under logs/\n")

    best: dict[str, str] = {}
    seen_in_master: list[str] = []
    per_file: dict[Path, int] = {}
    for p in sources:
        recs = split_records(p.read_text(errors="replace"))
        per_file[p] = len(recs)
        for r in recs:
            m = SITE.search(r)
            if not m:
                continue                       # no Site: cannot be keyed, skipped
            site = m.group(1)
            if p == MASTER:
                seen_in_master.append(site)
            if site not in best or completeness(r) > completeness(best[site]):
                best[site] = r

    dupes = Counter(seen_in_master)
    n_dupe = sum(v - 1 for v in dupes.values() if v > 1)
    missing = sorted(set(best) - set(seen_in_master))
    print(f"master:  {len(seen_in_master)} records, {len(set(seen_in_master))} unique, "
          f"{n_dupe} duplicated")
    for site, n in dupes.most_common(5):
        if n > 1:
            print(f"           {site}  x{n}")
    print(f"elsewhere: {len(missing)} game(s) present in other files and ABSENT "
          f"from the master")
    print(f"\nunion: {len(best)} distinct games")

    if a.check and not a.apply:
        print("\n--check: nothing written. Re-run with --apply.")
        return 0

    # Ordered by date then Site so the file is stable across runs: a rebuild
    # that reshuffles records makes every future diff useless.
    def order(site: str) -> tuple[str, str]:
        d = re.search(r'^\[UTCDate "([^"]+)"\]', best[site], re.M)
        t = re.search(r'^\[UTCTime "([^"]+)"\]', best[site], re.M)
        return (f"{d.group(1) if d else '9999.99.99'} "
                f"{t.group(1) if t else '99:99:99'}", site)

    out = "\n\n".join(best[s].strip() for s in sorted(best, key=order)) + "\n"

    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = MASTER.with_suffix(f".pgn.bak-{stamp}")
    shutil.copy2(MASTER, backup)
    MASTER.write_text(out)

    # Verify, and put it back if anything was lost. A dedupe that drops a game
    # is strictly worse than the duplicates it removed.
    after = SITE.findall(MASTER.read_text(errors="replace"))
    lost = set(best) - set(after)
    if lost or len(set(after)) != len(best):
        shutil.copy2(backup, MASTER)
        sys.exit(f"REVERTED: {len(lost)} game(s) would have been lost. "
                 f"Backup kept at {backup.name}")
    print(f"\nwrote {MASTER.name}: {len(after)} records, {len(set(after))} unique")
    print(f"  backup {backup.name}")
    print(f"  net: {len(seen_in_master)} -> {len(after)} records, "
          f"{n_dupe} duplicate(s) removed, {len(missing)} missing game(s) added")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
