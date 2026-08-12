#!/usr/bin/env python3
"""A partial tuning run must not erase the groups it did not measure.

`runs/lab/tune-search.json` has been destroyed twice by the same shape of bug,
and rebuilt from the per-arm directories both times:

  2026-08-11, morning -- `--report --stage2-only` ended by writing the file
    like a real run, replacing five completed arms with five FAILED rows. Fixed
    by making `--report` return before the write.
  2026-08-11, afternoon -- the REAL `--stage2-only` run then did it again
    through the write path that was left alone: it replaced the 08-09 c_puct
    sweep with `{"skipped": ...}` and dropped the FPU line measured at
    c_puct_init=1.25 outright.

Reconstructible-by-luck is not a property to rely on a third time. The rule
this guards: a run may add or replace the groups it actually measured, and a
SKIPPED stage never overwrites arms that were really played.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import tune_search  # noqa: E402

FAIL = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global FAIL
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
    if not ok:
        FAIL += 1


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "tune-search.json"
        tune_search.OUT = out

        # nothing on disk: the fresh record passes through untouched
        fresh = {"seed": 1, "stage1": {"0.875": {"elo": 58.3}}}
        check("no prior file -> fresh record returned",
              tune_search._merge(fresh) == fresh)

        out.write_text(json.dumps({
            "seed": 1,
            "stage1": {"0.875": {"elo": 58.3}, "1.25": {"elo": 42.3}},
            "stage2_at_ci1.25": {"-0.05": {"elo": 55.6}},
        }, indent=2))

        # the exact 2026-08-11 afternoon run: stage 1 skipped, stage 2 at a new
        # c_puct_init. Neither prior group may move.
        merged = tune_search._merge({
            "seed": 2,
            "stage1": {"skipped": "--stage2-only"},
            "stage2_at_ci0.875": {"-0.05": {"elo": 44.1}},
        })
        check("a skipped stage does not overwrite arms that were played",
              merged["stage1"] == {"0.875": {"elo": 58.3}, "1.25": {"elo": 42.3}},
              json.dumps(merged["stage1"]))
        check("a group this run never touched survives",
              merged["stage2_at_ci1.25"] == {"-0.05": {"elo": 55.6}})
        check("the group this run measured is present",
              merged["stage2_at_ci0.875"] == {"-0.05": {"elo": 44.1}})
        check("a scalar this run measured is updated", merged["seed"] == 2)

        # a REAL stage 1 does replace a prior one: this guards data loss, not
        # re-measurement.
        merged = tune_search._merge({"stage1": {"0.875": {"elo": 61.0}}})
        check("a real stage 1 still replaces a prior stage 1",
              merged["stage1"] == {"0.875": {"elo": 61.0}})

        # and the bare "stage2" key is gone: two runs at different c_puct_init
        # cannot share a slot.
        src = (ROOT / "scripts" / "tune_search.py").read_text()
        check('stage 2 is keyed by the c_puct_init it was measured at',
              'setdefault(f"stage2_at_ci{best_c}"' in src
              and 'setdefault("stage2"' not in src)

    print(f"\n{'PASS' if not FAIL else str(FAIL) + ' FAILED'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
