#!/usr/bin/env python
"""Which held-out positions also appear in training, per bag.

    scripts/build_clean_indices.py --target state_value
    scripts/build_clean_indices.py --target behavioral_cloning --check

Zero GPU. One streaming pass over the training bag; heavy on disk, so run it on
an otherwise idle box, and never during a wall-clock match.

**Why this is a hard prerequisite for the capacity decision, not hygiene.**
`data/test/clean_bc_indices.json` records **16.19% train/test overlap** for the
policy bag, 10,067 of 62,178 positions, and it has ZERO references anywhere in
the repo: nothing reads it, so every held-out number this project has quoted is
computed over a set that is one sixth memorisable. There is no equivalent for
the state-value bag at all, and state-value held-out loss is what gated BOTH
value-net promotions.

That is tolerable while comparing two nets of the SAME size, because the
contamination flatters both equally. It stops being tolerable the moment a
capacity comparison happens: leaked rows are memorisable, memorisation scales
with parameters, so contamination flatters the LARGER model on exactly the
comparison the d=384 decision rests on. A fused d=384 trunk could win on
held-out loss by remembering rather than by judging, and nothing downstream
would notice.

**The method, and why it is exact rather than probabilistic.** The naive
direction is to build a set of every training position and test the held-out
ones against it: 530 million records, which as a Python set is tens of GB and
this box has livelocked on memory pressure before. Invert it. The held-out bag
is 62,178 records, so hash THOSE into a set (a few MB), then stream the
training bag once and mark which of them it hits. O(held-out) memory, one
sequential read, and an exact answer with no Bloom filter and no false
positives.

**What counts as the same position.** The FEN, normalised to its first four
fields: piece placement, side to move, castling rights, en passant square. The
halfmove clock and the fullmove number are dropped deliberately. A model cannot
tell two positions apart by move number and does not receive it as a
distinguishing feature in any useful sense; counting them as different would
under-report the overlap, which is the direction that flatters the result.
Recorded in the output so the choice is auditable rather than assumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sumofish import bagz  # noqa: E402

DECODE = {"behavioral_cloning": bagz.decode_behavioral_cloning,
          "state_value": bagz.decode_state_value}


def position_key(fen: str) -> bytes:
    """The first four FEN fields, hashed. See the module docstring."""
    return hashlib.blake2b(" ".join(fen.split()[:4]).encode(),
                           digest_size=16).digest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", default="state_value",
                    choices=sorted(DECODE))
    ap.add_argument("--train", default=None,
                    help="training bag (default data/train/<target>_data.bag)")
    ap.add_argument("--test", default=None,
                    help="held-out bag (default data/test/<target>_data.bag)")
    ap.add_argument("--out", default=None,
                    help="default data/test/clean_<tag>_indices.json")
    ap.add_argument("--check", action="store_true",
                    help="recompute and compare against the existing file "
                         "instead of writing; exits 1 on disagreement")
    ap.add_argument("--progress-every", type=int, default=20_000_000)
    a = ap.parse_args()

    tag = {"behavioral_cloning": "bc", "state_value": "sv"}[a.target]
    train = Path(a.train or ROOT / f"data/train/{a.target}_data.bag")
    test = Path(a.test or ROOT / f"data/test/{a.target}_data.bag")
    out = Path(a.out or ROOT / f"data/test/clean_{tag}_indices.json")
    for p in (train, test):
        if not p.exists():
            sys.exit(f"missing bag: {p}")
    decode = DECODE[a.target]

    # ---- 1. the held-out side, which is small -----------------------------
    t0 = time.perf_counter()
    reader = bagz.BagReader(str(test))
    n_test = len(reader)
    # One key can appear at several held-out indices; keep every index so the
    # clean set is exact rather than deduplicated behind your back.
    by_key: dict[bytes, list[int]] = {}
    for i in range(n_test):
        fen = decode(reader[i])[0]
        by_key.setdefault(position_key(fen), []).append(i)
    reader.close()
    print(f"held-out: {n_test:,} records, {len(by_key):,} distinct positions "
          f"({n_test - len(by_key):,} internal duplicates) in "
          f"{time.perf_counter() - t0:.1f}s")

    # ---- 2. one streaming pass over training ------------------------------
    hit: set[bytes] = set()
    reader = bagz.BagReader(str(train))
    n_train = len(reader)
    print(f"training: {n_train:,} records, {train.stat().st_size / 1e9:.1f} GB. "
          f"One pass, O(held-out) memory.")
    t0 = time.perf_counter()
    for j in range(n_train):
        k = position_key(decode(reader[j])[0])
        if k in by_key:
            hit.add(k)
        if a.progress_every and (j + 1) % a.progress_every == 0:
            el = time.perf_counter() - t0
            print(f"  {j + 1:>13,} / {n_train:,}  ({(j + 1) / n_train:5.1%})  "
                  f"{len(hit):,} of {len(by_key):,} held-out positions hit  "
                  f"{el / 60:.1f} min elapsed, "
                  f"~{el / (j + 1) * (n_train - j - 1) / 60:.1f} min left")
    reader.close()

    contaminated = sorted(i for k in hit for i in by_key[k])
    clean = sorted(set(range(n_test)) - set(contaminated))
    doc = {
        "target": a.target,
        "contaminated_positions": len(contaminated),
        "total_test_positions": n_test,
        "overlap_fraction": round(len(contaminated) / n_test, 5),
        "clean_record_indices": clean,
        # Provenance, because the 2026-07-28 BC file has none and nobody can
        # now say what "same position" meant when it was built.
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "train_bag": str(train), "train_records": n_train,
        "test_bag": str(test),
        "key": "blake2b-128 of the first four FEN fields "
               "(placement, side to move, castling, en passant); the halfmove "
               "clock and fullmove number are deliberately excluded",
    }

    print(f"\n  {len(contaminated):,} of {n_test:,} held-out records "
          f"({len(contaminated) / n_test:.2%}) also appear in training")
    print(f"  {len(clean):,} clean")

    if a.check:
        if not out.exists():
            print(f"  --check: {out} does not exist")
            return 1
        old = json.loads(out.read_text())
        same = (old.get("clean_record_indices") == clean)
        print(f"  --check: {'AGREES' if same else 'DISAGREES'} with {out.name} "
              f"(was {old.get('contaminated_positions')} contaminated, "
              f"now {len(contaminated)})")
        return 0 if same else 1

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc))
    print(f"  wrote {out}")
    print("\n  Nothing reads this yet for state_value. Wire it into "
          "scripts/eval_heldout.py\n  so every held-out number is reported "
          "clean-subset alongside full-set,\n  which is plan item 1.6's actual "
          "deliverable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
