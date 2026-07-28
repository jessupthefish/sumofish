#!/usr/bin/env bash
# Fetch N action-value shards. The full set is 2148 shards / 1.1TB; we take a
# prefix. Action-value is the paper's strongest prediction target: 83.3% puzzle
# accuracy vs 65.7% for behavioral cloning at identical architecture.
#
# Each shard is ~1.25GB / ~16M records. An 8-hour run on this box consumes
# ~290M positions, so 40 shards is roughly two epochs of headroom.
set -euo pipefail
N="${1:-40}"
BASE=https://storage.googleapis.com/searchless_chess/data
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/train"
mkdir -p "$DEST"
for i in $(seq -f "%05g" 0 $((N-1))); do
  f="action_value-$i-of-02148_data.bag"
  [ -f "$DEST/$f" ] && { echo "have $f"; continue; }
  echo ">>> $f ($((10#$i+1))/$N)"
  wget -c -q --show-progress --progress=dot:giga -P "$DEST" "$BASE/train/$f"
done
echo "done: $(ls "$DEST"/action_value-*.bag 2>/dev/null | wc -l) shards"
du -sh "$DEST"
