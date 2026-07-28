#!/usr/bin/env bash
# Selectively fetch ChessBench (Ruoss et al. 2024, arXiv:2402.04494).
#
# The upstream data/download.sh pulls all 2148 action-value shards, which is
# 1.1TB. We don't need that. This grabs the two complete training sets plus
# every evaluation file, about 70GB:
#
#   train/behavioral_cloning  33.8 GB  predict Stockfish's chosen move
#   train/state_value         36.0 GB  predict a position's win probability
#   test/*                     150 MB  held-out eval
#   puzzles.csv                4.5 MB  the 88.9% accuracy benchmark
#
# Behavioral cloning is the Phase 2 target. State-value is what turns the model
# into an evaluation function, which is what MCTS needs in Phase 4, so it is
# worth pulling in the same sitting rather than waiting on a second download.
#
# Resumable: wget -c, safe to re-run and safe to interrupt.
#
# To add action-value shards later (1.25 GB each, best results in the paper):
#   for i in $(seq -f "%05g" 0 99); do
#     wget -c "$BASE/train/action_value-$i-of-02148_data.bag" -P data/train
#   done

set -euo pipefail

BASE=https://storage.googleapis.com/searchless_chess/data
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data"

mkdir -p "$DEST/train" "$DEST/test"

get() {  # get <relative-path> <destination-dir>
  local url="$BASE/$1" dir="$2"
  echo ">>> $1"
  wget -c -q --show-progress --progress=dot:giga -P "$dir" "$url"
}

echo "=== small files first, so the tokenizer work can start immediately ==="
get puzzles.csv                          "$DEST"
get eco_openings.pgn                     "$DEST"
get test/behavioral_cloning_data.bag     "$DEST/test"
get test/state_value_data.bag            "$DEST/test"
get test/action_value_data.bag           "$DEST/test"

echo "=== training sets (~70GB, this is the long part) ==="
get train/behavioral_cloning_data.bag    "$DEST/train"
get train/state_value_data.bag           "$DEST/train"

echo
echo "=== done ==="
du -sh "$DEST"/* | sort -h
