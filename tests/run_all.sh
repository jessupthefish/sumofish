#!/usr/bin/env bash
# Every oracle, in one command. Run this before trusting anything in rust/.
#
#   tests/run_all.sh          # fast: ~40s
#   tests/run_all.sh --deep   # adds 594M-node perft and a bigger differential
#
# Five oracles, deliberately independent -- each catches what the others cannot:
#
#   1. perft vs PUBLISHED counts     -- move sets, no python-chess involved
#   2. differential vs python-chess  -- move ORDER, which perft cannot see
#   3. push parity vs python-chess   -- every FEN field after every ply, which
#                                       perft only checks where it changes
#                                       legality (so it misses the halfmove
#                                       clock and the fullmove number entirely)
#   4. terminal parity vs rules.py   -- mate/stalemate/material/fifty-move and
#                                       threefold, with deliberate repetition
#                                       cycles because random play never
#                                       reaches one
#   5. identity vs chessgpu.mcts     -- a BYTE-IDENTICAL root visit vector, the
#                                       Stage C acceptance test
#   6. softmax                       -- what is and is not reproducible; layer 3
#                                       is a canary on a known blocker
#   7. tokenizer                     -- vs BOTH Python entry points
#   8. identity WITH REUSE           -- the same, across whole games, where the
#                                       retained subtree carries state forward
#   9. leaf dedup                    -- that it is IDENTITY-PRESERVING, plus how
#                                       much duplicated work it removes
#  10. mate distance                 -- proofs checked against an EXHAUSTIVE
#                                       solver, and the node reduction
#  11. virtual loss (vloss_fix)      -- Rust unit tests: no leaked virtual
#                                       loss, value_sum untouched by it, and
#                                       the flag is proven to change search
#                                       (not silently wired to a no-op)
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
DEEP=${1:-}

echo "=== building ==="
( cd rust && VIRTUAL_ENV="$PWD/../.venv" maturin develop --release 2>&1 | tail -1 )

echo
echo "=== oracle 1: perft vs published counts ==="
( cd rust && cargo test --release --test perft 2>&1 | tail -4 )
if [ "$DEEP" = "--deep" ]; then
    ( cd rust && cargo test --release --test perft -- --ignored --nocapture 2>&1 | tail -9 )
fi

echo
echo "=== oracle 1b: rust unit tests (softmax, tokenizer, vloss_fix) ==="
( cd rust && cargo test --release --lib 2>&1 | tail -20 )

echo
echo "=== oracle 2: differential vs python-chess, order-sensitive ==="
if [ "$DEEP" = "--deep" ]; then
    $PY tests/differential.py --positions 100000 | tail -3
else
    $PY tests/differential.py --positions 20000 | tail -3
fi

echo
echo "=== oracle 3: push parity, every FEN field every ply ==="
if [ "$DEEP" = "--deep" ]; then
    $PY tests/push_parity.py --games 5000 | tail -3
else
    $PY tests/push_parity.py --games 1000 | tail -3
fi

echo
echo "=== oracle 4: terminal parity vs chessgpu/rules.py ==="
if [ "$DEEP" = "--deep" ]; then
    $PY tests/terminal_parity.py --games 3000 | tail -5
else
    $PY tests/terminal_parity.py --games 400 | tail -5
fi

echo
echo "=== oracle 5: identity vs chessgpu.mcts (byte-identical visits) ==="
if [ "$DEEP" = "--deep" ]; then
    $PY tests/identity_search.py --positions 40 --sims 800 --batch 64 | tail -2
else
    $PY tests/identity_search.py --positions 10 --sims 400 --batch 64 | tail -2
fi

echo
echo "=== oracle 8: identity with tree reuse, across whole games ==="
if [ "$DEEP" = "--deep" ]; then
    $PY tests/identity_reuse.py --games 10 --plies 16 --sims 400 | tail -4
else
    $PY tests/identity_reuse.py --games 4 --plies 8 --sims 200 | tail -4
fi

echo
echo "=== oracle 9: leaf dedup is identity-preserving, and what it saves ==="
$PY tests/verify_dedup.py --positions 8 --sims 800 | tail -12

echo
echo "=== oracle 10: mate distance, proofs vs an exhaustive solver ==="
$PY tests/verify_mate.py --positions 60 --sims 400 | tail -12

echo
echo "=== oracle 6: what is reproducible about the prior softmax ==="
$PY tests/verify_softmax.py --positions 4000 | tail -8

echo
echo "=== oracle 7: tokenizer vs both Python entry points ==="
if [ "$DEEP" = "--deep" ]; then
    $PY tests/verify_tokenizer.py --positions 40000 | tail -5
else
    $PY tests/verify_tokenizer.py --positions 8000 | tail -5
fi

echo
echo "=== the action-space table is still in sync with the tokenizer ==="
$PY tests/verify_action_space.py | tail -2

echo
echo "=== think_time()'s floor never exceeds the actual remaining clock ==="
$PY tests/verify_think_time.py | tail -12

echo
echo "=== --auto-resume continues the data stream, doesn't replay it ==="
$PY tests/verify_data_frac.py | tail -12

echo
echo "=== Rust-only CHESSGPU_* flags warn instead of silently no-op on Python core ==="
$PY tests/verify_rust_flag_guard.py | tail -12

echo
echo "=== speed: move generation ==="
$PY tests/bench_movegen.py | tail -12

echo
echo "=== speed: the tree ==="
$PY tests/bench_tree.py | tail -8

echo
echo "=== speed: tokenization ==="
$PY tests/bench_tokenizer.py | tail -9
