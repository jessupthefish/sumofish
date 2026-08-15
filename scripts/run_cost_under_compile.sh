#!/bin/bash
# Re-price the fusion under COMPILE, which is the deployed config since
# 2026-08-15 and was not the deployed config when plan 1.1 was measured.
#
# Why this is not optional. The fusion prize is one whole forward pass's LAUNCH
# overhead: two sequential passes become one, and at d=256 the launch term was
# 61-68% of a pass. `compile` uses CUDA graphs, which cut exactly that term.
# So the 1.76x fused-d256 refund and the 1.64x at d=384, both measured with
# compile OFF, are now numbers about a configuration that no longer ships. The
# knee between d=384 and d=448 moves too, and it moves DOWN.
#
# Every number this produces supersedes its counterpart in
# runs/lab/cost-precision-2026-08-14/. Same method as that pass, because the
# comparison only means anything if the method is identical: interleaved
# rep-major so GPU clock drift cannot alias onto width, a same-shape control
# whose reading IS the noise floor, and enough seconds per arm to see past it.
set -uo pipefail
cd /home/nomad/dev/active/sumofish
PY=.venv/bin/python
OUT=runs/lab/cost-under-compile-2026-08-15
mkdir -p "$OUT"
S=${SECONDS_PER_ARM:-20}
REPS=${REPS:-3}

# One arm per process: --compile refuses anything else, and the bench explains
# why in its own refusal message. The arithmetic happens here, across
# processes, instead of inside one.
run() {   # run <label> <spec> <extra flags...>
    local label="$1"; shift
    local spec="$1"; shift
    timeout 1800 $PY scripts/bench_search.py --single "$spec" --compile \
        --batch 64 --seconds "$S" --out "$OUT/$label.json" "$@" 2>&1 | grep -E 'nps|wrote'
    if [ ! -s "$OUT/$label.json" ]; then
        echo "ABORT: $label wrote no result."
        exit 1
    fi
}

for rep in $(seq 1 "$REPS"); do
    echo
    echo "################ rep $rep of $REPS ################"
    # Baseline and control FIRST in each rep, so drift shows up in the control
    # rather than hiding in a width.
    run "baseline-rep$rep" 9M
    run "control-rep$rep"  d256
    for d in d256 d320 d384 d448 d512; do
        run "fused-$d-rep$rep" "$d" --fused
    done
done

echo
echo "################ done: $OUT ################"
ls -1 "$OUT" | wc -l
