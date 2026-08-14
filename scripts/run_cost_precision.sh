#!/bin/bash
# Plan 1.1, precision pass. The 12s sweep is not good enough to pick a width.
#
# What the first pass established, and it is real: the model `today(d) = C +
# b*d^2` fits all eight widths at R^2 = 0.9935 with b = 2.690e-4, and the
# d=1024 point pins b hard because the effect there is 3x the baseline.
#
# What it could NOT establish is the decision. The 9M baseline, re-measured
# eight times, ranged 121.2 to 131.3 us/node, so the noise floor is +-4%. The
# choice is between d=320 and d=512, where the modelled differences are 5-25%,
# and at d=384 the measurement came back FASTER than d=256, which is
# physically impossible and is therefore noise. A number without an interval is
# not a result, and neither is one whose interval covers the decision.
#
# Three changes, all aimed at the interval rather than at more widths:
#   1. 25s per arm instead of 12, and three repetitions, so each width gets a
#      spread rather than a point.
#   2. INTERLEAVED: rep-major, not width-major. GPU clocks drift over a run,
#      and width-major ordering aliases that drift onto width, which is exactly
#      the confound that made the first pass unreadable.
#   3. FUSED arms, because the fused net is the configuration actually
#      proposed. The unfused sweep prices width; only the fused arm prices the
#      thing being decided.
#
# Control: unfused d256 against the 9M baseline is the same shape twice, so it
# must read 1.00x. Whatever it actually reads is this pass's noise floor, and
# no width difference smaller than that means anything.
set -uo pipefail
cd /home/nomad/dev/active/sumofish
PY=.venv/bin/python
OUT=runs/lab/cost-precision-2026-08-14
mkdir -p "$OUT"
S=${SECONDS_PER_ARM:-25}
REPS=${REPS:-3}

for rep in $(seq 1 "$REPS"); do
    echo
    echo "################ rep $rep of $REPS ################"

    echo "--- control: unfused d256 vs 9M (same shape twice, must read 1.00x) ---"
    timeout 900 $PY scripts/bench_search.py --preset d256 --baseline 9M \
        --batch 64 --seconds "$S" --out "$OUT/control-rep$rep.json" 2>&1 | tail -4

    for d in d256 d320 d384 d448 d512; do
        echo "--- fused $d, rep $rep ---"
        timeout 900 $PY scripts/bench_search.py --preset "$d" --baseline 9M --fused \
            --batch 64 --seconds "$S" --out "$OUT/fused-$d-rep$rep.json" 2>&1 | tail -4
    done
done

echo
echo "################ done: $OUT ################"
ls -1 "$OUT" | wc -l
