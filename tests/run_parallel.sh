#!/usr/bin/env bash
# Run Dubins Tables 3/4/5 across parallel worker processes, then merge.
# Slowest rows (T3 dt=0.025, T4 horizon=100) are split by seed across 2 workers.
set -u
cd /d/workspace/DT-MPC 2>/dev/null || cd "$(dirname "$0")/.."

export DTMPC_RUNS=10 DTMPC_DUB_STEPS=70
# Keep each JAX process modest so 10 workers share 32 cores without oversubscribing.
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3

rm -f experiment_part_*.txt exp_log_*.log
ALL="0,1,2,3,4,5,6,7,8,9"

worker() {  # rows seeds id
    DTMPC_ROWS="$1" DTMPC_SEEDS="$2" DTMPC_OUT="experiment_part_$3.txt" \
        python tests/run_experiments.py > "exp_log_$3.log" 2>&1 &
}

# Slow rows (dt=0.025) and the high-noise rows (4.0, 10.0 — now run the full step
# cap since we no longer break on violation) are split by seed across 2 workers.
worker "3:0.025"     "0,1,2,3,4"  01
worker "3:0.025"     "5,6,7,8,9"  02
worker "5:10.0"      "0,1,2,3,4"  03
worker "5:10.0"      "5,6,7,8,9"  04
worker "5:4.0"       "0,1,2,3,4"  05
worker "5:4.0"       "5,6,7,8,9"  06
worker "4:100"       "$ALL"       07
worker "3:0.05,5:1.0" "$ALL"      08
worker "4:50,5:0.25"  "$ALL"      09
worker "4:25,3:0.1"   "$ALL"      10

echo "launched $(jobs -p | wc -l) workers; waiting..."
wait
echo "workers done; merging..."
DTMPC_MERGE="experiment_part_*.txt" DTMPC_OUT=experiment_results.txt \
    python tests/run_experiments.py
echo "ALL DONE"
