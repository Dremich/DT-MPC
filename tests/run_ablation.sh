#!/usr/bin/env bash
# Ablation at the default setting (dt=0.05, horizon=100, noise_std=4.0):
# DT-MPC (theta adapted) vs NT-MPC (fixed theta) vs nominal-only.
# Workers run in parallel and dump trajectories to .npz; the merge step appends a
# quantitative table to experiment_results.txt and saves the plots.
set -u
cd /d/workspace/DT-MPC 2>/dev/null || cd "$(dirname "$0")/.."

export DTMPC_RUNS=10 DTMPC_ABL_STEPS=80
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3

rm -f ablation_part_*.npz abl_log_*.log

worker() {  # variant seeds id
    DTMPC_ABL="$1" DTMPC_SEEDS="$2" DTMPC_ABL_OUT="ablation_part_$3.npz" \
        python tests/run_experiments.py > "abl_log_$3.log" 2>&1 &
}

worker "dtmpc"   "0,1,2,3"            01
worker "dtmpc"   "4,5,6"              02
worker "dtmpc"   "7,8,9"              03
worker "ntmpc"   "0,1,2,3,4"          04
worker "ntmpc"   "5,6,7,8,9"          05
worker "nominal" "0,1,2,3,4,5,6,7,8,9" 06

echo "launched $(jobs -p | wc -l) ablation workers; waiting..."
wait
echo "merging + plotting ablation..."
DTMPC_ABL_MERGE="ablation_part_*.npz" python tests/run_experiments.py
echo "ABLATION DONE"
