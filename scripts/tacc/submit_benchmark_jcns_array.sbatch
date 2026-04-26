#!/usr/bin/env bash
#SBATCH -J tasai_jcns_arr
#SBATCH -A <YOUR_TACC_ALLOCATION>
#SBATCH -p skx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o <WORK_ROOT>/tasai/run_outputs/%x-%j.out
#SBATCH --array=0-9

# Example (10 seeds, log_gp):
#   METHOD=log_gp SEED_START=0 MAX_MEASUREMENTS=200 sbatch submit_benchmark_jcns_array.sbatch
#
# Example (10 seeds, tasai):
#   METHOD=tasai SEED_START=0 MAX_MEASUREMENTS=200 sbatch submit_benchmark_jcns_array.sbatch

set -euo pipefail

#
# Site-specific notes:
# - Set WORK_ROOT to your own TACC work directory before submission.
# - The output path above and the script path below both assume the repo has
#   been copied to $WORK_ROOT/tasai/src.
#
WORK_ROOT=${WORK_ROOT:-<WORK_ROOT>}
SCRIPT="$WORK_ROOT/tasai/src/scripts/tacc/run_benchmark_jcns.sh"

SEED_START=${SEED_START:-0}
export SEED=$((SEED_START + SLURM_ARRAY_TASK_ID))

exec "$SCRIPT"
