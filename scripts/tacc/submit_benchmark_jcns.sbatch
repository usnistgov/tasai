#!/usr/bin/env bash
#SBATCH -J tasai_jcns
#SBATCH -A <YOUR_TACC_ALLOCATION>
#SBATCH -p skx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o <WORK_ROOT>/tasai/run_outputs/%x-%j.out

# Example:
#   METHOD=log_gp N_RUNS=10 MAX_MEASUREMENTS=200 sbatch submit_benchmark_jcns.sbatch

set -euo pipefail

#
# Site-specific notes:
# - Set WORK_ROOT to your own TACC work directory before submission.
# - The output path above and the script path below both assume the repo has
#   been copied to $WORK_ROOT/tasai/src.
#
WORK_ROOT=${WORK_ROOT:-<WORK_ROOT>}
SCRIPT="$WORK_ROOT/tasai/src/scripts/tacc/run_benchmark_jcns.sh"

exec "$SCRIPT"
