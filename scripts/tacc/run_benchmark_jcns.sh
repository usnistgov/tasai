#!/usr/bin/env bash
set -euo pipefail

#
# Site-specific notes:
# - Set WORK_ROOT to your own TACC work directory before running this helper.
# - The defaults below assume the repository has been copied to
#   $WORK_ROOT/tasai/src and benchmark outputs should be written to
#   $WORK_ROOT/tasai/run_outputs.
#
WORK_ROOT=${WORK_ROOT:-<WORK_ROOT>}
SOURCE_DIR="${SOURCE_DIR:-$WORK_ROOT/tasai/src}"
OUT_DIR="${OUT_DIR:-$WORK_ROOT/tasai/run_outputs}"

METHOD=${METHOD:-log_gp}
SCENARIO=${SCENARIO:-}
N_RUNS=${N_RUNS:-5}
MAX_MEASUREMENTS=${MAX_MEASUREMENTS:-120}
ERROR_THRESHOLD=${ERROR_THRESHOLD:-0.2}
TASAI_BACKEND=${TASAI_BACKEND:-pyspinw}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}
SEED=${SEED:-}
JOB_ID=${SLURM_JOB_ID:-local}
TASK_ID=${SLURM_ARRAY_TASK_ID:-}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-}
CHECKPOINT_PREFIX=${CHECKPOINT_PREFIX:-jcns_checkpoint_${METHOD}_job${JOB_ID}}
COOPER_NATHANS=${COOPER_NATHANS:-0}
HCOL=${HCOL:-}
VCOL=${VCOL:-}
EFIXED=${EFIXED:-}
GAUSSIAN_FWHM=${GAUSSIAN_FWHM:-}
RESCALC_PATH=${RESCALC_PATH:-}

source "$WORK_ROOT/conda/etc/profile.d/conda.sh"
export CONDA_ENVS_PATH="$WORK_ROOT/conda_envs"
conda activate tasai

if [[ -n "$RESCALC_PATH" ]]; then
  export PYTHONPATH="$RESCALC_PATH:${PYTHONPATH:-}"
fi

mkdir -p "$OUT_DIR"

seed_suffix=""
seed_arg=()
if [[ -n "$SEED" ]]; then
  N_RUNS=1
  seed_suffix="_seed${SEED}"
  seed_arg=(--seed "$SEED")
fi

job_suffix="_job${JOB_ID}"
if [[ -n "$TASK_ID" ]]; then
  job_suffix="${job_suffix}_task${TASK_ID}"
fi

SUMMARY_JSON="$OUT_DIR/jcns_${METHOD}_${RUN_TAG}${job_suffix}${seed_suffix}.json"

ARGS=(
  --method "$METHOD"
  --n-runs "$N_RUNS"
  --max-measurements "$MAX_MEASUREMENTS"
  --error-threshold "$ERROR_THRESHOLD"
  --tasai-backend "$TASAI_BACKEND"
  --summary-json "$SUMMARY_JSON"
  --no-plot
  "${seed_arg[@]}"
)

if [[ "$COOPER_NATHANS" == "1" ]]; then
  ARGS+=(--cooper-nathans)
  if [[ -n "$HCOL" ]]; then
    ARGS+=(--hcol $HCOL)
  fi
  if [[ -n "$VCOL" ]]; then
    ARGS+=(--vcol $VCOL)
  fi
  if [[ -n "$EFIXED" ]]; then
    ARGS+=(--efixed "$EFIXED")
  fi
  if [[ -n "$GAUSSIAN_FWHM" ]]; then
    ARGS+=(--gaussian-fwhm "$GAUSSIAN_FWHM")
  fi
fi

if [[ -n "$CHECKPOINT_DIR" ]]; then
  checkpoint_prefix="$CHECKPOINT_PREFIX"
  if [[ -n "$TASK_ID" ]]; then
    checkpoint_prefix="${checkpoint_prefix}_task${TASK_ID}"
  fi
  ARGS+=(--checkpoint-dir "$CHECKPOINT_DIR" --checkpoint-prefix "$checkpoint_prefix")
fi

if [[ -n "$SCENARIO" ]]; then
  ARGS+=(--scenario "$SCENARIO")
fi

python -m tasai.examples.benchmark_jcns "${ARGS[@]}"
echo "Wrote $SUMMARY_JSON"
