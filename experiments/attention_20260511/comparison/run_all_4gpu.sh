#!/usr/bin/env bash
set -euo pipefail

COMPARISON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${COMPARISON_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${EXPERIMENT_ROOT}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FOLDS="${FOLDS:-0 1 2 3}"
CUDA_FOLD_GPUS="${CUDA_FOLD_GPUS:-0,1,2,3}"

cd "${PROJECT_ROOT}"
mkdir -p "${COMPARISON_DIR}/logs" "${COMPARISON_DIR}/results"

write_status() {
  local status="$1"
  {
    echo "# Attention Comparison Status"
    echo
    echo "- Status: ${status}"
    echo "- Project root: ${PROJECT_ROOT}"
    echo "- Experiment root: ${EXPERIMENT_ROOT}"
    echo "- Comparison dir: ${COMPARISON_DIR}"
    echo "- Folds: ${FOLDS}"
    echo "- CUDA fold GPUs: ${CUDA_FOLD_GPUS}"
    echo "- Updated: $(date -Is)"
    echo "- Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  } > "${COMPARISON_DIR}/RUNNING_STATUS.md"
}

export PROJECT_ROOT
export PYTHON_BIN
export FOLDS
export CUDA_FOLD_GPUS

write_status "checking_fairness"
"${PYTHON_BIN}" "${COMPARISON_DIR}/check_fairness.py" \
  > "${COMPARISON_DIR}/logs/fairness_check.log" 2>&1

write_status "running"
for experiment in \
  "00_baseline_resnet34_unet_pw12" \
  "01_tbn_d1" \
  "02_tbn_d1_hnproto" \
  "03_skipgate_d4d3"
do
  bash "${EXPERIMENT_ROOT}/${experiment}/run.sh" \
    > "${COMPARISON_DIR}/logs/${experiment}.log" 2>&1
done

write_status "summarizing"
"${PYTHON_BIN}" "${COMPARISON_DIR}/summarize_attention_experiments.py" \
  --experiment-root "${EXPERIMENT_ROOT}" \
  --output-dir "${COMPARISON_DIR}/results" \
  > "${COMPARISON_DIR}/logs/summary.log" 2>&1

write_status "finished"
