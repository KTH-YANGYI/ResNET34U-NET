#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="${1:?usage: run_single_experiment.sh EXPERIMENT_DIR CONFIG BUILD_PROTOTYPES DESCRIPTION}"
CONFIG="${2:?usage: run_single_experiment.sh EXPERIMENT_DIR CONFIG BUILD_PROTOTYPES DESCRIPTION}"
BUILD_PROTOTYPES="${3:?usage: run_single_experiment.sh EXPERIMENT_DIR CONFIG BUILD_PROTOTYPES DESCRIPTION}"
DESCRIPTION="${4:?usage: run_single_experiment.sh EXPERIMENT_DIR CONFIG BUILD_PROTOTYPES DESCRIPTION}"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${EXPERIMENT_DIR}/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FOLDS_TEXT="${FOLDS:-0 1 2 3}"
CUDA_FOLD_GPUS_TEXT="${CUDA_FOLD_GPUS:-0,1,2,3}"

cd "${PROJECT_ROOT}"

read -r -a FOLDS_ARRAY <<< "${FOLDS_TEXT}"
IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_FOLD_GPUS_TEXT}"

if [ "${#FOLDS_ARRAY[@]}" -ne "${#GPU_ARRAY[@]}" ]; then
  echo "FOLDS count (${#FOLDS_ARRAY[@]}) must match CUDA_FOLD_GPUS count (${#GPU_ARRAY[@]})." >&2
  exit 2
fi

mkdir -p "${EXPERIMENT_DIR}/logs" "${EXPERIMENT_DIR}/results"

write_status() {
  local status="$1"
  {
    echo "# Experiment Status"
    echo
    echo "- Status: ${status}"
    echo "- Description: ${DESCRIPTION}"
    echo "- Project root: ${PROJECT_ROOT}"
    echo "- Experiment dir: ${EXPERIMENT_DIR}"
    echo "- Config: ${CONFIG}"
    echo "- Folds: ${FOLDS_ARRAY[*]}"
    echo "- CUDA fold GPUs: ${CUDA_FOLD_GPUS_TEXT}"
    echo "- Build prototype bank: ${BUILD_PROTOTYPES}"
    echo "- Updated: $(date -Is)"
    echo "- Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  } > "${EXPERIMENT_DIR}/RUNNING_STATUS.md"
}

run_parallel_folds() {
  local phase="$1"
  local status=0
  local pids=()

  for idx in "${!FOLDS_ARRAY[@]}"; do
    local fold="${FOLDS_ARRAY[$idx]}"
    local gpu="${GPU_ARRAY[$idx]}"
    (
      set -euo pipefail
      export CUDA_VISIBLE_DEVICES="${gpu}"
      export PYTHONUNBUFFERED=1
      echo "${phase} fold=${fold} gpu=${gpu} started $(date -Is)"
      if [ "${phase}" = "prototype_bank" ]; then
        "${PYTHON_BIN}" scripts/build_stage1_prototype_bank.py --config "${CONFIG}" --fold "${fold}"
      elif [ "${phase}" = "stage2_train" ]; then
        "${PYTHON_BIN}" scripts/train_stage2.py --config "${CONFIG}" --fold "${fold}"
      else
        echo "Unknown phase: ${phase}" >&2
        exit 2
      fi
      echo "${phase} fold=${fold} gpu=${gpu} finished $(date -Is)"
    ) > "${EXPERIMENT_DIR}/logs/fold${fold}_${phase}.log" 2>&1 &
    local pid="$!"
    echo "${pid}" > "${EXPERIMENT_DIR}/logs/fold${fold}_${phase}.pid"
    pids+=("${pid}")
  done

  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done

  if [ "${status}" -ne 0 ]; then
    echo "${phase} failed; see ${EXPERIMENT_DIR}/logs/*_${phase}.log" >&2
    exit "${status}"
  fi
}

write_experiment_manifest() {
  {
    echo "# $(basename "${EXPERIMENT_DIR}")"
    echo
    echo "- Description: ${DESCRIPTION}"
    echo "- Config: ${CONFIG}"
    echo "- Results: ${EXPERIMENT_DIR}/results"
    echo "- Logs: ${EXPERIMENT_DIR}/logs"
    echo "- Folds: ${FOLDS_ARRAY[*]}"
    echo "- CUDA fold GPUs: ${CUDA_FOLD_GPUS_TEXT}"
    echo "- Build prototype bank: ${BUILD_PROTOTYPES}"
    echo "- Started: $(date -Is)"
    echo "- Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo
    echo "Fairness settings are fixed by the config set and validated by comparison/check_fairness.py."
  } > "${EXPERIMENT_DIR}/EXPERIMENT.md"
}

write_status "running"
write_experiment_manifest

if [ "${BUILD_PROTOTYPES}" = "1" ]; then
  run_parallel_folds "prototype_bank"
fi

run_parallel_folds "stage2_train"

"${PYTHON_BIN}" scripts/search_oof_postprocess.py --config "${CONFIG}" --folds "$(IFS=,; echo "${FOLDS_ARRAY[*]}")" \
  > "${EXPERIMENT_DIR}/logs/oof_postprocess.log" 2>&1

"${PYTHON_BIN}" scripts/infer_holdout_ensemble.py --config "${CONFIG}" --folds "$(IFS=,; echo "${FOLDS_ARRAY[*]}")" \
  > "${EXPERIMENT_DIR}/logs/holdout_ensemble.log" 2>&1

{
  echo
  echo "- Finished: $(date -Is)"
  echo "- OOF: ${EXPERIMENT_DIR}/results/stage2/oof_global_postprocess.json"
  echo "- Holdout: ${EXPERIMENT_DIR}/results/stage2/holdout_ensemble/holdout_metrics.json"
} >> "${EXPERIMENT_DIR}/EXPERIMENT.md"

write_status "finished"
