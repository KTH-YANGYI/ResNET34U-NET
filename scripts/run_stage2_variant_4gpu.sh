#!/usr/bin/env bash
set -euo pipefail

config="${1:?usage: run_stage2_variant_4gpu.sh CONFIG EXPERIMENT_DIR}"
experiment_dir="${2:?usage: run_stage2_variant_4gpu.sh CONFIG EXPERIMENT_DIR}"
fold_text="${FOLDS:-0,1,2,3}"
fold_text="${fold_text//,/ }"
folds=(${fold_text})
gpu_text="${CUDA_FOLD_GPUS:-0,1,2,3}"
gpu_text="${gpu_text//,/ }"
gpus=(${gpu_text})
build_prototypes="${BUILD_PROTOTYPES:-0}"

mkdir -p "${experiment_dir}/logs"

run_parallel_folds() {
  local phase="$1"
  local log_suffix="$3"
  local status=0
  local pids=()
  local idx=0

  if [ "${#gpus[@]}" -lt "${#folds[@]}" ]; then
    echo "Need at least one GPU id per concurrent fold in CUDA_FOLD_GPUS; got ${#gpus[@]} for ${#folds[@]} folds." >&2
    exit 2
  fi

  for fold in "${folds[@]}"; do
    local gpu="${gpus[$idx]}"
    idx=$((idx + 1))
    (
      set -euo pipefail
      export CUDA_VISIBLE_DEVICES="${gpu}"
      export PYTHONUNBUFFERED=1
      echo "${phase} fold=${fold} gpu=${gpu} started $(date -Is)"
      if [ "${phase}" = "prototype_bank" ]; then
        python scripts/build_stage1_prototype_bank.py --config "${config}" --fold "${fold}"
      else
        python scripts/train_stage2.py --config "${config}" --fold "${fold}"
      fi
      echo "${phase} fold=${fold} gpu=${gpu} finished $(date -Is)"
    ) > "${experiment_dir}/logs/fold${fold}_${log_suffix}.log" 2>&1 &
    local pid="$!"
    echo "${pid}" > "${experiment_dir}/logs/fold${fold}_${log_suffix}.pid"
    pids+=("${pid}")
  done

  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done

  if [ "${status}" -ne 0 ]; then
    echo "${phase} failed; see ${experiment_dir}/logs/*_${log_suffix}.log" >&2
    exit "${status}"
  fi
}

{
  echo "# Experiment Status"
  echo
  echo "- Config: ${config}"
  echo "- Experiment dir: ${experiment_dir}"
  echo "- Folds: ${folds[*]}"
  echo "- Started: $(date -Is)"
  echo "- Build prototypes: ${build_prototypes}"
} > "${experiment_dir}/RUNNING_STATUS.md"

if [ "${build_prototypes}" = "1" ]; then
  run_parallel_folds \
    "prototype_bank" \
    "prototype"
fi

run_parallel_folds \
  "stage2_train" \
  "stage2"

python scripts/search_oof_postprocess.py --config "${config}" --folds "$(IFS=,; echo "${folds[*]}")" \
  > "${experiment_dir}/logs/oof_postprocess.log" 2>&1

python scripts/infer_holdout_ensemble.py --config "${config}" --folds "$(IFS=,; echo "${folds[*]}")" \
  > "${experiment_dir}/logs/holdout_ensemble.log" 2>&1

{
  echo "# Experiment Status"
  echo
  echo "- Config: ${config}"
  echo "- Experiment dir: ${experiment_dir}"
  echo "- Folds: ${folds[*]}"
  echo "- Finished: $(date -Is)"
  echo "- OOF and holdout outputs follow the config paths."
} > "${experiment_dir}/RUNNING_STATUS.md"
