#!/usr/bin/env bash
set -euo pipefail

config="${1:?usage: run_stage2_variant_4gpu.sh CONFIG EXPERIMENT_DIR}"
experiment_dir="${2:?usage: run_stage2_variant_4gpu.sh CONFIG EXPERIMENT_DIR}"
folds=(${FOLDS:-0 1 2 3})
build_prototypes="${BUILD_PROTOTYPES:-0}"

mkdir -p "${experiment_dir}/logs"

run_parallel_folds() {
  local phase="$1"
  local command_template="$2"
  local log_suffix="$3"
  local status=0
  local pids=()

  for fold in "${folds[@]}"; do
    local gpu="${fold}"
    (
      set -euo pipefail
      export CUDA_VISIBLE_DEVICES="${gpu}"
      export PYTHONUNBUFFERED=1
      echo "${phase} fold=${fold} gpu=${gpu} started $(date -Is)"
      eval "${command_template}"
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
    "python scripts/build_stage1_prototype_bank.py --config \"${config}\" --fold \"\${fold}\"" \
    "prototype"
fi

run_parallel_folds \
  "stage2_train" \
  "python scripts/train_stage2.py --config \"${config}\" --fold \"\${fold}\"" \
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
  echo "- OOF: ${experiment_dir}/results/stage2/oof_global_postprocess.json"
  echo "- Holdout: ${experiment_dir}/results/stage2/holdout_ensemble/holdout_metrics.json"
} > "${experiment_dir}/RUNNING_STATUS.md"
