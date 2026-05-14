#!/usr/bin/env bash
set -euo pipefail

stage1_config="${1:-configs/stage1_p0_a40_s1e50.yaml}"
stage2_config="${2:-configs/stage2_p0_a40_s1e50_s2e50_bs48.yaml}"
experiment_dir="${EXPERIMENT_DIR:-outputs/experiments/p0_a40_s1e50_s2e50_bs48_20260509}"
folds=(${FOLDS:-0 1 2 3})

mkdir -p "${experiment_dir}/logs"

pids=()
for fold in "${folds[@]}"; do
  gpu="${fold}"
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PYTHONUNBUFFERED=1
    echo "pipeline fold=${fold} gpu=${gpu} started $(date -Is)"
    python scripts/train_stage1.py --config "${stage1_config}" --fold "${fold}"
    python scripts/train_stage2.py --config "${stage2_config}" --fold "${fold}"
    echo "pipeline fold=${fold} gpu=${gpu} finished $(date -Is)"
  ) > "${experiment_dir}/logs/fold${fold}_pipeline.log" 2>&1 &
  pid="$!"
  echo "${pid}" > "${experiment_dir}/logs/fold${fold}.pid"
  pids+=("${pid}")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [ "${status}" -ne 0 ]; then
  echo "At least one pipeline fold failed; skip pooled OOF postprocess search." >&2
  exit "${status}"
fi

python scripts/search_oof_postprocess.py --config "${stage2_config}" --folds "$(IFS=,; echo "${folds[*]}")" \
  > "${experiment_dir}/logs/oof_postprocess.log" 2>&1
