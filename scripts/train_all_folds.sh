#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
FOLDS="0,1,2,3"
GPUS="${CUDA_FOLD_GPUS:-}"
STAGE1_CONFIG="configs/stage1.yaml"
STAGE2_CONFIG="configs/stage2.yaml"
LOG_DIR="outputs/logs"
RUN_PREPARE=1
RUN_STAGE1=1
RUN_STAGE2=1
RUN_HOLDOUT=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_all_folds.sh [options]

Options:
  --gpus LIST           GPU ids to use, for example 0 or 0,1,2,3.
                        If omitted, the script tries to detect GPUs with nvidia-smi.
  --folds LIST          Fold ids to run. Default: 0,1,2,3.
  --skip-prepare        Skip prepare_samples.py and build_patch_index.py.
  --stage1-only         Run only Stage1 after optional preparation.
  --stage2-only         Run only Stage2 after optional sample preparation.
  --with-holdout        Run infer_holdout.py for each fold after Stage2.
  --python PATH         Python executable. Default: python or $PYTHON.
  -h, --help            Show this help.

Examples:
  bash scripts/train_all_folds.sh --gpus 0
  bash scripts/train_all_folds.sh --gpus 0,1,2,3
  bash scripts/train_all_folds.sh --gpus 0,1 --skip-prepare
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --folds)
      FOLDS="$2"
      shift 2
      ;;
    --skip-prepare)
      RUN_PREPARE=0
      shift
      ;;
    --stage1-only)
      RUN_STAGE2=0
      RUN_HOLDOUT=0
      shift
      ;;
    --stage2-only)
      RUN_STAGE1=0
      shift
      ;;
    --with-holdout)
      RUN_HOLDOUT=1
      shift
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

FOLDS="${FOLDS// /}"
GPUS="${GPUS// /}"

if [[ -z "$GPUS" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
  fi
fi

if [[ -z "$GPUS" ]]; then
  GPUS="0"
fi

IFS=',' read -r -a FOLD_LIST <<< "$FOLDS"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"

if [[ "${#FOLD_LIST[@]}" -eq 0 ]]; then
  echo "No folds selected." >&2
  exit 2
fi

if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  echo "No GPUs selected." >&2
  exit 2
fi

mkdir -p "$LOG_DIR"

run_logged() {
  local name="$1"
  shift

  local log_path="$LOG_DIR/${name}.log"
  echo "[$(date '+%F %T')] start ${name}"

  if "$@" > "$log_path" 2>&1; then
    echo "[$(date '+%F %T')] done  ${name} | log: ${log_path}"
  else
    echo "[$(date '+%F %T')] fail  ${name} | log: ${log_path}" >&2
    tail -n 80 "$log_path" >&2 || true
    exit 1
  fi
}

wait_batch() {
  local stage_name="$1"
  shift

  local status=0
  local item pid fold log_path

  for item in "$@"; do
    IFS='|' read -r pid fold log_path <<< "$item"
    if wait "$pid"; then
      echo "[$(date '+%F %T')] done  ${stage_name} fold ${fold} | log: ${log_path}"
    else
      echo "[$(date '+%F %T')] fail  ${stage_name} fold ${fold} | log: ${log_path}" >&2
      tail -n 80 "$log_path" >&2 || true
      status=1
    fi
  done

  if [[ "$status" -ne 0 ]]; then
    exit "$status"
  fi
}

run_fold_stage() {
  local stage_name="$1"
  local script_path="$2"
  local config_path="$3"

  local -a batch_items=()
  local gpu_count="${#GPU_LIST[@]}"
  local job_index=0
  local fold gpu log_path pid

  echo "Running ${stage_name} for folds: ${FOLDS} on GPUs: ${GPUS}"

  for fold in "${FOLD_LIST[@]}"; do
    gpu="${GPU_LIST[$((job_index % gpu_count))]}"
    log_path="$LOG_DIR/${stage_name}_fold${fold}.log"

    echo "[$(date '+%F %T')] start ${stage_name} fold ${fold} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$script_path" --config "$config_path" --fold "$fold" > "$log_path" 2>&1 &
    pid="$!"
    batch_items+=("${pid}|${fold}|${log_path}")
    job_index=$((job_index + 1))

    if [[ "${#batch_items[@]}" -eq "$gpu_count" ]]; then
      wait_batch "$stage_name" "${batch_items[@]}"
      batch_items=()
    fi
  done

  if [[ "${#batch_items[@]}" -gt 0 ]]; then
    wait_batch "$stage_name" "${batch_items[@]}"
  fi
}

echo "Python: ${PYTHON_BIN}"
echo "Folds:  ${FOLDS}"
echo "GPUs:   ${GPUS}"
echo "Logs:   ${LOG_DIR}"

if [[ "$RUN_PREPARE" -eq 1 ]]; then
  run_logged "prepare_samples" "$PYTHON_BIN" scripts/prepare_samples.py

  if [[ "$RUN_STAGE1" -eq 1 ]]; then
    run_logged "build_patch_index" "$PYTHON_BIN" scripts/build_patch_index.py --config "$STAGE1_CONFIG"
  fi
fi

if [[ "$RUN_STAGE1" -eq 1 ]]; then
  run_fold_stage "stage1" scripts/train_stage1.py "$STAGE1_CONFIG"
fi

if [[ "$RUN_STAGE2" -eq 1 ]]; then
  run_fold_stage "stage2" scripts/train_stage2.py "$STAGE2_CONFIG"
fi

if [[ "$RUN_HOLDOUT" -eq 1 ]]; then
  run_fold_stage "holdout" scripts/infer_holdout.py "$STAGE2_CONFIG"
fi

echo "[$(date '+%F %T')] all requested jobs finished."
