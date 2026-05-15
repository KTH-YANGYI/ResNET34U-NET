#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
GPU_DEVICES="${CUDA_GPUS:-${CUDA_GPU:-0}}"
CONFIG="configs/canonical_baseline.yaml"
CANONICAL_CONFIG="configs/canonical_baseline.yaml"
LOG_DIR="outputs/logs"
RUN_PREPARE=1
RUN_STAGE1=1
RUN_STAGE2=1
RUN_HOLDOUT=0
ALLOW_EXPERIMENT_STAGE1=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_pipeline.sh [options]

Options:
  --gpu ID              GPU id to expose. Default: 0.
  --gpus IDS            Comma-separated GPU ids to expose, e.g. 0,1,2,3.
  --skip-prepare        Skip prepare_samples.py and build_patch_index.py.
  --config PATH         Use one canonical config for Stage1 and Stage2.
                        Default: configs/canonical_baseline.yaml.
  --stage1-only         Run only Stage1 after optional preparation.
  --stage2-only         Run only Stage2 after optional sample preparation.
  --allow-experiment-stage1
                        Allow Stage1 with a non-canonical experiment config.
  --with-holdout        Run infer_holdout.py after Stage2.
  --python PATH         Python executable. Default: python or $PYTHON.
  -h, --help            Show this help.

Examples:
  bash scripts/train_pipeline.sh --gpu 0
  bash scripts/train_pipeline.sh --gpus 0,1,2,3
  bash scripts/train_pipeline.sh --gpu 0 --with-holdout
  bash scripts/train_pipeline.sh --gpu 0 --skip-prepare
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU_DEVICES="$2"
      shift 2
      ;;
    --gpus)
      GPU_DEVICES="$2"
      shift 2
      ;;
    --skip-prepare)
      RUN_PREPARE=0
      shift
      ;;
    --config)
      CONFIG="$2"
      shift 2
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
    --allow-experiment-stage1)
      ALLOW_EXPERIMENT_STAGE1=1
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

gpu_count() {
  local ids="$1"
  if [[ -z "$ids" ]]; then
    echo 0
    return
  fi
  awk -F',' '{print NF}' <<< "$ids"
}

run_train() {
  local script_path="$1"
  shift

  local count
  count="$(gpu_count "$GPU_DEVICES")"
  if [[ "$count" -gt 1 ]]; then
    env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" \
      "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$count" \
      "$script_path" "$@"
  else
    env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$PYTHON_BIN" "$script_path" "$@"
  fi
}

echo "Python: ${PYTHON_BIN}"
echo "GPUs:   ${GPU_DEVICES}"
echo "Config: ${CONFIG}"
echo "Logs:   ${LOG_DIR}"

canonical_abs="$(cd "$(dirname "$CANONICAL_CONFIG")" && pwd -P)/$(basename "$CANONICAL_CONFIG")"
if [[ -d "$(dirname "$CONFIG")" ]]; then
  config_abs="$(cd "$(dirname "$CONFIG")" && pwd -P)/$(basename "$CONFIG")"
else
  config_abs="$CONFIG"
fi

if [[ "$RUN_STAGE1" -eq 1 && "$config_abs" != "$canonical_abs" && "$ALLOW_EXPERIMENT_STAGE1" -ne 1 ]]; then
  cat >&2 <<EOF
Refusing to run Stage1 with experiment config: ${CONFIG}

Experiment configs usually inherit the canonical Stage1 paths and are intended
for Stage2-only runs after the canonical Stage1 checkpoint exists.

Use:
  bash scripts/train_pipeline.sh --gpus ${GPU_DEVICES} --config ${CONFIG} --stage2-only

If you intentionally want a separate full experiment Stage1, rerun with:
  --allow-experiment-stage1
EOF
  exit 2
fi

if [[ "$RUN_PREPARE" -eq 1 ]]; then
  run_logged "prepare_samples" "$PYTHON_BIN" scripts/prepare_samples.py

  if [[ "$RUN_STAGE1" -eq 1 ]]; then
    run_logged "build_patch_index" "$PYTHON_BIN" scripts/build_patch_index.py --config "$CONFIG"
  fi
fi

if [[ "$RUN_STAGE1" -eq 1 ]]; then
  run_logged "stage1" run_train scripts/train_stage1.py --config "$CONFIG"
fi

if [[ "$RUN_STAGE2" -eq 1 ]]; then
  run_logged "stage2" run_train scripts/train_stage2.py --config "$CONFIG"
fi

if [[ "$RUN_HOLDOUT" -eq 1 ]]; then
  run_logged "holdout" env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$PYTHON_BIN" scripts/infer_holdout.py --config "$CONFIG"
fi

echo "[$(date '+%F %T')] requested pipeline finished."
