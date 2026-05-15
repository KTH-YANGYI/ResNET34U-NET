#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
GPU_DEVICES="${CUDA_GPUS:-0,1,2,3}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_DIR/../dataset_crack_normal_unet_811}"
PLAN_ROOT="${PLAN_ROOT:-outputs/experiments/811_fixed_split}"
SCOPE="${SCOPE:-phase1}"
BASELINE_PROFILE="${BASELINE_PROFILE:-full}"
WITH_HOLDOUT="${WITH_HOLDOUT:-1}"

ROOT_ABS="$PROJECT_DIR/$PLAN_ROOT"
CANONICAL_CONFIG="$PROJECT_DIR/configs/canonical_baseline.yaml"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_811_experiment_plan.sh [options]

Options:
  --scope NAME          sanity, phase1, phase2, or phase2_full. Default: phase1.
  --baseline-profile P  Baseline profile for phase2: full, normal_fp_loss,
                        no_hard_normal, or no_stage1. Default: full.
  --gpus IDS           Comma-separated GPU ids to expose. Default: 0,1,2,3.
  --dataset-root PATH   Dataset root. Default: ../dataset_crack_normal_unet_811.
  --no-holdout          Skip holdout/test evaluation after Stage2 runs.
  --python PATH         Python executable. Default: python or $PYTHON.
  -h, --help            Show this help.

Scopes:
  sanity       M0 baseline seed 20260515 only.
  phase1       Baseline selection: M0/T1/T2/T3 seed 20260515 only.
  phase2       Architecture comparison after baseline is chosen:
               M0/M1/M4/M5 3 seeds, M2/M3/M6 1 seed.
  phase2_full  phase2 plus M2/M3 3 seeds.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scope)
      SCOPE="$2"
      shift 2
      ;;
    --baseline-profile)
      BASELINE_PROFILE="$2"
      shift 2
      ;;
    --gpus)
      GPU_DEVICES="$2"
      shift 2
      ;;
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --no-holdout)
      WITH_HOLDOUT=0
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

case "$SCOPE" in
  sanity|phase1|phase2|phase2_full)
    ;;
  *)
    echo "Unsupported scope: $SCOPE" >&2
    exit 2
    ;;
esac

case "$BASELINE_PROFILE" in
  full|normal_fp_loss|no_hard_normal|no_stage1)
    ;;
  *)
    echo "Unsupported baseline profile: $BASELINE_PROFILE" >&2
    exit 2
    ;;
esac

if [[ "$SCOPE" == "phase1" || "$SCOPE" == "sanity" ]]; then
  RUN_ROOT_REL="$PLAN_ROOT/phase1_baseline_selection"
elif [[ "$SCOPE" == "phase2_full" ]]; then
  RUN_ROOT_REL="$PLAN_ROOT/phase2_architecture_${BASELINE_PROFILE}_full"
else
  RUN_ROOT_REL="$PLAN_ROOT/phase2_architecture_${BASELINE_PROFILE}"
fi

RUN_ROOT_ABS="$PROJECT_DIR/$RUN_ROOT_REL"
CONFIG_DIR="$RUN_ROOT_ABS/configs"
LOG_DIR="$RUN_ROOT_ABS/logs"
mkdir -p "$CONFIG_DIR" "$LOG_DIR"

run_logged() {
  local name="$1"
  shift

  local log_path="$LOG_DIR/${name}.log"
  echo "[$(date '+%F %T')] start ${name}"
  if "$@" > "$log_path" 2>&1; then
    echo "[$(date '+%F %T')] done  ${name} | log: ${log_path}"
  else
    echo "[$(date '+%F %T')] fail  ${name} | log: ${log_path}" >&2
    tail -n 120 "$log_path" >&2 || true
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

yaml_list() {
  local csv="$1"
  local output="["
  local first=1
  IFS=',' read -r -a parts <<< "$csv"
  for part in "${parts[@]}"; do
    part="$(echo "$part" | xargs)"
    [[ -z "$part" ]] && continue
    if [[ "$first" -eq 0 ]]; then
      output+=", "
    fi
    output+="\"$part\""
    first=0
  done
  output+="]"
  echo "$output"
}

baseline_profile_yaml() {
  case "$BASELINE_PROFILE" in
    full)
      return
      ;;
    normal_fp_loss)
      cat <<'EOF'
  normal_fp_loss_weight: 0.05
  normal_fp_topk_ratio: 0.10
EOF
      ;;
    no_hard_normal)
      cat <<'EOF'
  use_hard_normal_replay: false
  stage2_hard_normal_ratio: 0.0
EOF
      ;;
    no_stage1)
      cat <<'EOF'
  pretrained: true
  allow_pretrained_fallback: false
EOF
      ;;
  esac
}

selected_stage1_checkpoint() {
  local seed="$1"
  if [[ "$BASELINE_PROFILE" == "no_stage1" ]]; then
    echo ""
  else
    stage1_checkpoint "$seed"
  fi
}

selected_baseline_strict() {
  if [[ "$BASELINE_PROFILE" == "no_stage1" ]]; then
    echo "false"
  else
    echo "true"
  fi
}

stage1_config_path() {
  local seed="$1"
  echo "$CONFIG_DIR/stage1_s${seed}.yaml"
}

stage1_save_dir() {
  local seed="$1"
  echo "$PLAN_ROOT/stage1_s${seed}/stage1"
}

stage1_checkpoint() {
  local seed="$1"
  echo "$PLAN_ROOT/stage1_s${seed}/stage1/best_stage1.pt"
}

write_stage1_config() {
  local seed="$1"
  local config_path
  config_path="$(stage1_config_path "$seed")"

  cat > "$config_path" <<EOF
extends: $CANONICAL_CONFIG
name: 811_stage1_s${seed}
version: "2026-05-15"
canonical: false

common:
  seed: $seed

stage1:
  save_dir: $(stage1_save_dir "$seed")
EOF
}

run_config_path() {
  local run_name="$1"
  echo "$CONFIG_DIR/${run_name}.yaml"
}

write_stage2_config() {
  local run_name="$1"
  local seed="$2"
  local model_name="$3"
  local model_variant="$4"
  local strict="$5"
  local checkpoint="$6"
  local extra_yaml="${7:-}"
  local config_path
  config_path="$(run_config_path "$run_name")"

  cat > "$config_path" <<EOF
extends: $CANONICAL_CONFIG
name: $run_name
version: "2026-05-15"
canonical: false

common:
  seed: $seed

stage1:
  save_dir: $(stage1_save_dir "$seed")

stage2:
  model_name: $model_name
  model_variant: $model_variant
  stage1_checkpoint: "$checkpoint"
  stage1_load_strict: $strict
  save_dir: $RUN_ROOT_REL/$run_name/stage2
EOF

  if [[ -n "$extra_yaml" ]]; then
    printf '%s\n' "$extra_yaml" >> "$config_path"
  fi

  if [[ "${ACTIVE_BASELINE_EXTRA_YAML:-}" != "" ]]; then
    printf '%s\n' "$ACTIVE_BASELINE_EXTRA_YAML" >> "$config_path"
  fi
}

write_no_stage1_config() {
  local run_name="$1"
  local seed="$2"
  local config_path
  config_path="$(run_config_path "$run_name")"

  cat > "$config_path" <<EOF
extends: $CANONICAL_CONFIG
name: $run_name
version: "2026-05-15"
canonical: false

common:
  seed: $seed

stage2:
  model_name: 811_t1_no_stage1
  model_variant: resnet34_unet_baseline
  stage1_checkpoint: ""
  stage1_load_strict: false
  pretrained: true
  allow_pretrained_fallback: false
  save_dir: $RUN_ROOT_REL/$run_name/stage2
EOF
}

run_stage1() {
  local seed="$1"
  local config_path
  config_path="$(stage1_config_path "$seed")"
  write_stage1_config "$seed"

  if [[ -f "$PROJECT_DIR/$(stage1_checkpoint "$seed")" ]]; then
    echo "[$(date '+%F %T')] skip  stage1_s${seed} | checkpoint exists"
    return
  fi

  run_logged "stage1_s${seed}" \
    run_train scripts/train_stage1.py --config "$config_path"
}

run_stage2() {
  local run_name="$1"
  local config_path
  config_path="$(run_config_path "$run_name")"

  if [[ -f "$PROJECT_DIR/$RUN_ROOT_REL/$run_name/stage2/best_stage2.pt" ]]; then
    echo "[$(date '+%F %T')] skip  ${run_name} | checkpoint exists"
  else
    run_logged "$run_name" \
      run_train scripts/train_stage2.py --config "$config_path"
  fi

  if [[ "$WITH_HOLDOUT" -eq 1 ]]; then
    if [[ -f "$PROJECT_DIR/$RUN_ROOT_REL/$run_name/stage2/holdout/holdout_metrics.json" ]]; then
      echo "[$(date '+%F %T')] skip  ${run_name}_holdout | metrics exist"
    else
      run_logged "${run_name}_holdout" \
        env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$PYTHON_BIN" scripts/infer_holdout.py --config "$config_path"
    fi
  fi
}

run_prototype_bank() {
  local run_name="$1"
  local config_path
  config_path="$(run_config_path "$run_name")"

  if [[ -f "$PROJECT_DIR/$RUN_ROOT_REL/$run_name/prototype_bank.pt" ]]; then
    echo "[$(date '+%F %T')] skip  ${run_name}_prototype_bank | bank exists"
    return
  fi

  run_logged "${run_name}_prototype_bank" \
    env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$PYTHON_BIN" scripts/build_stage1_prototype_bank.py --config "$config_path"
}

write_baseline_run() {
  local seed="$1"
  local run_name="811_m0_baseline_s${seed}"
  write_stage2_config \
    "$run_name" \
    "$seed" \
    "811_m0_baseline" \
    "resnet34_unet_baseline" \
    "$(selected_baseline_strict)" \
    "$(selected_stage1_checkpoint "$seed")"
}

write_tbn_run() {
  local seed="$1"
  local run_name="811_m1_tbn_d1_s${seed}"
  write_stage2_config \
    "$run_name" \
    "$seed" \
    "811_m1_tbn_d1" \
    "tbn_d1" \
    "false" \
    "$(selected_stage1_checkpoint "$seed")" \
    "  transformer_bottleneck_layers: 1
  transformer_bottleneck_heads: 8
  transformer_bottleneck_dropout: 0.1"
}

write_skip_run() {
  local seed="$1"
  local run_id="$2"
  local levels_csv="$3"
  local run_name="811_${run_id}_s${seed}"
  write_stage2_config \
    "$run_name" \
    "$seed" \
    "811_${run_id}" \
    "skipgate_d4d3" \
    "false" \
    "$(selected_stage1_checkpoint "$seed")" \
    "  skip_attention_levels: $(yaml_list "$levels_csv")
  skip_attention_gamma_init: 0.0"
}

write_selfattn_run() {
  local seed="$1"
  local run_name="811_m5_selfattn_d4d3_s${seed}"
  write_stage2_config \
    "$run_name" \
    "$seed" \
    "811_m5_selfattn_d4d3" \
    "selfattn_d4d3" \
    "false" \
    "$(selected_stage1_checkpoint "$seed")" \
    "  self_attention_levels: [\"d4\", \"d3\"]
  self_attention_heads: 4
  self_attention_dropout: 0.1
  self_attention_sr_ratios:
    d4: 2
    d3: 4
  self_attention_gamma_init: 0.0"
}

write_proto_run() {
  local seed="$1"
  local run_name="811_m6_hnproto_s${seed}"
  write_stage2_config \
    "$run_name" \
    "$seed" \
    "811_m6_hnproto" \
    "tbn_d1_hnproto" \
    "false" \
    "$(selected_stage1_checkpoint "$seed")" \
    "  transformer_bottleneck_layers: 1
  transformer_bottleneck_heads: 8
  transformer_bottleneck_dropout: 0.1
  prototype_bank_path: $RUN_ROOT_REL/$run_name/prototype_bank.pt
  prototype_attention_heads: 8
  prototype_attention_dropout: 0.1
  prototype_pos_max: 128
  prototype_neg_max: 128
  prototype_l2_normalize: true
  prototype_batch_size: 64"
}

write_ablation_configs() {
  local seed="$1"
  write_no_stage1_config "811_t1_no_stage1_s${seed}" "$seed"
  write_stage2_config \
    "811_t2_no_hard_normal_s${seed}" \
    "$seed" \
    "811_t2_no_hard_normal" \
    "resnet34_unet_baseline" \
    "true" \
    "$(stage1_checkpoint "$seed")" \
    "  use_hard_normal_replay: false
  stage2_hard_normal_ratio: 0.0"
  write_stage2_config \
    "811_t3_normal_fp_loss_s${seed}" \
    "$seed" \
    "811_t3_normal_fp_loss" \
    "resnet34_unet_baseline" \
    "true" \
    "$(stage1_checkpoint "$seed")" \
    "  normal_fp_loss_weight: 0.05
  normal_fp_topk_ratio: 0.10"
}

cd "$PROJECT_DIR"

echo "Project: $PROJECT_DIR"
echo "Dataset: $DATASET_ROOT"
echo "Scope:   $SCOPE"
echo "Baseline profile: $BASELINE_PROFILE"
echo "GPUs:    $GPU_DEVICES"
echo "Plan:    $PLAN_ROOT"
echo "Run dir: $RUN_ROOT_REL"

run_logged "check_environment" "$PYTHON_BIN" scripts/check_environment.py --strict
run_logged "prepare_samples" "$PYTHON_BIN" scripts/prepare_samples.py --dataset-root "$DATASET_ROOT"
run_logged "build_patch_index" "$PYTHON_BIN" scripts/build_patch_index.py --config "$CANONICAL_CONFIG"

ACTIVE_BASELINE_EXTRA_YAML=""
if [[ "$SCOPE" == "phase2" || "$SCOPE" == "phase2_full" ]]; then
  ACTIVE_BASELINE_EXTRA_YAML="$(baseline_profile_yaml)"
fi

if [[ "$SCOPE" == "phase2" || "$SCOPE" == "phase2_full" ]]; then
  seeds=(20260515 20260516 20260517)
else
  seeds=(20260515)
fi

for seed in "${seeds[@]}"; do
  run_stage1 "$seed"
done

for seed in "${seeds[@]}"; do
  write_baseline_run "$seed"
  run_stage2 "811_m0_baseline_s${seed}"
done

if [[ "$SCOPE" == "sanity" ]]; then
  echo "[$(date '+%F %T')] sanity scope finished."
  exit 0
fi

if [[ "$SCOPE" == "phase1" ]]; then
  write_ablation_configs 20260515
  run_stage2 "811_t1_no_stage1_s20260515"
  run_stage2 "811_t2_no_hard_normal_s20260515"
  run_stage2 "811_t3_normal_fp_loss_s20260515"
  echo "[$(date '+%F %T')] phase1 baseline selection finished. Review phase1 results and choose --baseline-profile before phase2."
  exit 0
fi

for seed in "${seeds[@]}"; do
  write_tbn_run "$seed"
  run_stage2 "811_m1_tbn_d1_s${seed}"
done

for seed in "${seeds[@]}"; do
  write_skip_run "$seed" "m4_skip_d4d3" "d4,d3"
  run_stage2 "811_m4_skip_d4d3_s${seed}"
done

for seed in "${seeds[@]}"; do
  write_selfattn_run "$seed"
  run_stage2 "811_m5_selfattn_d4d3_s${seed}"
done

skip_seeds=(20260515)
if [[ "$SCOPE" == "phase2_full" ]]; then
  skip_seeds=(20260515 20260516 20260517)
fi
for seed in "${skip_seeds[@]}"; do
  write_skip_run "$seed" "m2_skip_d4" "d4"
  run_stage2 "811_m2_skip_d4_s${seed}"
  write_skip_run "$seed" "m3_skip_d3" "d3"
  run_stage2 "811_m3_skip_d3_s${seed}"
done

write_proto_run 20260515
run_prototype_bank "811_m6_hnproto_s20260515"
run_stage2 "811_m6_hnproto_s20260515"

echo "[$(date '+%F %T')] 811 experiment plan finished."
