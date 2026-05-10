#!/usr/bin/env bash
set -euo pipefail

experiment_root="${1:-/mimer/NOBACKUP/groups/smart-rail/Yi Yang/CV_contact_wire/UNET_two_stage/transformer_attention_milestone_20260510}"

run_one() {
  local name="$1"
  local config="$2"
  local build_prototypes="$3"
  local experiment_dir="${experiment_root}/${name}"
  local source_dir="${experiment_dir}/source"

  if [ ! -d "${source_dir}" ]; then
    echo "Missing source snapshot: ${source_dir}" >&2
    exit 1
  fi

  echo "===== ${name} started $(date -Is) ====="
  (
    cd "${source_dir}"
    BUILD_PROTOTYPES="${build_prototypes}" bash scripts/run_stage2_variant_4gpu.sh "${config}" "${experiment_dir}"
  )
  echo "===== ${name} finished $(date -Is) ====="
}

{
  echo "# Experiment Set Status"
  echo
  echo "- Experiment root: ${experiment_root}"
  echo "- Started: $(date -Is)"
  echo "- GPU plan: 4 x A40, one fold per GPU"
} > "${experiment_root}/RUNNING_STATUS.md"

run_one "01_transformer_bottleneck" "configs/transformer_a40_20260510/stage2_tbn_d1_a40.yaml" "0"
run_one "02_hard_negative_prototype_attention" "configs/transformer_a40_20260510/stage2_hnproto_a40.yaml" "1"
run_one "03_decoder_skip_attention_gate" "configs/transformer_a40_20260510/stage2_skipgate_a40.yaml" "0"

python "${experiment_root}/03_decoder_skip_attention_gate/source/scripts/summarize_transformer_experiments.py" \
  --experiment-root "${experiment_root}"

{
  echo "# Experiment Set Status"
  echo
  echo "- Experiment root: ${experiment_root}"
  echo "- Finished: $(date -Is)"
  echo "- Final report: ${experiment_root}/FINAL_EXPERIMENT_REPORT.md"
} > "${experiment_root}/RUNNING_STATUS.md"
