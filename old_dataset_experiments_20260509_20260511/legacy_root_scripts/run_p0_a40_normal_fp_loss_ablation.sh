#!/usr/bin/env bash
set -euo pipefail

runs=(
  "pw12_fp003|configs/stage2_p0_a40_e50_bs48_pw12_fp003.yaml|outputs/experiments/p0_a40_e50_bs48_pw12_fp003_20260509|normal_fp_loss_weight=0.03 on the pos_weight=12 Stage2 baseline"
  "pw12_fp005|configs/stage2_p0_a40_e50_bs48_pw12_fp005.yaml|outputs/experiments/p0_a40_e50_bs48_pw12_fp005_20260509|normal_fp_loss_weight=0.05 on the pos_weight=12 Stage2 baseline"
)

write_experiment_note() {
  local name="$1"
  local config="$2"
  local experiment_dir="$3"
  local purpose="$4"

  mkdir -p "${experiment_dir}"
  cat > "${experiment_dir}/EXPERIMENT.md" <<NOTE
# ${name}

Purpose: ${purpose}.

This is the second-batch follow-up after hard-normal replay was stabilized.
It keeps the best pos_weight ablation baseline at pos_weight=12 and tests
whether a small normal false-positive penalty improves pooled OOF behavior
without hurting defect recall.

- Config: ${config}
- Runner: scripts/run_p0_a40_stage2_e50_bs48.sh
- Output root: ${experiment_dir}
- Fold logs: ${experiment_dir}/logs/fold{0..3}_stage2.log
- Pooled OOF log: ${experiment_dir}/logs/oof_postprocess.log
- Pooled OOF result: ${experiment_dir}/stage2/oof_global_postprocess.json

Original training outputs are left in place. This directory is the source
record for this individual experiment.
NOTE
}

for item in "${runs[@]}"; do
  IFS="|" read -r name config experiment_dir purpose <<< "${item}"
  write_experiment_note "${name}" "${config}" "${experiment_dir}" "${purpose}"

  if [ -f "${experiment_dir}/stage2/oof_global_postprocess.json" ] && [ "${FORCE_RERUN:-0}" != "1" ]; then
    echo "normal_fp_loss_ablation ${name} already has pooled OOF; skip $(date -Is)"
    continue
  fi

  echo "normal_fp_loss_ablation ${name} started $(date -Is)"
  EXPERIMENT_DIR="${experiment_dir}" bash scripts/run_p0_a40_stage2_e50_bs48.sh "${config}"
  echo "normal_fp_loss_ablation ${name} finished $(date -Is)"
done
