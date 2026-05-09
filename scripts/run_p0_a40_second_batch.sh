#!/usr/bin/env bash
set -euo pipefail

runs=(
  "pw6|configs/stage2_p0_a40_e50_bs48_pw6.yaml|outputs/experiments/p0_a40_e50_bs48_pw6_20260509"
  "pw8|configs/stage2_p0_a40_e50_bs48_pw8.yaml|outputs/experiments/p0_a40_e50_bs48_pw8_20260509"
  "pw12|configs/stage2_p0_a40_e50_bs48_pw12.yaml|outputs/experiments/p0_a40_e50_bs48_pw12_20260509"
)

if [ "${RUN_NORMAL_FP:-0}" = "1" ]; then
  runs+=(
    "pw12_fp003|configs/stage2_p0_a40_e50_bs48_pw12_fp003.yaml|outputs/experiments/p0_a40_e50_bs48_pw12_fp003_20260509"
    "pw12_fp005|configs/stage2_p0_a40_e50_bs48_pw12_fp005.yaml|outputs/experiments/p0_a40_e50_bs48_pw12_fp005_20260509"
  )
fi

for item in "${runs[@]}"; do
  IFS="|" read -r name config experiment_dir <<< "${item}"
  echo "second_batch ${name} started $(date -Is)"
  EXPERIMENT_DIR="${experiment_dir}" bash scripts/run_p0_a40_stage2_e50_bs48.sh "${config}"
  echo "second_batch ${name} finished $(date -Is)"
done
