#!/usr/bin/env bash
set -euo pipefail

cache_note_dir="outputs/experiments/p0_a40_patchdataset_cache_20260509"
mkdir -p "${cache_note_dir}"
cat > "${cache_note_dir}/EXPERIMENT.md" <<NOTE
# patchdataset_worker_cache

Purpose: enable worker-local image/mask caching in PatchDataset for Stage1 patch
training and replay scans.

This is a code/runtime optimization. It does not change Stage2 predictions by itself.
Use configs/stage1_p0_a40_cache.yaml for a Stage1 rerun that enables:

- patch_worker_cache: true
- patch_worker_cache_max_items: 0

Each DataLoader worker keeps its own in-process cache, so repeated patches from the
same source image avoid repeated PIL image/mask reads while preserving original files.
NOTE

runs=(
  "deep_supervision|configs/stage2_p0_a40_e50_bs48_pw12_deepsup.yaml|outputs/experiments/p0_a40_e50_bs48_pw12_deepsup_20260509|deep supervision auxiliary segmentation heads on the pos_weight=12 baseline"
  "boundary_aux|configs/stage2_p0_a40_e50_bs48_pw12_boundary.yaml|outputs/experiments/p0_a40_e50_bs48_pw12_boundary_20260509|boundary auxiliary loss on the pos_weight=12 baseline"
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

This is a third-batch experiment. It keeps the best pos_weight=12 baseline and
changes one training mechanism so the result can be compared directly against
the completed pos_weight ablation.

- Config: ${config}
- Runner: scripts/run_p0_a40_stage2_e50_bs48.sh
- Output root: ${experiment_dir}
- Fold logs: ${experiment_dir}/logs/fold{0..3}_stage2.log
- Pooled OOF log: ${experiment_dir}/logs/oof_postprocess.log
- Pooled OOF result: ${experiment_dir}/stage2/oof_global_postprocess.json
NOTE
}

for item in "${runs[@]}"; do
  IFS="|" read -r name config experiment_dir purpose <<< "${item}"
  write_experiment_note "${name}" "${config}" "${experiment_dir}" "${purpose}"

  if [ -f "${experiment_dir}/stage2/oof_global_postprocess.json" ] && [ "${FORCE_RERUN:-0}" != "1" ]; then
    echo "third_batch ${name} already has pooled OOF; skip $(date -Is)"
    continue
  fi

  echo "third_batch ${name} started $(date -Is)"
  EXPERIMENT_DIR="${experiment_dir}" bash scripts/run_p0_a40_stage2_e50_bs48.sh "${config}"
  echo "third_batch ${name} finished $(date -Is)"
done
