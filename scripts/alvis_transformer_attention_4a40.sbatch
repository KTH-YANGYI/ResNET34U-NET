#!/usr/bin/env bash
#SBATCH --account=NAISS2026-3-280
#SBATCH --partition=alvis
#SBATCH --job-name=unet_attn3_a40
#SBATCH --gpus-per-node=A40:4
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x_%j.out
#SBATCH --error=slurm-%x_%j.err

set -euo pipefail

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/mimer/NOBACKUP/groups/smart-rail/Yi Yang/CV_contact_wire/UNET_two_stage/transformer_attention_milestone_20260510}"
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch-bundle}"
EXTRA_MODULES="${EXTRA_MODULES:-}"

module purge
module load "${PYTORCH_MODULE}"
if [[ -n "${EXTRA_MODULES}" ]]; then
  # shellcheck disable=SC2086
  module load ${EXTRA_MODULES}
fi

python -c "import sys; print('python', sys.version)"
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu_count', torch.cuda.device_count())"

bash "${EXPERIMENT_ROOT}/run_all_3_experiments.sh" "${EXPERIMENT_ROOT}"
