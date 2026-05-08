#!/usr/bin/env bash
#SBATCH --account=CHANGE_ME
#SBATCH --partition=alvis
#SBATCH --job-name=unet_a100
#SBATCH --gpus-per-node=A100:1
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm-%x_%j.out
#SBATCH --error=slurm-%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch-bundle}"
EXTRA_MODULES="${EXTRA_MODULES:-}"
GPUS="${CUDA_FOLD_GPUS:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$PROJECT_DIR"
mkdir -p outputs/logs

module purge
module load "$PYTORCH_MODULE"
if [[ -n "$EXTRA_MODULES" ]]; then
  # shellcheck disable=SC2086
  module load $EXTRA_MODULES
fi

python -c "import sys; print('python', sys.version)"
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu_count', torch.cuda.device_count())"
python scripts/check_environment.py --strict

bash scripts/train_all_folds.sh --gpus "$GPUS" $EXTRA_ARGS
