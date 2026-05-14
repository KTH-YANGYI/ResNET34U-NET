#!/usr/bin/env bash
#SBATCH -A NAISS2025-5-421
#SBATCH -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A100:4
#SBATCH -t 24:00:00
#SBATCH -J unet-attn-4a100
#SBATCH -o experiments/attention_20260511/comparison/logs/slurm_%j.out
#SBATCH -e experiments/attention_20260511/comparison/logs/slurm_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mimer/NOBACKUP/groups/smart-rail/Yi Yang/CV_contact_wire/UNET_two_stage}"
FOLDS="${FOLDS:-0 1 2 3}"
CUDA_FOLD_GPUS="${CUDA_FOLD_GPUS:-0,1,2,3}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${PROJECT_ROOT}"
mkdir -p experiments/attention_20260511/comparison/logs

module purge
module load PyTorch-bundle

export PROJECT_ROOT
export FOLDS
export CUDA_FOLD_GPUS
export PYTHON_BIN

bash experiments/attention_20260511/comparison/run_all_4gpu.sh
