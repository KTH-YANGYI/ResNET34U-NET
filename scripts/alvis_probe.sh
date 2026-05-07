#!/usr/bin/env bash
set -uo pipefail

LOG_DIR="${LOG_DIR:-outputs/logs}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/alvis_env_probe.txt}"
mkdir -p "$LOG_DIR"

{
  echo "===== system ====="
  date
  hostname
  pwd
  echo "USER=${USER:-}"
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

  echo
  echo "===== modules currently loaded ====="
  module list 2>&1 || true

  echo
  echo "===== useful module searches ====="
  for name in PyTorch-bundle PyTorch torchvision Python SciPy-bundle Pillow PyYAML tqdm; do
    echo
    echo "----- module spider $name -----"
    module spider "$name" 2>&1 | sed -n '1,80p' || true
  done

  echo
  echo "===== executables ====="
  which python || true
  python --version 2>&1 || true
  which pip || true
  pip --version 2>&1 || true
  which nvidia-smi || true
  nvidia-smi || true

  echo
  echo "===== selected python imports ====="
  python - <<'PY' || true
import importlib
import importlib.metadata
import os
import sys

print("python:", sys.version.replace("\n", " "))
print("executable:", sys.executable)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", ""))

modules = [
    "torch",
    "torchvision",
    "numpy",
    "PIL",
    "yaml",
    "tqdm",
    "scipy",
]

for name in modules:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"OK      {name:12s} {version}")
    except Exception as exc:
        print(f"MISSING {name:12s} {exc}")

try:
    import torch

    print("torch cuda available:", torch.cuda.is_available())
    print("torch cuda device_count:", torch.cuda.device_count())
    for index in range(torch.cuda.device_count()):
        print(f"torch cuda device {index}:", torch.cuda.get_device_name(index))
except Exception as exc:
    print("torch cuda check failed:", exc)

print("installed distributions:")
for dist in sorted(importlib.metadata.distributions(), key=lambda item: item.metadata["Name"].lower()):
    name = dist.metadata["Name"]
    version = dist.version
    print(f"  {name}=={version}")
PY

  echo
  echo "===== pip list ====="
  python -m pip list 2>&1 || true
} | tee "$LOG_PATH"

echo
echo "Probe written to: $LOG_PATH"
