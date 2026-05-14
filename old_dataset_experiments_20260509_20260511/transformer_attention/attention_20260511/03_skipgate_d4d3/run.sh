#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${EXPERIMENT_DIR}/../../.." && pwd)}"

bash "${PROJECT_ROOT}/experiments/attention_20260511/comparison/run_single_experiment.sh" \
  "${EXPERIMENT_DIR}" \
  "${EXPERIMENT_DIR}/config.yaml" \
  "0" \
  "Decoder skip attention gates at d4 and d3"
