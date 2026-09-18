#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/../.." && pwd)
PYTHON_BIN=${POLY_PYTHON:-/tmp/claude-1002/-home-ubuntu-342-jinkwon-poly/850b1f38-b028-47de-ada7-7e63e7e380a0/scratchpad/venv-vllm085/bin/python}
RUN_ROOT=${POLY_RUN_ROOT:-$ROOT/target_prior_only_20260916/full_grid_5models_bf16}
HF_RUNTIME=${POLY_HF_RUNTIME:-$ROOT/target_prior_only_20260916/.hf-sync-runtime}

export HF_HOME=${HF_HOME:-/home/ubuntu/342/jinkwon/hf_cache}
export PYTHONPATH="$HF_RUNTIME${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" "$HERE/sync_hf_bucket.py" \
  --run-root "$RUN_ROOT" \
  --bucket-id "${POLY_HF_BUCKET_ID:-jin-kwon/poly}" \
  --prefix "${POLY_HF_BUCKET_PREFIX:-PolyJigsaw/full_grid_20260917_v1/primary_large/incremental/full_grid_5models_bf16}" \
  --interval "${POLY_HF_SYNC_INTERVAL:-300}"
