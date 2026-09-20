#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
  echo "usage: $0 MODEL_TAG GPU_ID [NUM_SHARDS SHARD_INDEX]" >&2
  exit 2
fi

TAG=$1
GPU_ID=$2
NUM_SHARDS=${3:-1}
SHARD_INDEX=${4:-0}
BUNDLE_DIR=$(cd "$(dirname "$0")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
QWEN32_MODEL=${QWEN32_MODEL:?Set QWEN32_MODEL to the local Qwen2.5-32B-Instruct revision directory}
RUN_DIR=$BUNDLE_DIR/models/$TAG/all

test -f "$RUN_DIR/responses.jsonl"
test -f "$RUN_DIR/manifest.json"
export CUDA_VISIBLE_DEVICES=$GPU_ID
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

ARGS=(
  --run "$RUN_DIR"
  --model "$QWEN32_MODEL"
  --batch "${QWEN32_BATCH:-16}"
)
if [ "$NUM_SHARDS" -gt 1 ]; then
  ARGS+=(--num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX")
fi

if [ "$GPU_ID" = "0" ]; then
  exec "$PYTHON_BIN" "/home/ubuntu/342/jinkwon/poly/transfer_qwen32_20260921/gpu0_batch_trial.py" "$BUNDLE_DIR/code/qwen32_dual_judge.py" "${ARGS[@]}"
fi
exec "$PYTHON_BIN" "$BUNDLE_DIR/code/qwen32_dual_judge.py" "${ARGS[@]}"
