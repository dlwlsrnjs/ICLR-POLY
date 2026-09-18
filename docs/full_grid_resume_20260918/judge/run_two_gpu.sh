#!/usr/bin/env bash
set -euo pipefail

if (($# != 4)); then
  echo "usage: $0 <dataset: lg|mj> <input-dir> <output-dir> <data-dir>" >&2
  exit 2
fi

DATASET=$1
INPUT_DIR=$2
OUTPUT_DIR=$3
DATA_DIR=$4
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${POLY_PYTHON:-/tmp/claude-1002/-home-ubuntu-342-jinkwon-poly/850b1f38-b028-47de-ada7-7e63e7e380a0/scratchpad/venv-vllm085/bin/python}
BATCH_SIZE=${POLY_JUDGE_BATCH_SIZE:-128}
INCLUDE_MODELS=${POLY_JUDGE_INCLUDE_MODELS:-}

extra_args=()
for model in $INCLUDE_MODELS; do
  extra_args+=(--include-prefix "${model}_${DATASET}__")
done

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUTPUT_DIR/logs"

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" "$SCRIPT_DIR/judge_existing.py" \
  --dataset "$DATASET" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --data-dir "$DATA_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-shards 2 \
  --shard-index 0 \
  "${extra_args[@]}" \
  >"$OUTPUT_DIR/logs/judge-${DATASET}-gpu0.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" "$SCRIPT_DIR/judge_existing.py" \
  --dataset "$DATASET" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --data-dir "$DATA_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-shards 2 \
  --shard-index 1 \
  "${extra_args[@]}" \
  >"$OUTPUT_DIR/logs/judge-${DATASET}-gpu1.log" 2>&1 &
PID1=$!

cleanup() {
  kill -TERM "$PID0" "$PID1" 2>/dev/null || true
  wait "$PID0" "$PID1" 2>/dev/null || true
}
on_signal() {
  trap - INT TERM HUP
  cleanup
  exit 130
}
trap on_signal INT TERM HUP

set +e
wait -n "$PID0" "$PID1"
RC=$?
if ((RC != 0)); then
  cleanup
  exit "$RC"
fi
wait "$PID0"
RC0=$?
wait "$PID1"
RC1=$?
((RC0 == 0)) || exit "$RC0"
exit "$RC1"
