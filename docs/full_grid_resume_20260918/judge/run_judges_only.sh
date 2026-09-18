#!/usr/bin/env bash
set -uo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/../.." && pwd)
RUN_ROOT=${POLY_RUN_ROOT:-$ROOT/target_prior_only_20260916/full_grid_5models_bf16}
DATA_DIR=${POLY_DATA_DIR:-$ROOT/polyjigsaw_grid_collect_runtime/data}
PYTHON_BIN=${POLY_PYTHON:-/tmp/claude-1002/-home-ubuntu-342-jinkwon-poly/850b1f38-b028-47de-ada7-7e63e7e380a0/scratchpad/venv-vllm085/bin/python}
MODELS=${POLY_JUDGE_INCLUDE_MODELS:-"qwen25_14b phi3_medium_14b"}
DATASETS=${POLY_JUDGE_ONLY_DATASETS:-"mj"}
RETRY_DELAY=${POLY_JUDGE_RETRY_DELAY:-30}
RAW_DIR=$RUN_ROOT/raw
JUDGED_DIR=$RUN_ROOT/judged
STATUS_DIR=$RUN_ROOT/status
STATUS_FILE=$STATUS_DIR/pipeline.status

mkdir -p "$JUDGED_DIR" "$STATUS_DIR"

child=""
write_status() {
  local message=$1 tmp
  tmp=$STATUS_FILE.tmp.$$
  printf '%s %s\n' "$message" "$(date -Is)" >"$tmp"
  mv "$tmp" "$STATUS_FILE"
}
cleanup() {
  if [[ -n "$child" ]]; then
    kill -TERM "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
}
on_signal() {
  trap - INT TERM HUP
  cleanup
  write_status "PAUSED generation; judgment-only runner interrupted"
  exit 130
}
trap on_signal INT TERM HUP

dataset_complete() {
  local dataset=$1 model raw_count judged_count meta_count
  for model in $MODELS; do
    raw_count=$(find "$RAW_DIR/$dataset" -maxdepth 1 -type f -name "${model}_${dataset}__*.jsonl" 2>/dev/null | wc -l)
    judged_count=$(find "$JUDGED_DIR/$dataset" -maxdepth 1 -type f -name "${model}_${dataset}__*.jsonl" 2>/dev/null | wc -l)
    meta_count=$(find "$JUDGED_DIR/$dataset" -maxdepth 1 -type f -name "${model}_${dataset}__*.jsonl.meta.json" 2>/dev/null | wc -l)
    ((raw_count > 0 && judged_count == raw_count && meta_count == raw_count)) || return 1
  done
}

for dataset in $DATASETS; do
  while ! dataset_complete "$dataset"; do
    write_status "RUNNING judgment-only $dataset; generation paused"
    POLY_PYTHON="$PYTHON_BIN" \
    POLY_JUDGE_INCLUDE_MODELS="$MODELS" \
      "$HERE/run_two_gpu.sh" "$dataset" "$RAW_DIR/$dataset" "$JUDGED_DIR/$dataset" "$DATA_DIR" &
    child=$!
    wait "$child"
    rc=$?
    child=""
    if dataset_complete "$dataset"; then
      break
    fi
    write_status "RETRY judgment-only $dataset rc=$rc; generation paused"
    sleep "$RETRY_DELAY"
  done
done

write_status "PAUSED generation; requested judgments complete"
