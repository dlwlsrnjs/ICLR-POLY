#!/usr/bin/env bash
set -uo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/../.." && pwd)
RUN_ROOT=${POLY_RUN_ROOT:-$ROOT/target_prior_only_20260916/full_grid_5models_bf16}
DATA_DIR=${POLY_DATA_DIR:-$ROOT/polyjigsaw_grid_collect_runtime/data}
PYTHON_BIN=${POLY_PYTHON:-/tmp/claude-1002/-home-ubuntu-342-jinkwon-poly/850b1f38-b028-47de-ada7-7e63e7e380a0/scratchpad/venv-vllm085/bin/python}
PRIORITY_MODELS=${POLY_PRIORITY_JUDGE_MODELS:-"qwen25_14b phi3_medium_14b"}
DATASETS=${POLY_PRIORITY_JUDGE_DATASETS:-"lg mj"}
RAW_DIR=$RUN_ROOT/raw
JUDGED_DIR=$RUN_ROOT/judged
STATUS_DIR=$RUN_ROOT/status

mkdir -p "$JUDGED_DIR" "$STATUS_DIR"

child=""
cleanup() {
  if [[ -n "$child" ]]; then
    kill -TERM "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
}
on_signal() {
  trap - INT TERM HUP
  cleanup
  exit 130
}
trap on_signal INT TERM HUP

for ds in $DATASETS; do
  printf 'RUNNING priority judging %s %s\n' "$ds" "$(date -Is)" >"$STATUS_DIR/pipeline.status"
  POLY_PYTHON="$PYTHON_BIN" \
  POLY_JUDGE_INCLUDE_MODELS="$PRIORITY_MODELS" \
    "$HERE/run_two_gpu.sh" "$ds" "$RAW_DIR/$ds" "$JUDGED_DIR/$ds" "$DATA_DIR" &
  child=$!
  wait "$child"
  rc=$?
  if ((rc != 0)); then
    child=""
    printf 'FAILED priority judging %s rc=%s %s\n' "$ds" "$rc" "$(date -Is)" >"$STATUS_DIR/pipeline.status"
    exit "$rc"
  fi
  child=""
done

printf 'DONE priority judging; resuming generation %s\n' "$(date -Is)" >"$STATUS_DIR/pipeline.status"
exec "$HERE/run_five_models.sh"
