#!/usr/bin/env bash
set -uo pipefail
if [[ -f /home/ubuntu/342/jinkwon/poly/target_prior_only_20260916/full_grid_5models_bf16/STOP_REQUESTED ]]; then exit 0; fi

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/../.." && pwd)
COLLECTOR=${POLY_COLLECTOR:-$ROOT/polyjigsaw_grid_collect_runtime/collect_grid.py}
DATA_DIR=${POLY_DATA_DIR:-$ROOT/polyjigsaw_grid_collect_runtime/data}
PYTHON_BIN=${POLY_PYTHON:-/tmp/claude-1002/-home-ubuntu-342-jinkwon-poly/850b1f38-b028-47de-ada7-7e63e7e380a0/scratchpad/venv-vllm085/bin/python}
RUN_ROOT=${POLY_RUN_ROOT:-$ROOT/target_prior_only_20260916/full_grid_5models_bf16}
export HF_HOME=${HF_HOME:-/home/ubuntu/342/jinkwon/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
RAW_DIR=$RUN_ROOT/raw
JUDGED_DIR=$RUN_ROOT/judged
LOG_DIR=$RUN_ROOT/logs
STATUS_DIR=$RUN_ROOT/status

GPU0_MODELS=${POLY_GPU0_MODELS:-"qwen25_14b qwen25_32b"}
GPU1_MODELS=${POLY_GPU1_MODELS:-"phi3_medium_14b mistral24b gemma2_27b"}
DATASETS=${POLY_DATASETS:-"lg mj"}
DATASET_RETRIES=${POLY_DATASET_RETRIES:-3}
RETRY_DELAY=${POLY_RETRY_DELAY:-30}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "$RAW_DIR" "$JUDGED_DIR" "$LOG_DIR" "$STATUS_DIR"

children=()
kill_tree() {
  local parent=$1 child
  while read -r child; do
    [[ -n "$child" ]] && kill_tree "$child"
  done < <(pgrep -P "$parent" 2>/dev/null || true)
  kill -TERM "$parent" 2>/dev/null || true
}
cleanup() {
  local pid
  for pid in "${children[@]:-}"; do
    kill_tree "$pid"
  done
  for pid in "${children[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
}
on_signal() {
  trap - INT TERM HUP
  cleanup
  exit 130
}
trap on_signal INT TERM HUP

run_generation_worker() {
  local gpu=$1
  shift
  local tag ds one_rc attempt log
  for tag in "$@"; do
    for ds in $DATASETS; do
      log="$LOG_DIR/generate_${tag}_${ds}.log"
      : >"$log"
      attempt=1
      while ((attempt <= DATASET_RETRIES)); do
        printf 'RUNNING attempt=%s/%s %s/%s GPU%s %s\n' "$attempt" "$DATASET_RETRIES" "$tag" "$ds" "$gpu" "$(date -Is)" >"$STATUS_DIR/${tag}_${ds}.status"
        if CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" "$COLLECTOR" \
            --raw-only --tag "$tag" --ds "$ds" --data "$DATA_DIR" --out "$RAW_DIR" \
            >>"$log" 2>&1; then
          printf 'DONE %s/%s GPU%s %s\n' "$tag" "$ds" "$gpu" "$(date -Is)" >"$STATUS_DIR/${tag}_${ds}.status"
          break
        else
          one_rc=$?
          if ((attempt >= DATASET_RETRIES)); then
            printf 'FAILED(%s) attempts=%s %s/%s GPU%s %s\n' "$one_rc" "$attempt" "$tag" "$ds" "$gpu" "$(date -Is)" >"$STATUS_DIR/${tag}_${ds}.status"
            return 1
          fi
          printf 'RETRYING(%s) attempt=%s/%s %s/%s GPU%s %s\n' "$one_rc" "$attempt" "$DATASET_RETRIES" "$tag" "$ds" "$gpu" "$(date -Is)" >"$STATUS_DIR/${tag}_${ds}.status"
          sleep $((RETRY_DELAY * attempt))
          attempt=$((attempt + 1))
        fi
      done
    done
  done
  return 0
}

read -r -a gpu0_models <<<"$GPU0_MODELS"
read -r -a gpu1_models <<<"$GPU1_MODELS"
run_generation_worker 0 "${gpu0_models[@]}" &
children+=("$!")
run_generation_worker 1 "${gpu1_models[@]}" &
children+=("$!")

generation_rc=0
for pid in "${children[@]}"; do
  wait "$pid" || generation_rc=1
done
children=()
if ((generation_rc != 0)); then
  printf 'FAILED generation %s\n' "$(date -Is)" >"$STATUS_DIR/pipeline.status"
  exit "$generation_rc"
fi


judge_rc=0
for ds in $DATASETS; do
  if find "$RAW_DIR/$ds" -maxdepth 1 -type f -name '*.jsonl' -print -quit 2>/dev/null | grep -q .; then
    POLY_PYTHON="$PYTHON_BIN" "$HERE/run_two_gpu.sh" \
        "$ds" "$RAW_DIR/$ds" "$JUDGED_DIR/$ds" "$DATA_DIR" &
    children=("$!")
    if ! wait "${children[0]}"; then
      judge_rc=1
      children=()
      break
    fi
    children=()
  fi
done

if ((judge_rc != 0)); then
  printf 'FAILED judging %s\n' "$(date -Is)" >"$STATUS_DIR/pipeline.status"
  exit "$judge_rc"
fi
printf 'DONE %s\n' "$(date -Is)" >"$STATUS_DIR/pipeline.status"
exit 0
