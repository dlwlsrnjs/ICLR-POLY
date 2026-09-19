#!/usr/bin/env bash
set -euo pipefail
PACKAGE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$PACKAGE" rev-parse --show-toplevel)
PYTHON=${PYTHON:-python3}
OUT=${OUT:-$REPO/local_runs/willingness331_shared_en_ar}
GPU=${GPU:-0}
MODEL_TAG=${MODEL_TAG:-qwen25_7b}
case "${1:-help}" in
 verify)
  "$PYTHON" "$PACKAGE/verify_bundle.py"
  ;;
 qa)
  : "${QA_MODEL_PATH:?Set QA_MODEL_PATH to the pinned Qwen2.5-32B snapshot}"
  "$PYTHON" "$PACKAGE/verify_bundle.py"
  mkdir -p "$OUT/inputs"
  if [[ -e "$OUT/inputs/items.json" ]]; then cmp "$PACKAGE/data/items.json" "$OUT/inputs/items.json"; else cp "$PACKAGE/data/items.json" "$OUT/inputs/items.json"; fi
  "$PYTHON" "$PACKAGE/code/qa_arabic.py" --run "$OUT" --source "$PACKAGE/data/translations.jsonl" --model "$QA_MODEL_PATH" --gpu "$GPU"
  ;;
 prepare)
  for config in "$PACKAGE"/configs/*.json; do
   tag=$(basename "$config" .json)
   "$PYTHON" "$PACKAGE/code/prepare.py" --repo "$REPO" --config "$config" --items "$PACKAGE/data/items.json" --translations "$OUT/qa/accepted_translations.json" --out "$OUT/panel/$tag"
  done
  ;;
 collect)
  : "${MODEL_PATH:?Set MODEL_PATH to the selected model pinned snapshot directory}"
  "$PYTHON" "$PACKAGE/code/collect.py" --run "$OUT/panel/$MODEL_TAG" --gpu "$GPU" --model-path "$MODEL_PATH" --memory-utilization "${MEMORY_UTILIZATION:-.30}" --batch-size "${BATCH_SIZE:-8}" --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-2048}"
  ;;
 smoke)
  : "${MODEL_PATH:?Set MODEL_PATH to the pinned Qwen2.5-7B snapshot}"
  "$PYTHON" "$PACKAGE/code/prepare.py" --repo "$REPO" --config "$PACKAGE/smoke_inputs/config.json" --items "$PACKAGE/smoke_inputs/items.json" --translations "$PACKAGE/smoke_inputs/translations.json" --out "$OUT/smoke_qwen7b"
  "$PYTHON" "$PACKAGE/code/collect.py" --run "$OUT/smoke_qwen7b" --gpu "$GPU" --model-path "$MODEL_PATH" --memory-utilization "${MEMORY_UTILIZATION:-.30}" --batch-size 8 --max-num-batched-tokens 2048
  ;;
 *) echo 'Usage: bash run.sh verify|qa|prepare|collect|smoke'; exit 2 ;;
esac
