#!/usr/bin/env bash
# PolyJigsaw grid collection driver. Resumable. Splits work across servers via MODELS / DATASETS env.
#   MODELS="qwen25_7b gemma2_9b_it"  DATASETS="lg mj"  bash run.sh
# Big targets (model_map big=true) auto-place judges on cuda:1; others co-locate on cuda:0.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
OUT="${OUT:-$HERE/full_grid}"; LOG="$OUT/logs"; mkdir -p "$LOG"
ALL="qwen25_3b qwen25_7b qwen25_14b llama32_3b_it llama31_8b_it gemma2_2b_it gemma2_9b_it glm4_9b mistral7b falcon3_3b falcon3_7b falcon3_10b phi35_mini phi3_medium_14b gemma2_27b qwen25_32b mistral24b"
MODELS="${MODELS:-$ALL}"; DATASETS="${DATASETS:-lg mj}"
big(){ $PY -c "import json,sys;print('1' if json.load(open('$HERE/data/model_map.json'))['$1']['big'] else '0')"; }
narms(){ ls "$OUT/$2/${1}_${2}__"*.jsonl 2>/dev/null | wc -l; }
for TAG in $MODELS; do
  for DS in $DATASETS; do
    [ "$(narms $TAG $DS)" -ge 160 ] && { echo "$TAG/$DS already complete"; continue; }
    if [ "$(big $TAG)" = "1" ]; then CVD=0,1; JDEV=cuda:1; else CVD=0; JDEV=cuda:0; fi
    echo "=== $TAG/$DS (jdev=$JDEV) $(date +%F_%H:%M:%S) ==="
    CUDA_VISIBLE_DEVICES=$CVD POLY_JUDGE_DEVICE=$JDEV \
      $PY "$HERE/collect_grid.py" --tag "$TAG" --ds "$DS" --data "$HERE/data" --out "$OUT" \
      >"$LOG/${TAG}_${DS}.log" 2>&1
    echo "  $TAG/$DS -> $(narms $TAG $DS)/160 arms  ($(grep -c DONE "$LOG/${TAG}_${DS}.log" 2>/dev/null) done marker)"
  done
done
echo "ALL REQUESTED DONE. grid files: $(find "$OUT" -name '*.jsonl' | wc -l)"
