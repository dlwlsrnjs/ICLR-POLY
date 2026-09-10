#!/bin/bash
# Row-level dual-judge diagnostic (translate-then-judge) on the saved cross-check generations.
# GPU-guarded like judge_crosscheck_run.sh; idempotent (skips finished phases).
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
OUT=results/judge_crosscheck_20260907
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_USE_FLASHINFER_SAMPLER=0
unset VLLM_ATTENTION_BACKEND
pick() { need=$1; while :; do
  g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -v n=$need '$2>=n{print $1; exit}' | tr -d ',')
  [ -n "$g" ] && { echo $g; return 0; }; sleep 45; done; }
if [ ! -f $OUT/_cc_rows_translated.json ]; then
  g=$(pick 12000); echo "[$(date +%H:%M:%S)] TRANSLATE (NLLB) on GPU$g"
  CUDA_VISIBLE_DEVICES=$g $VP/python scripts/judge_crosscheck_rows.py --phase translate --outdir $OUT || true
  [ -f $OUT/_cc_rows_translated.json ] || { echo "translate failed"; exit 1; }
else echo "translate OK"; fi
tries=0
while [ ! -f $OUT/crosscheck_rows_summary.json ] && [ $tries -lt 3 ]; do
  tries=$((tries+1)); g=$(pick 52000); echo "[$(date +%H:%M:%S)] JUDGE rows (Qwen3Guard x2 + MD-Judge x2) on GPU$g try $tries"
  CUDA_VISIBLE_DEVICES=$g $VP/python scripts/judge_crosscheck_rows.py --phase judge --outdir $OUT || true
  [ -f $OUT/crosscheck_rows_summary.json ] || { echo "[$(date +%H:%M:%S)] judge failed, waiting"; sleep 60; }
done
[ -f $OUT/crosscheck_rows_summary.json ] && echo JUDGE_CROSSCHECK_ROWS_DONE || echo JUDGE_CROSSCHECK_ROWS_FAILED
