#!/bin/bash
# QUEUED live query-efficiency (reviewer score-changer): run the arm selector against a LIVE
# endpoint and report queries-to-threshold for every search strategy, with each probe a real
# generate -> recon-gate -> safety-judge (no stored matrix, no synthetic probe noise).
# Runs AFTER the held-out expansion finishes so the two jobs never fight for the one freed card,
# then GPU-guarded + idempotent. Does NOT touch paper tables; writes results/live_query_efficiency_*/ (0600).
set +e
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ENFORCE_EAGER=1
export VLLM_USE_FLASHINFER_SAMPLER=0

pick_gpu(){ need=$1; for w in $(seq 1 5760); do
  while read idx free; do [ "$free" -ge "$need" ] && { echo "$idx"; return 0; }; done \
    < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  echo "[wait $(date +%H:%M:%S)] no GPU with ${need}MB free" >&2; sleep 30; done; return 1; }

# --- wait for the held-out expansion to finish (avoid contending for the single freed card) ---
EXP_LOG=results/heldout_expand.log
echo "[$(date +%H:%M:%S)] live-queryeff queued; waiting for held-out expansion to finish"
while :; do
  grep -q EXPAND_GEN_DONE "$EXP_LOG" 2>/dev/null && { echo "[$(date +%H:%M:%S)] expansion done-marker seen"; break; }
  pgrep -f heldout_expand.sh >/dev/null 2>&1 || { echo "[$(date +%H:%M:%S)] expansion process gone; proceeding"; break; }
  sleep 60
done

OUT=results/live_query_efficiency_20260909
TAG=mistral7b
if [ -f "$OUT/$TAG.json" ]; then echo "OK $OUT/$TAG.json (idempotent skip)"; echo "LIVE_QUERYEFF_DONE $(date +%H:%M:%S)"; exit 0; fi
NEEDG=52000   # 7B target (util 0.25 ~= 20GB) + resident recon+guard judges (~31GB), both on one card
g=$(pick_gpu "$NEEDG")||{ echo "no GPU"; exit 1; }
echo "[$(date +%H:%M:%S)] LIVE-QUERYEFF GPU$g -> $OUT/$TAG.json"
for t in 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g $VP/python scripts/live_query_efficiency.py \
    --target mistralai/Mistral-7B-Instruct-v0.3 --tag $TAG \
    --exclude-target-tag mistral7b \
    --util 0.25 --max-model-len 6144 --budget 12 --calib 24 \
    --outdir $OUT && [ -f "$OUT/$TAG.json" ] && break
  echo "[$(date +%H:%M:%S)] retry $t"; sleep 20; g=$(pick_gpu "$NEEDG")||exit 1
done
[ -f "$OUT/$TAG.json" ] && echo "LIVE_QUERYEFF_DONE $(date +%H:%M:%S)" || echo "LIVE_QUERYEFF_FAILED $(date +%H:%M:%S)"
echo "NEXT: review $OUT/$TAG.json (queries_to_threshold) and fold into the query-efficiency discussion."
