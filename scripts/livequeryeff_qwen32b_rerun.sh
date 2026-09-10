#!/bin/bash
# Re-run qwen25_32b LIVE (OOM'd at max-len 3072/util0.85 -> KV starved). qwen25_32b is a PANEL model,
# i.e. exactly where the offline "ours 3 vs uninformed 8-9 (~3x)" query-savings claim should hold, so
# this is the decisive live test. Split: 32B target GPU0 (util0.92, max-len 2048 to leave KV room),
# resident judges GPU1. Chained AFTER chain2 (both cards busy with gemma/llamas). Idempotent.
set +e
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ENFORCE_EAGER=1 VLLM_USE_FLASHINFER_SAMPLER=0
OUT=results/live_query_efficiency_20260909
echo "[$(date +%H:%M:%S)] qwen32b-rerun waiting for chain2 (gemma/llamas) to finish"
for i in $(seq 1 720); do
  grep -q "LIVE_QUERYEFF_CHAIN2_DONE" results/livequeryeff_chain2.log 2>/dev/null && break
  pgrep -f livequeryeff_chain2.sh >/dev/null || break
  sleep 30
done
echo "[$(date +%H:%M:%S)] qwen32b-rerun starting"
[ -f "$OUT/qwen25_32b.json" ] && { echo "OK already"; echo "QWEN32B_RERUN_DONE"; exit 0; }
CUDA_VISIBLE_DEVICES=0,1 $VP/python scripts/live_query_efficiency.py \
  --target Qwen/Qwen2.5-32B-Instruct --tag qwen25_32b --exclude-target-tag qwen25_32b --trust-remote-code \
  --util 0.92 --max-model-len 2048 --judge-device cuda:1 --budget 12 --calib 24 --outdir $OUT
echo "QWEN32B_RERUN_DONE $(date +%H:%M:%S)"
