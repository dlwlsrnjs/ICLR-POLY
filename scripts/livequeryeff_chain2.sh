#!/bin/bash
# Chain after the heavy pair (Qwen32B+Mistral24B): 3 more LIVE endpoints for vendor breadth + a Google
# heavy. gemma-2-27b (heavy) -> split (target GPU0 util0.80 + TRITON attn, judges GPU1); then the two
# Meta Llamas (small) in parallel one card each. All LOTO (targets absent from the prior set anyway).
set +e
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ENFORCE_EAGER=1 VLLM_USE_FLASHINFER_SAMPLER=0
OUT=results/live_query_efficiency_20260909
echo "[$(date +%H:%M:%S)] chain2 waiting for heavy pair to finish"
for i in $(seq 1 480); do
  grep -q "LIVE_QUERYEFF_HEAVY_DONE" results/livequeryeff_heavy.log 2>/dev/null && break
  pgrep -f livequeryeff_heavy.sh >/dev/null || break
  sleep 30
done
echo "[$(date +%H:%M:%S)] chain2 starting"
# gemma-2-27b heavy: split (target GPU0, judges GPU1) + TRITON attention (gemma-2 crashes default)
run_heavy(){ tag="$1"; ref="$2"; util="$3"
  [ -f "$OUT/$tag.json" ] && { echo "OK $tag"; return; }
  echo "[$(date +%H:%M:%S)] LIVE-HEAVY $tag (GPU0 target util$util + TRITON, GPU1 judges)"
  CUDA_VISIBLE_DEVICES=0,1 VLLM_ATTENTION_BACKEND=TRITON_ATTN $VP/python scripts/live_query_efficiency.py \
    --target "$ref" --tag "$tag" --exclude-target-tag "$tag" --trust-remote-code \
    --util "$util" --max-model-len 3072 --judge-device cuda:1 --budget 12 --calib 24 --outdir $OUT; }
run_small(){ dev="$1"; tag="$2"; ref="$3"
  [ -f "$OUT/$tag.json" ] && { echo "OK $tag"; return; }
  echo "[$(date +%H:%M:%S)] LIVE $tag on GPU$dev"
  CUDA_VISIBLE_DEVICES=$dev $VP/python scripts/live_query_efficiency.py \
    --target "$ref" --tag "$tag" --exclude-target-tag "$tag" --trust-remote-code \
    --util 0.30 --max-model-len 6144 --judge-device cuda:0 --budget 12 --calib 24 --outdir $OUT; }
run_heavy gemma2_27b google/gemma-2-27b-it 0.80
# two small Meta Llamas in parallel (one card each)
( run_small 0 llama31_8b_it meta-llama/Llama-3.1-8B-Instruct ) &
( run_small 1 llama32_3b_it meta-llama/Llama-3.2-3B-Instruct ) &
wait
echo "LIVE_QUERYEFF_CHAIN2_DONE $(date +%H:%M:%S)"
