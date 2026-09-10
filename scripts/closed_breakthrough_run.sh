#!/bin/bash
# After the closed capstone, run the EXHAUSTIVE 15-arm judged search (12 base + 3 untested
# low-resource-answer/hi_en arms) on the two robust Anthropic models, to empirically back the
# "no configuration in our space beats Claude 4.5" claim with real reconstruction-gated verified ASR
# per arm. Judges LOCAL on GPU1. Keys via env only. Idempotent per tag.
set +e
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export ANTHROPIC_API_KEY="$(cat /home/ubuntu/342/jinkwon/.secrets/anthropic_api_key)"
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
OUT=results/closed_breakthrough_20260909
echo "[$(date +%H:%M:%S)] starting breakthrough search"
bt(){ local model="$1" tag="$2"
  [ -f "$OUT/$tag.json" ] && { echo "OK $OUT/$tag"; return; }
  echo "======== BREAKTHROUGH $tag ($model) ========"
  $VP/python scripts/closed_breakthrough.py --backend anthropic --model "$model" --tag "$tag" \
     --calib 12 --concurrency 4 --max-tokens 512 --outdir "$OUT"
}
bt claude-haiku-4-5-20251001 claude_haiku
bt claude-sonnet-4-5         claude_sonnet
echo "CLOSED_BREAKTHROUGH_ALL_DONE $(date +%H:%M:%S)"
