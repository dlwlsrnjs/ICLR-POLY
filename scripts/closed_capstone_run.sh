#!/bin/bash
# CAPSTONE on the commercial closed panel: one FROZEN two-axis selector + FREE two-axis benign probe,
# transferred with no re-tuning to Gemini Flash + Claude Haiku + Claude Sonnet (extends the GPT-4o
# capstone to a multi-vendor closed panel). Targets are cloud APIs (keys from 0600 .secrets, via env
# only, never inline); judges LOCAL on GPU1 (~31GB; GPU0 is crowded). Per-model concurrency: the
# free-tier Gemini key 503s ("high demand") under load -> concurrency 1 for Gemini; Anthropic cc 4.
# Gemini Pro omitted (that key -> 429 for every Pro model). Idempotent per tag.
set +e
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export OPENAI_API_KEY="$(cat /home/ubuntu/342/jinkwon/.secrets/openai_api_key)"
export GEMINI_API_KEY="$(cat /home/ubuntu/342/jinkwon/.secrets/gemini_api_key)"
export ANTHROPIC_API_KEY="$(cat /home/ubuntu/342/jinkwon/.secrets/anthropic_api_key)"
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1   # judges on GPU1 (64GB free)
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
OUT=results/capstone_closed_20260909
cap(){ local be="$1" model="$2" tag="$3" cc="$4"
  [ -f "$OUT/$tag.json" ] && { echo "OK $OUT/$tag"; return; }
  echo "======== CAPSTONE $tag ($be:$model, cc=$cc) ========"
  $VP/python scripts/capstone_frozen_two_axis.py --backend "$be" --model "$model" --tag "$tag" \
     --concurrency "$cc" --calib 20 --budget 10 --max-tokens 512 --outdir "$OUT"
  echo "CAP_DONE $tag $(date +%H:%M:%S)"
}
cap anthropic claude-haiku-4-5-20251001  claude_haiku  4
cap anthropic claude-sonnet-4-5          claude_sonnet 4
cap gemini    gemini-flash-latest        gemini_flash  1
echo "CLOSED_CAPSTONE_ALL_DONE $(date +%H:%M:%S)"
