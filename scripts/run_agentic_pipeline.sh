#!/usr/bin/env bash
# One target through the LIVE agentic probing pipeline:
#   adaptive probe (no judge) -> (A,C) -> structured warm-start -> online GP-UCB -> final judge.
# args: TAG MODEL GPU STRUCTURED_PRIOR [EXTRA_FLAGS]
set -uo pipefail
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
VPY=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python
TAG="$1"; MODEL="$2"; GPU="$3"; PRIOR="$4"; EXTRA="${5:-}"
OUT=results/agentic_pipeline_20260904
mkdir -p "$OUT"
echo "[$(date +%H:%M:%S)] AGENTIC-$TAG(gpu$GPU) start"
CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/online_live.py --target "$MODEL" --tag "$TAG" \
  --agentic --structured-prior "$PRIOR" --budget 10 --threshold 0.5 \
  --util 0.45 --outdir "$OUT" $EXTRA > "$OUT/$TAG.log" 2>&1 \
  && echo "[$(date +%H:%M:%S)] AGENTIC-$TAG done" || { echo "AGENTIC-$TAG FAILED"; tail -12 "$OUT/$TAG.log"; }
