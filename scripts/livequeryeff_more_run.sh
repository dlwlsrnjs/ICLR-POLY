#!/bin/bash
# LIVE selector on 2 more open held-out endpoints (reviewer's 5->6-7 lever: confirm the replay-based
# query-cost holds under real batch noise + adaptive stopping). Same driver as mistral7b; each probe is
# a real online query, LOTO prior (exclude-target-tag), GPU-guarded (needs ~52GB: target util 0.25 +
# resident recon+guard judges on one card), idempotent. New file.
set +e
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ENFORCE_EAGER=1 VLLM_USE_FLASHINFER_SAMPLER=0
OUT=results/live_query_efficiency_20260909; NEEDG=52000
pick_gpu(){ need=$1; for w in $(seq 1 5760); do
  g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -v n=$need '$2>=n{gsub(/,/,"",$1);print $1;exit}')
  [ -n "$g" ] && { echo "$g"; return 0; }; echo "[wait $(date +%H:%M:%S)] no GPU ${need}MB" >&2; sleep 30; done; }
run_live(){ tag="$1"; ref="$2"; excl="$3"
  [ -f "$OUT/$tag.json" ] && { echo "OK $OUT/$tag.json"; return 0; }
  for t in 1 2 3; do g=$(pick_gpu "$NEEDG")
    echo "[$(date +%H:%M:%S)] LIVE-QUERYEFF GPU$g -> $OUT/$tag.json (try $t)"
    CUDA_VISIBLE_DEVICES=$g $VP/python scripts/live_query_efficiency.py \
      --target "$ref" --tag "$tag" --exclude-target-tag "$excl" --trust-remote-code \
      --util 0.25 --max-model-len 6144 --budget 12 --calib 24 --outdir $OUT && [ -f "$OUT/$tag.json" ] && return 0
    sleep 20; done; echo "FAILED $tag"; return 1; }
run_live phi35_mini microsoft/Phi-3.5-mini-instruct phi35_mini
run_live glm4_9b    THUDM/glm-4-9b-chat-hf          glm4_9b
echo "LIVE_QUERYEFF_MORE_DONE $(date +%H:%M:%S)"
