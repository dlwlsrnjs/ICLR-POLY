#!/bin/bash
set +e
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ENFORCE_EAGER=1 VLLM_USE_FLASHINFER_SAMPLER=0
OUT=results/live_query_efficiency_20260909
one(){ dev="$1"; tag="$2"; ref="$3"
  [ -f "$OUT/$tag.json" ] && { echo "OK $tag"; return; }
  echo "[$(date +%H:%M:%S)] LIVE $tag on GPU$dev"
  CUDA_VISIBLE_DEVICES=$dev $VP/python scripts/live_query_efficiency.py --target "$ref" --tag "$tag" \
    --exclude-target-tag "$tag" --trust-remote-code --util 0.30 --max-model-len 6144 --budget 12 --calib 24 --outdir $OUT
  echo "[$(date +%H:%M:%S)] DONE_ONE $tag"
}
# GPU0 chain: glm4 then olmo2 ; GPU1 chain: falcon
( one 0 glm4_9b THUDM/glm-4-9b-chat-hf; one 0 olmo2_7b allenai/OLMo-2-1124-7B-Instruct ) &
( one 1 falcon3_7b tiiuae/Falcon3-7B-Instruct ) &
wait
echo "LIVE_QUERYEFF_PINNED_DONE $(date +%H:%M:%S)"
