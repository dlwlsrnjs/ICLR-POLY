#!/bin/bash
set +e
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ENFORCE_EAGER=1 VLLM_USE_FLASHINFER_SAMPLER=0
OUT=results/live_query_efficiency_20260909; NEEDG=52000
pick_gpu(){ need=$1; for w in $(seq 1 5760); do
  g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -v n=$need '$2>=n{gsub(/,/,"",$1);print $1;exit}')
  [ -n "$g" ] && { echo "$g"; return 0; }; echo "[wait $(date +%H:%M:%S)] no GPU ${need}MB" >&2; sleep 30; done; }
run_live(){ tag="$1"; ref="$2"
  [ -f "$OUT/$tag.json" ] && { echo "OK $tag"; return 0; }
  for t in 1 2 3; do g=$(pick_gpu "$NEEDG"); echo "[$(date +%H:%M:%S)] LIVE $tag GPU$g (try $t)"
    CUDA_VISIBLE_DEVICES=$g $VP/python scripts/live_query_efficiency.py --target "$ref" --tag "$tag" \
      --exclude-target-tag "$tag" --trust-remote-code --util 0.25 --max-model-len 6144 --budget 12 --calib 24 --outdir $OUT \
      && [ -f "$OUT/$tag.json" ] && return 0; sleep 20; done; }
run_live falcon3_7b tiiuae/Falcon3-7B-Instruct
run_live olmo2_7b   allenai/OLMo-2-1124-7B-Instruct
echo "LIVE_QUERYEFF_EXTRA_DONE $(date +%H:%M:%S)"
