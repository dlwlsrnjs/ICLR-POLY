#!/bin/bash
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ATTENTION_BACKEND=TRITON_ATTN
[ -f results/benign_arms_20260907/gemma2_9b_it.json ] && { echo already; exit 0; }
for w in $(seq 1 240); do
  g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk '$2>=24000{print $1; exit}')
  [ -n "$g" ] && { echo "[$(date +%H:%M:%S)] GPU$g free -> gemma9b benign"
    CUDA_VISIBLE_DEVICES=$g $VP/python scripts/benign_arms_probe.py --target google/gemma-2-9b-it --tag gemma2_9b_it \
      --order results/lang_rank_20260905/resource_order.json --benign private_artifacts/panel_v2/benign_probe.jsonl \
      --answer-lang Norwegian --n-items 40 --util 0.30 --max-model-len 4096 --outdir results/benign_arms_20260907 && [ -f results/benign_arms_20260907/gemma2_9b_it.json ] && { echo DONE; exit 0; }; }
  sleep 30
done
