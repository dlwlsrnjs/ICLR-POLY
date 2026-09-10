#!/bin/bash
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
HARM=private_artifacts/multijail_v1/harm_grid.jsonl
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 VLLM_ATTENTION_BACKEND=TRITON_ATTN
for a in 1 2 3 4 5 6; do
  [ -f results/mj_multijail_20260906/gemma2_9b_it.json ] && { echo "OK gemma"; break; }
  echo "[gemma try $a $(date +%H:%M:%S)]"
  $VP/python scripts/multijail_eval.py --target google/gemma-2-9b-it --tag gemma2_9b_it --harm $HARM --outdir results/mj_multijail_20260906 --n-items 64 --util 0.40 --max-model-len 4096
  sleep 10
done
echo DONEGEMMA
