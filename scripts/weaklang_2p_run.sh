#!/bin/bash
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
HARM=private_artifacts/multijail_v1/harm_grid.jsonl; ORDER=private_artifacts/multijail_v1/resource_order.json
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
gen_judge() { tag="$1"; ref="$2"; util="$3"; be="$4"
  [ -f results/mj_weaklang_20260906/$tag.json ] && { echo "have $tag"; return; }
  if [ "$be" = triton ]; then export VLLM_ATTENTION_BACKEND=TRITON_ATTN; unset VLLM_USE_FLASHINFER_SAMPLER; else export VLLM_USE_FLASHINFER_SAMPLER=0; unset VLLM_ATTENTION_BACKEND; fi
  echo "[$(date +%H:%M:%S)] GEN $tag"
  for t in 1 2 3; do $VP/python scripts/weaklang_twophase.py --phase gen --target "$ref" --tag $tag --order $ORDER --harm $HARM --n 4 --n-items 64 --util $util --max-model-len 4096 && break; sleep 10; done
  echo "[$(date +%H:%M:%S)] JUDGE $tag"
  for t in 1 2 3; do $VP/python scripts/weaklang_twophase.py --phase judge --tag $tag --order $ORDER --harm $HARM && break; sleep 10; done
}
gen_judge qwen25_7b Qwen/Qwen2.5-7B-Instruct 0.35 flash
gen_judge llama31_8b_it meta-llama/Llama-3.1-8B-Instruct 0.35 flash
gen_judge gemma2_9b_it google/gemma-2-9b-it 0.40 triton
echo WEAKLANG_2P_DONE
