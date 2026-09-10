#!/bin/bash
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
HARM=private_artifacts/multijail_v1/harm_grid.jsonl; ORDER=private_artifacts/multijail_v1/resource_order.json
OUT=results/mj_triple_20260906
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${GPU:-1}
gen_judge() { tag="$1"; ref="$2"; util="$3"; be="$4"; al="$5"
  [ -f $OUT/$tag.json ] && { echo "have $tag"; return; }
  if [ "$be" = triton ]; then export VLLM_ATTENTION_BACKEND=TRITON_ATTN; unset VLLM_USE_FLASHINFER_SAMPLER; \
  else export VLLM_USE_FLASHINFER_SAMPLER=0; unset VLLM_ATTENTION_BACKEND; fi
  echo "[$(date +%H:%M:%S)] GEN $tag (answer=$al)"
  for t in 1 2 3; do $VP/python scripts/triple_combo_eval.py --phase gen --target "$ref" --tag $tag \
     --order $ORDER --harm $HARM --answer-lang "$al" --n 4 --n-items 64 --util $util --max-model-len 4096 --outdir $OUT && break; sleep 10; done
  echo "[$(date +%H:%M:%S)] JUDGE $tag"
  for t in 1 2 3; do $VP/python scripts/triple_combo_eval.py --phase judge --tag $tag \
     --order $ORDER --harm $HARM --answer-lang "$al" --outdir $OUT && break; sleep 10; done
}
gen_judge qwen25_7b     Qwen/Qwen2.5-7B-Instruct    0.35 flash  Swahili
gen_judge llama31_8b_it meta-llama/Llama-3.1-8B-Instruct 0.35 flash  Swahili
gen_judge gemma2_9b_it  google/gemma-2-9b-it        0.40 triton Swahili
echo TRIPLE_2P_DONE
