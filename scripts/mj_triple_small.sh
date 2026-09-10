#!/bin/bash
# Fill mj_triple for the 3 small models so the MJ bandit can match Lingua's 7-model set.
# 2-phase, small models fit easily. gemma uses TRITON_ATTN (softcapping).
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
HARM=private_artifacts/multijail_v1/harm_grid.jsonl
ORDER=private_artifacts/multijail_v1/resource_order.json
OUT=results/mj_triple_20260906
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${GPU:-1}
U=0.35; ML=4096
gen_judge() { tag="$1"; ref="$2"; be="$3"
  [ -f $OUT/$tag.json ] && { echo "OK $tag"; return; }
  if [ "$be" = triton ]; then export VLLM_ATTENTION_BACKEND=TRITON_ATTN; unset VLLM_USE_FLASHINFER_SAMPLER
  else export VLLM_USE_FLASHINFER_SAMPLER=0; unset VLLM_ATTENTION_BACKEND; fi
  echo "[gen $tag $(date +%H:%M:%S)]"
  for a in 1 2 3; do $VP/python scripts/triple_combo_eval.py --phase gen --target "$ref" --tag $tag \
    --order $ORDER --harm $HARM --answer-lang Swahili --n 4 --n-items 64 --util $U --max-model-len $ML --outdir $OUT && break; sleep 12; done
  echo "[judge $tag $(date +%H:%M:%S)]"
  for a in 1 2 3; do $VP/python scripts/triple_combo_eval.py --phase judge --tag $tag \
    --order $ORDER --harm $HARM --answer-lang Swahili --outdir $OUT && [ -f $OUT/$tag.json ] && break; sleep 12; done
}
gen_judge qwen25_3b     Qwen/Qwen2.5-3B-Instruct       flash
gen_judge llama32_3b_it meta-llama/Llama-3.2-3B-Instruct flash
gen_judge gemma2_2b_it  google/gemma-2-2b-it           triton
echo MJ_TRIPLE_SMALL_DONE
