#!/bin/bash
# Seed-robustness: re-run triple arm on the 3 strong MJ models with a 2nd puzzle seed.
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
HARM=private_artifacts/multijail_v1/harm_grid.jsonl
ORDER=private_artifacts/multijail_v1/resource_order.json
OUT=results/mj_triple_seed2_20260907; SEED=20260901
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${GPU:-1}
U=0.35; ML=4096
gj() { tag="$1"; ref="$2"; be="$3"
  [ -f $OUT/$tag.json ] && { echo "OK $tag"; return; }
  if [ "$be" = triton ]; then export VLLM_ATTENTION_BACKEND=TRITON_ATTN; unset VLLM_USE_FLASHINFER_SAMPLER
  else export VLLM_USE_FLASHINFER_SAMPLER=0; unset VLLM_ATTENTION_BACKEND; fi
  echo "[gen $tag $(date +%H:%M:%S)]"
  for a in 1 2 3; do $VP/python scripts/triple_combo_eval.py --phase gen --target "$ref" --tag $tag \
    --order $ORDER --harm $HARM --answer-lang Swahili --n 4 --n-items 64 --util $U --max-model-len $ML --outdir $OUT --seed $SEED && break; sleep 12; done
  echo "[judge $tag $(date +%H:%M:%S)]"
  for a in 1 2 3; do $VP/python scripts/triple_combo_eval.py --phase judge --tag $tag \
    --order $ORDER --harm $HARM --answer-lang Swahili --outdir $OUT --seed $SEED && [ -f $OUT/$tag.json ] && break; sleep 12; done
}
gj qwen25_7b     Qwen/Qwen2.5-7B-Instruct        flash
gj llama31_8b_it meta-llama/Llama-3.1-8B-Instruct flash
gj gemma2_9b_it  google/gemma-2-9b-it            triton
echo MJ_TRIPLE_SEED2_DONE
