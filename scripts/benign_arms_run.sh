#!/bin/bash
# Leak-free benign prior probe (Lingua): reconstruction-only, target-only, no judge. Light.
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
ORDER=results/lang_rank_20260905/resource_order.json
BENIGN=private_artifacts/panel_v2/benign_probe.jsonl
OUT=results/benign_arms_20260907; AL=Norwegian
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${GPU:-1}
go() { tag="$1"; ref="$2"; be="$3"; util="$4"
  [ -f $OUT/$tag.json ] && { echo "OK $tag"; return; }
  if [ "$be" = triton ]; then export VLLM_ATTENTION_BACKEND=TRITON_ATTN; unset VLLM_USE_FLASHINFER_SAMPLER
  else export VLLM_USE_FLASHINFER_SAMPLER=0; unset VLLM_ATTENTION_BACKEND; fi
  echo "[$(date +%H:%M:%S)] benign-probe $tag"
  for a in 1 2 3; do $VP/python scripts/benign_arms_probe.py --target "$ref" --tag $tag \
    --order $ORDER --benign $BENIGN --answer-lang $AL --n-items 40 --util $util --max-model-len 4096 --outdir $OUT && break; sleep 12; done
}
go qwen25_3b     Qwen/Qwen2.5-3B-Instruct        flash  0.35
go qwen25_7b     Qwen/Qwen2.5-7B-Instruct        flash  0.35
go qwen25_14b    Qwen/Qwen2.5-14B-Instruct       flash  0.45
go llama32_3b_it meta-llama/Llama-3.2-3B-Instruct flash  0.35
go llama31_8b_it meta-llama/Llama-3.1-8B-Instruct flash  0.35
go gemma2_2b_it  google/gemma-2-2b-it            triton 0.35
go gemma2_9b_it  google/gemma-2-9b-it            triton 0.40
echo BENIGN_ARMS_DONE
