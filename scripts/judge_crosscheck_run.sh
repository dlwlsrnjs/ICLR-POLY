#!/bin/bash
# Dual-judge cross-check, robustly GPU-guarded (persistent). Waits for real headroom before each step
# so it survives heavy contention from other users; idempotent (skips finished gens).
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
HARM=private_artifacts/multijail_v1/harm_grid.jsonl; ORDER=private_artifacts/multijail_v1/resource_order.json
OUT=results/judge_crosscheck_20260907
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p $OUT
pick() { need=$1; while :; do
  g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -v n=$need '$2>=n{print $1; exit}')
  [ -n "$g" ] && { echo $g; return 0; }; sleep 45; done; }
gen() { tag="$1"; ref="$2"; be="$3"; util="$4"
  [ -f $OUT/_raw_cc_$tag.json ] && { echo "gen OK $tag"; return; }
  if [ "$be" = triton ]; then export VLLM_ATTENTION_BACKEND=TRITON_ATTN; else unset VLLM_ATTENTION_BACKEND; fi
  while [ ! -f $OUT/_raw_cc_$tag.json ]; do
    g=$(pick 34000); echo "[$(date +%H:%M:%S)] GEN $tag on GPU$g"
    CUDA_VISIBLE_DEVICES=$g $VP/python scripts/judge_crosscheck.py --phase gen --target "$ref" --tag $tag \
      --order $ORDER --harm $HARM --answer-lang Swahili --n 4 --n-items 64 --util $util --max-model-len 4096 --outdir $OUT || true
    [ -f $OUT/_raw_cc_$tag.json ] || { echo "[$(date +%H:%M:%S)] $tag gen failed, waiting"; sleep 60; }
  done
}
gen qwen25_7b     Qwen/Qwen2.5-7B-Instruct        flash  0.35
gen llama31_8b_it meta-llama/Llama-3.1-8B-Instruct flash  0.35
gen gemma2_9b_it  google/gemma-2-9b-it           triton 0.40
unset VLLM_ATTENTION_BACKEND
while [ ! -f $OUT/crosscheck_summary.json ]; do
  g=$(pick 52000); echo "[$(date +%H:%M:%S)] JUDGE (Qwen3Guard+MD-Judge) on GPU$g"
  CUDA_VISIBLE_DEVICES=$g $VP/python scripts/judge_crosscheck.py --phase judge --tag all --order $ORDER --harm $HARM \
    --answer-lang Swahili --tags qwen25_7b,llama31_8b_it,gemma2_9b_it --outdir $OUT || true
  [ -f $OUT/crosscheck_summary.json ] || { echo "[$(date +%H:%M:%S)] judge failed, waiting"; sleep 60; }
done
echo JUDGE_CROSSCHECK_DONE
