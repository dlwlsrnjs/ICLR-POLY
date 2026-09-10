#!/bin/bash
# Per-domain breakdown on Lingua-SafetyBench (8 scenarios, 250 items), keeping per-item judgments,
# for the whole nine-target panel. Five configurations spanning the configuration families.
# Idempotent: a target whose json already exists is skipped, so this can be re-run after a failure.
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
ORDER=results/lang_rank_20260905/resource_order.json
HARM=private_artifacts/panel_v2/harm_grid.jsonl
OUT=results/domain_breakdown_20260907; AL=Norwegian
mkdir -p $OUT
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pick() { need=$1; while :; do
  g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -v n=$need '$2>=n{gsub(/,/,"",$1); print $1; exit}')
  [ -n "$g" ] && { echo $g; return 0; }; sleep 45; done; }
go() { tag="$1"; ref="$2"; be="$3"; util="$4"; need="$5"; mml="${6:-4096}"
  [ -f $OUT/$tag.json ] && { echo "OK $tag"; return; }
  if [ "$be" = triton ]; then export VLLM_ATTENTION_BACKEND=TRITON_ATTN; unset VLLM_USE_FLASHINFER_SAMPLER
  else export VLLM_USE_FLASHINFER_SAMPLER=0; unset VLLM_ATTENTION_BACKEND; fi
  if [ ! -f $OUT/_raw_$tag.json ]; then
    for a in 1 2 3; do g=$(pick $need); echo "[$(date +%H:%M:%S)] GEN $tag on GPU$g try $a"
      CUDA_VISIBLE_DEVICES=$g $VP/python scripts/domain_breakdown.py --phase gen --target "$ref" --tag $tag \
        --order $ORDER --harm $HARM --answer-lang $AL --n-items 250 --util $util --max-model-len $mml --outdir $OUT && break
      sleep 20; done
  fi
  unset VLLM_ATTENTION_BACKEND
  for a in 1 2 3; do g=$(pick 40000); echo "[$(date +%H:%M:%S)] JUDGE $tag on GPU$g try $a"
    CUDA_VISIBLE_DEVICES=$g $VP/python scripts/domain_breakdown.py --phase judge --tag $tag \
      --order $ORDER --harm $HARM --answer-lang $AL --n-items 250 --outdir $OUT && [ -f $OUT/$tag.json ] && break
    sleep 20; done
}
go qwen25_7b     Qwen/Qwen2.5-7B-Instruct         flash  0.35 34000
go gemma2_9b_it  google/gemma-2-9b-it             triton 0.40 40000
go llama31_8b_it meta-llama/Llama-3.1-8B-Instruct flash  0.35 34000
go qwen25_14b    Qwen/Qwen2.5-14B-Instruct        flash  0.45 45000
go gemma2_2b_it  google/gemma-2-2b-it             triton 0.35 34000
go qwen25_3b     Qwen/Qwen2.5-3B-Instruct         flash  0.35 34000
go llama32_3b_it meta-llama/Llama-3.2-3B-Instruct flash  0.35 34000
go qwen25_32b    Qwen/Qwen2.5-32B-Instruct        flash  0.90 74000 3072
go gemma2_27b    google/gemma-2-27b-it            triton 0.80 66000 3072
echo DOMAIN_BREAKDOWN_DONE
