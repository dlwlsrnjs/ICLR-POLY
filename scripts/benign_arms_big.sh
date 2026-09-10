#!/bin/bash
# Leak-free benign prior probe for the two large targets, completing the nine-model panel.
# Same protocol as benign_arms_run.sh (reconstruction only, target only, no judge, 40 FLORES items),
# with a GPU guard because these need most of a card.
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
ORDER=results/lang_rank_20260905/resource_order.json
BENIGN=private_artifacts/panel_v2/benign_probe.jsonl
OUT=results/benign_arms_20260907; AL=Norwegian
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pick() { need=$1; while :; do
  g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -v n=$need '$2>=n{print $1; exit}')
  [ -n "$g" ] && { echo $g; return 0; }; sleep 45; done; }
go() { tag="$1"; ref="$2"; be="$3"; util="$4"; need="$5"
  [ -f $OUT/$tag.json ] && { echo "OK $tag"; return; }
  if [ "$be" = triton ]; then export VLLM_ATTENTION_BACKEND=TRITON_ATTN; unset VLLM_USE_FLASHINFER_SAMPLER
  else export VLLM_USE_FLASHINFER_SAMPLER=0; unset VLLM_ATTENTION_BACKEND; fi
  for a in 1 2 3; do
    g=$(pick $need); echo "[$(date +%H:%M:%S)] benign-probe $tag on GPU$g (try $a)"
    CUDA_VISIBLE_DEVICES=$g $VP/python scripts/benign_arms_probe.py --target "$ref" --tag $tag \
      --order $ORDER --benign $BENIGN --answer-lang $AL --n-items 40 --util $util \
      --max-model-len 3072 --outdir $OUT && break
    sleep 20
  done
}
go qwen25_32b Qwen/Qwen2.5-32B-Instruct flash  0.90 71000
go gemma2_27b google/gemma-2-27b-it     triton 0.80 62000
echo BENIGN_ARMS_BIG_DONE
