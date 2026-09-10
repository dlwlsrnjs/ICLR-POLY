#!/bin/bash
# Harmless discriminative signals for every target, for both collections' answer languages.
# Cheap: 16 harmless requests per answer frame, no judge.
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pick() { need=$1; while :; do
  g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -v n=$need '$2>=n{gsub(/,/,"",$1); print $1; exit}')
  [ -n "$g" ] && { echo $g; return 0; }; sleep 45; done; }
go() { tag="$1"; ref="$2"; be="$3"; util="$4"; need="$5"; al="$6"; outd="$7"; mml="${8:-4096}"
  [ -f $outd/$tag.json ] && { echo "OK $outd/$tag"; return; }
  if [ "$be" = triton ]; then export VLLM_ATTENTION_BACKEND=TRITON_ATTN; unset VLLM_USE_FLASHINFER_SAMPLER
  else export VLLM_USE_FLASHINFER_SAMPLER=0; unset VLLM_ATTENTION_BACKEND; fi
  for a in 1 2 3; do
    g=$(pick $need); echo "[$(date +%H:%M:%S)] signals $tag ($al) on GPU$g try $a"
    CUDA_VISIBLE_DEVICES=$g $VP/python scripts/benign_signals_probe.py --target "$ref" --tag $tag \
      --answer-lang $al --util $util --max-model-len $mml --outdir $outd && break
    sleep 20
  done
}
for spec in "Norwegian results/benign_signals_20260907" "Swahili results/benign_signals_mj_20260907"; do
  set -- $spec; AL=$1; OUT=$2
  go qwen25_3b     Qwen/Qwen2.5-3B-Instruct         flash  0.35 30000 $AL $OUT
  go qwen25_7b     Qwen/Qwen2.5-7B-Instruct         flash  0.35 34000 $AL $OUT
  go qwen25_14b    Qwen/Qwen2.5-14B-Instruct        flash  0.45 42000 $AL $OUT
  go llama32_3b_it meta-llama/Llama-3.2-3B-Instruct flash  0.35 30000 $AL $OUT
  go llama31_8b_it meta-llama/Llama-3.1-8B-Instruct flash  0.35 34000 $AL $OUT
  go gemma2_2b_it  google/gemma-2-2b-it             triton 0.35 30000 $AL $OUT
  go gemma2_9b_it  google/gemma-2-9b-it             triton 0.40 34000 $AL $OUT
  go qwen25_32b    Qwen/Qwen2.5-32B-Instruct        flash  0.90 71000 $AL $OUT 3072
  go gemma2_27b    google/gemma-2-27b-it            triton 0.80 62000 $AL $OUT 3072
done
echo BENIGN_SIGNALS_DONE
