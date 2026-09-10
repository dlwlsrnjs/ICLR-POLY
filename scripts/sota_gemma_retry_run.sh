#!/bin/bash
# Retry the 3 gemma-2 SOTA (DrAttack/FlipAttack) cells that crashed under the default vLLM v1 attention
# backend (gemma-2 logit softcapping fatally crashes EngineCore). The core pipeline runs gemma with
# VLLM_ATTENTION_BACKEND=TRITON_ATTN; the sota_baselines_run.sh omitted it. This fills the 3 missing
# panel cells so the held-in SOTA mean is over all 9 panel models (not 6). Two-phase, GPU-guarded,
# idempotent. New file (does not touch jinkwon's originals).
set +e
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ENFORCE_EAGER=1
export VLLM_ATTENTION_BACKEND=TRITON_ATTN; unset VLLM_USE_FLASHINFER_SAMPLER
MJ_HARM=private_artifacts/multijail_v1/harm_grid.jsonl
LG_HARM=private_artifacts/panel_v2/harm_grid.jsonl
MJ_OUT=results/sota_mj_20260909; LG_OUT=results/sota_lg_20260909
pick_gpu(){ need=$1; while :; do
  g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -v n=$need '$2>=n{gsub(/,/,"",$1); print $1; exit}')
  [ -n "$g" ] && { echo "$g"; return 0; }; echo "[wait $(date +%H:%M:%S)] no GPU ${need}MB" >&2; sleep 30; done; }
run2p(){ local extra="$1" harm="$2" nit="$3" outdir="$4" tag="$5" needg="$6" needj="$7"
  local outfile="$outdir/$tag.json"
  [ -f "$outfile" ] && { echo "OK $outfile"; return 0; }
  local g; g=$(pick_gpu "$needg")
  echo "[$(date +%H:%M:%S)] GEN GPU$g -> $outfile"
  for t in 1 2 3; do CUDA_VISIBLE_DEVICES=$g $VP/python scripts/sota_baselines_eval.py --phase gen $extra --harm $harm --n-items $nit --outdir $outdir && break; sleep 15; done
  g=$(pick_gpu "$needj")
  echo "[$(date +%H:%M:%S)] JUDGE GPU$g -> $outfile"
  for t in 1 2 3; do CUDA_VISIBLE_DEVICES=$g $VP/python scripts/sota_baselines_eval.py --phase judge $extra --harm $harm --n-items $nit --outdir $outdir && [ -f "$outfile" ] && return 0; sleep 15; done; }
do_sota(){ local tag="$1" ref="$2" util="$3" ng="$4" nj="$5" mml="${6:-4096}"
  local C="--target $ref --tag $tag --trust-remote-code --tokenizer-mode auto --util $util --max-model-len $mml"
  echo "======== SOTA(gemma retry) $tag ($ref) ========"
  run2p "$C" "$MJ_HARM" 64 "$MJ_OUT" "$tag" "$ng" "$nj"
  run2p "$C" "$LG_HARM" 40 "$LG_OUT" "$tag" "$ng" "$nj"
  echo "GEMMA_SOTA_DONE $tag"
}
# clear any half-written raw so gen re-runs cleanly under the triton backend
for t in gemma2_2b_it gemma2_9b_it gemma2_27b; do rm -f $MJ_OUT/_raw_$t.json $LG_OUT/_raw_$t.json 2>/dev/null; done
do_sota gemma2_2b_it  google/gemma-2-2b-it   0.45 40000 44000
do_sota gemma2_9b_it  google/gemma-2-9b-it   0.55 46000 44000
do_sota gemma2_27b    google/gemma-2-27b-it  0.80 66000 44000 3072
echo "GEMMA_SOTA_ALL_DONE $(date +%H:%M:%S)"
