#!/bin/bash
# Add 32B-class models (Qwen2.5-32B, Gemma-2-27B) to BOTH datasets across all 5 arm families,
# all two-phase (gen target-only / judge judges-only) so bf16 fits on one GPU. GPU-free guard picks
# whichever GPU is empty enough and waits otherwise. Idempotent (skips finished outputs).
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ENFORCE_EAGER=1
MJ_HARM=private_artifacts/multijail_v1/harm_grid.jsonl;      MJ_ORDER=private_artifacts/multijail_v1/resource_order.json
LG_HARM=private_artifacts/panel_v2/harm_grid.jsonl;          LG_ORDER=results/lang_rank_20260905/resource_order.json

# pick a GPU with >= $1 MB free (echo its index), waiting up to ~4h
pick_gpu() { need=$1; for w in $(seq 1 480); do
  while read idx free; do [ "$free" -ge "$need" ] && { echo "$idx"; return 0; }; done \
    < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  echo "[wait $(date +%H:%M:%S)] no GPU with ${need}MB free" >&2; sleep 30; done; return 1; }

run2p() { # SCRIPT EXTRA_GEN OUT NEEDgen NEEDjudge GENUTILflag
  local script="$1" extra="$2" outfile="$3" needg="$4" needj="$5"
  [ -f "$outfile" ] && { echo "OK $outfile"; return 0; }
  local g; g=$(pick_gpu "$needg") || { echo "no GPU for $outfile"; return 1; }
  echo "[$(date +%H:%M:%S)] GEN on GPU$g -> $outfile"
  for t in 1 2 3; do CUDA_VISIBLE_DEVICES=$g $VP/python $script --phase gen $extra && break; sleep 15; done
  g=$(pick_gpu "$needj") || return 1
  echo "[$(date +%H:%M:%S)] JUDGE on GPU$g -> $outfile"
  for t in 1 2 3; do CUDA_VISIBLE_DEVICES=$g $VP/python $script --phase judge $extra && [ -f "$outfile" ] && return 0; sleep 15; done; }

# $1 tag  $2 ref  $3 backend(flash|triton)  $4 gen-util  $5 needg(MB)  $6 needj(MB)
do_model() { tag="$1"; ref="$2"; be="$3"; gu="$4"; ng="$5"; nj="$6"
  if [ "$be" = triton ]; then export VLLM_ATTENTION_BACKEND=TRITON_ATTN; unset VLLM_USE_FLASHINFER_SAMPLER
  else export VLLM_USE_FLASHINFER_SAMPLER=0; unset VLLM_ATTENTION_BACKEND; fi
  COMMON="--target $ref --tag $tag --n 4 --util $gu --max-model-len 3072"
  # ---- MultiJail (n-items 64) ----
  run2p scripts/amount_disorder_2phase.py "--kind amount   $COMMON --order $MJ_ORDER --harm $MJ_HARM --n-items 64 --outdir results/mj_sequential_20260906" results/mj_sequential_20260906/$tag.json $ng $nj
  run2p scripts/amount_disorder_2phase.py "--kind disorder $COMMON --order $MJ_ORDER --harm $MJ_HARM --n-items 64 --outdir results/mj_disorder_20260906"   results/mj_disorder_20260906/$tag.json $ng $nj
  run2p scripts/mj_2phase_generic.py      "--kind combo    $COMMON --order $MJ_ORDER --harm $MJ_HARM --n-items 64 --outdir results/mj_combo_20260906"      results/mj_combo_20260906/$tag.json $ng $nj
  run2p scripts/mj_2phase_generic.py      "--kind method   $COMMON --order $MJ_ORDER --harm $MJ_HARM --n-items 64 --outdir results/mj_method_20260906"     results/mj_method_20260906/$tag.json $ng $nj
  run2p scripts/triple_combo_eval.py      "$COMMON --order $MJ_ORDER --harm $MJ_HARM --n-items 64 --answer-lang Swahili --outdir results/mj_triple_20260906" results/mj_triple_20260906/$tag.json $ng $nj
  # ---- Lingua (n-items 40) ----
  run2p scripts/amount_disorder_2phase.py "--kind amount   $COMMON --order $LG_ORDER --harm $LG_HARM --n-items 40 --outdir results/sequential_resource_20260906" results/sequential_resource_20260906/$tag.json $ng $nj
  run2p scripts/amount_disorder_2phase.py "--kind disorder $COMMON --order $LG_ORDER --harm $LG_HARM --n-items 40 --outdir results/disorder_sweep_20260906"       results/disorder_sweep_20260906/$tag.json $ng $nj
  run2p scripts/mj_2phase_generic.py      "--kind combo    $COMMON --order $LG_ORDER --harm $LG_HARM --n-items 40 --outdir results/combo_20260906"                results/combo_20260906/$tag.json $ng $nj
  run2p scripts/mj_2phase_generic.py      "--kind method   $COMMON --order $LG_ORDER --harm $LG_HARM --n-items 40 --outdir results/method_baselines_v2_20260906"  results/method_baselines_v2_20260906/$tag.json $ng $nj
  run2p scripts/triple_combo_eval.py      "$COMMON --order $LG_ORDER --harm $LG_HARM --n-items 40 --answer-lang Norwegian --outdir results/lingua_triple_20260907" results/lingua_triple_20260907/$tag.json $ng $nj
  echo "DONE_MODEL $tag"
}

do_model qwen25_32b Qwen/Qwen2.5-32B-Instruct flash  0.90 71000 34000
do_model gemma2_27b google/gemma-2-27b-it     triton 0.80 62000 34000
echo BIG_MODELS_ALL_DONE
