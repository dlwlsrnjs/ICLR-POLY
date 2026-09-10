#!/bin/bash
# Held-out generalization: build the FULL arm space + the harmless warm-start probe for models that
# were NOT used to construct the configuration space, the prior, or the winner's-curse calibration.
# Same two-phase protocol (gen target-only / judge judges-only), same arm families, both collections.
# GPU-guarded, idempotent (skips finished outputs), so it can be restarted at any point.
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ENFORCE_EAGER=1
export VLLM_USE_FLASHINFER_SAMPLER=0
MJ_HARM=private_artifacts/multijail_v1/harm_grid.jsonl;   MJ_ORDER=private_artifacts/multijail_v1/resource_order.json
LG_HARM=private_artifacts/panel_v2/harm_grid.jsonl;       LG_ORDER=results/lang_rank_20260905/resource_order.json
MJ_BEN=private_artifacts/multijail_v1/benign_probe.jsonl; LG_BEN=private_artifacts/panel_v2/benign_probe.jsonl

pick_gpu(){ need=$1; for w in $(seq 1 960); do
  while read idx free; do [ "$free" -ge "$need" ] && { echo "$idx"; return 0; }; done \
    < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  echo "[wait $(date +%H:%M:%S)] no GPU with ${need}MB free" >&2; sleep 30; done; return 1; }

run2p(){ local script="$1" extra="$2" outfile="$3" needg="$4" needj="$5"
  [ -f "$outfile" ] && { echo "OK $outfile"; return 0; }
  local g; g=$(pick_gpu "$needg")||{ echo "no GPU $outfile"; return 1; }
  echo "[$(date +%H:%M:%S)] GEN GPU$g -> $outfile"
  for t in 1 2 3; do CUDA_VISIBLE_DEVICES=$g $VP/python $script --phase gen $extra && break; sleep 15; done
  g=$(pick_gpu "$needj")||return 1
  echo "[$(date +%H:%M:%S)] JUDGE GPU$g -> $outfile"
  for t in 1 2 3; do CUDA_VISIBLE_DEVICES=$g $VP/python $script --phase judge $extra && [ -f "$outfile" ] && return 0; sleep 15; done; }

run_benign(){ local extra="$1" outfile="$2" needg="$3"
  [ -f "$outfile" ] && { echo "OK $outfile"; return 0; }
  local g; g=$(pick_gpu "$needg")||return 1
  echo "[$(date +%H:%M:%S)] BENIGN GPU$g -> $outfile"
  for t in 1 2 3; do CUDA_VISIBLE_DEVICES=$g $VP/python scripts/benign_arms_probe.py $extra && [ -f "$outfile" ] && return 0; sleep 15; done; }

# $1 tag  $2 ref  $3 trust-flag  $4 gen-util  $5 needg(MB)  $6 needj(MB)  $7 max-model-len
do_model(){ tag="$1"; ref="$2"; TR="$3"; gu="$4"; ng="$5"; nj="$6"; mml="${7:-4096}"
  C="--target $ref --tag $tag $TR --tokenizer-mode auto --util $gu --max-model-len $mml"
  echo "======== HELD-OUT $tag ($ref) ========"
  # ---- MultiJail (64 items) ----
  run2p scripts/amount_disorder_2phase.py "--kind amount   $C --n 4 --order $MJ_ORDER --harm $MJ_HARM --n-items 64 --outdir results/mj_sequential_20260906" results/mj_sequential_20260906/$tag.json $ng $nj
  run2p scripts/amount_disorder_2phase.py "--kind disorder $C --n 4 --order $MJ_ORDER --harm $MJ_HARM --n-items 64 --outdir results/mj_disorder_20260906"   results/mj_disorder_20260906/$tag.json $ng $nj
  run2p scripts/mj_2phase_generic.py      "--kind combo    $C --n 4 --order $MJ_ORDER --harm $MJ_HARM --n-items 64 --outdir results/mj_combo_20260906"      results/mj_combo_20260906/$tag.json $ng $nj
  run2p scripts/mj_2phase_generic.py      "--kind method   $C --n 4 --order $MJ_ORDER --harm $MJ_HARM --n-items 64 --outdir results/mj_method_20260906"     results/mj_method_20260906/$tag.json $ng $nj
  run2p scripts/triple_combo_eval.py      "$C --n 4 --order $MJ_ORDER --harm $MJ_HARM --n-items 64 --answer-lang Swahili --outdir results/mj_triple_20260906"  results/mj_triple_20260906/$tag.json $ng $nj
  run2p scripts/triple_hien_eval.py       "$C --n 4 --order $MJ_ORDER --harm $MJ_HARM --n-items 64 --answer-lang Swahili --outdir results/mj_hien_20260908"    results/mj_hien_20260908/$tag.json $ng $nj
  run2p scripts/extra_arms.py             "$C --order $MJ_ORDER --harm $MJ_HARM --n-items 64 --answer-lang Swahili --outdir results/extra_arms_mj_20260908"    results/extra_arms_mj_20260908/$tag.json $ng $nj
  run_benign "$C --order $MJ_ORDER --benign $MJ_BEN --answer-lang Swahili --n-items 64 --outdir results/benign_arms_mj_20260907" results/benign_arms_mj_20260907/$tag.json $ng
  # ---- Lingua-SafetyBench (40 items) ----
  run2p scripts/amount_disorder_2phase.py "--kind amount   $C --n 4 --order $LG_ORDER --harm $LG_HARM --n-items 40 --outdir results/sequential_resource_20260906" results/sequential_resource_20260906/$tag.json $ng $nj
  run2p scripts/amount_disorder_2phase.py "--kind disorder $C --n 4 --order $LG_ORDER --harm $LG_HARM --n-items 40 --outdir results/disorder_sweep_20260906"       results/disorder_sweep_20260906/$tag.json $ng $nj
  run2p scripts/mj_2phase_generic.py      "--kind combo    $C --n 4 --order $LG_ORDER --harm $LG_HARM --n-items 40 --outdir results/combo_20260906"                results/combo_20260906/$tag.json $ng $nj
  run2p scripts/mj_2phase_generic.py      "--kind method   $C --n 4 --order $LG_ORDER --harm $LG_HARM --n-items 40 --outdir results/method_baselines_v2_20260906"  results/method_baselines_v2_20260906/$tag.json $ng $nj
  run2p scripts/triple_combo_eval.py      "$C --n 4 --order $LG_ORDER --harm $LG_HARM --n-items 40 --answer-lang Norwegian --outdir results/lingua_triple_20260907" results/lingua_triple_20260907/$tag.json $ng $nj
  run2p scripts/triple_hien_eval.py       "$C --n 4 --order $LG_ORDER --harm $LG_HARM --n-items 40 --answer-lang Norwegian --outdir results/lingua_hien_20260908"   results/lingua_hien_20260908/$tag.json $ng $nj
  run2p scripts/extra_arms.py             "$C --order $LG_ORDER --harm $LG_HARM --n-items 40 --answer-lang Norwegian --outdir results/extra_arms_lg_20260908"       results/extra_arms_lg_20260908/$tag.json $ng $nj
  run_benign "$C --order $LG_ORDER --benign $LG_BEN --answer-lang Norwegian --n-items 40 --outdir results/benign_arms_20260907" results/benign_arms_20260907/$tag.json $ng
  echo "HELDOUT_DONE $tag"
}

do_model phi35_mini   microsoft/Phi-3.5-mini-instruct           "--trust-remote-code" 0.45 40000 44000
do_model mistral7b    mistralai/Mistral-7B-Instruct-v0.3         "--trust-remote-code" 0.45 40000 44000
do_model falcon3_7b   tiiuae/Falcon3-7B-Instruct                 "--trust-remote-code" 0.45 40000 44000
do_model glm4_9b      THUDM/glm-4-9b-chat-hf                     "--trust-remote-code" 0.55 46000 44000
do_model mistral24b   mistralai/Mistral-Small-24B-Instruct-2501  "--trust-remote-code" 0.90 74000 44000 3072
echo HELDOUT_ALL_DONE
