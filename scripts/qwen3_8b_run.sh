#!/bin/bash
# Held-out 10th model (valid substitute for the dropped internlm25 / invalid gemma-2-9b): Qwen/Qwen3-8B,
# a NON-panel model (panel is Qwen2.5, not Qwen3) with a clean tokenizer. Qwen3 has a thinking mode, so
# --no-thinking is required (thinking output breaks the [RECONSTRUCTED]/[ANSWER] format). Runs the same
# 16-eval do_model protocol as heldout_run.sh. Chained LAST (waits for the live query-eff job's marker,
# which waits for the expansion) so it never contends for the single freed GPU. GPU-guarded + idempotent.
# After it finishes, run heldout_selector.py + heldout_baseline_means.py (with the full set) to fold it in.
set +e
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ENFORCE_EAGER=1
export VLLM_USE_FLASHINFER_SAMPLER=0
MJ_HARM=private_artifacts/multijail_v1/harm_grid.jsonl;   MJ_ORDER=private_artifacts/multijail_v1/resource_order.json
LG_HARM=private_artifacts/panel_v2/harm_grid.jsonl;       LG_ORDER=results/lang_rank_20260905/resource_order.json
MJ_BEN=private_artifacts/multijail_v1/benign_probe.jsonl; LG_BEN=private_artifacts/panel_v2/benign_probe.jsonl

pick_gpu(){ need=$1; for w in $(seq 1 5760); do
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

# --- chain LAST: wait for the live query-eff job to finish (it already waits for the expansion) ---
echo "[$(date +%H:%M:%S)] qwen3_8b held-out queued; waiting for live query-eff job to finish"
while :; do
  grep -qE "LIVE_QUERYEFF_DONE|LIVE_QUERYEFF_FAILED" results/livequeryeff.log 2>/dev/null && { echo "[$(date +%H:%M:%S)] live marker seen"; break; }
  pgrep -f livequeryeff_run.sh >/dev/null 2>&1 || { echo "[$(date +%H:%M:%S)] live job gone; proceeding"; break; }
  sleep 60
done

tag=qwen3_8b; ref=Qwen/Qwen3-8B; ng=40000; nj=44000
C="--target $ref --tag $tag --trust-remote-code --no-thinking --tokenizer-mode auto --util 0.45 --max-model-len 4096"
echo "======== HELD-OUT (Qwen3-8B, no-thinking) $tag ($ref) ========"
# ---- MultiJail (64) ----
run2p scripts/amount_disorder_2phase.py "--kind amount   $C --n 4 --order $MJ_ORDER --harm $MJ_HARM --n-items 64 --outdir results/mj_sequential_20260906" results/mj_sequential_20260906/$tag.json $ng $nj
run2p scripts/amount_disorder_2phase.py "--kind disorder $C --n 4 --order $MJ_ORDER --harm $MJ_HARM --n-items 64 --outdir results/mj_disorder_20260906"   results/mj_disorder_20260906/$tag.json $ng $nj
run2p scripts/mj_2phase_generic.py      "--kind combo    $C --n 4 --order $MJ_ORDER --harm $MJ_HARM --n-items 64 --outdir results/mj_combo_20260906"      results/mj_combo_20260906/$tag.json $ng $nj
run2p scripts/mj_2phase_generic.py      "--kind method   $C --n 4 --order $MJ_ORDER --harm $MJ_HARM --n-items 64 --outdir results/mj_method_20260906"     results/mj_method_20260906/$tag.json $ng $nj
run2p scripts/triple_combo_eval.py      "$C --n 4 --order $MJ_ORDER --harm $MJ_HARM --n-items 64 --answer-lang Swahili --outdir results/mj_triple_20260906"  results/mj_triple_20260906/$tag.json $ng $nj
run2p scripts/triple_hien_eval.py       "$C --n 4 --order $MJ_ORDER --harm $MJ_HARM --n-items 64 --answer-lang Swahili --outdir results/mj_hien_20260908"    results/mj_hien_20260908/$tag.json $ng $nj
run2p scripts/extra_arms.py             "$C --order $MJ_ORDER --harm $MJ_HARM --n-items 64 --answer-lang Swahili --outdir results/extra_arms_mj_20260908"    results/extra_arms_mj_20260908/$tag.json $ng $nj
run_benign "$C --order $MJ_ORDER --benign $MJ_BEN --answer-lang Swahili --n-items 64 --outdir results/benign_arms_mj_20260907" results/benign_arms_mj_20260907/$tag.json $ng
# ---- Lingua-SafetyBench (40) ----
run2p scripts/amount_disorder_2phase.py "--kind amount   $C --n 4 --order $LG_ORDER --harm $LG_HARM --n-items 40 --outdir results/sequential_resource_20260906" results/sequential_resource_20260906/$tag.json $ng $nj
run2p scripts/amount_disorder_2phase.py "--kind disorder $C --n 4 --order $LG_ORDER --harm $LG_HARM --n-items 40 --outdir results/disorder_sweep_20260906"       results/disorder_sweep_20260906/$tag.json $ng $nj
run2p scripts/mj_2phase_generic.py      "--kind combo    $C --n 4 --order $LG_ORDER --harm $LG_HARM --n-items 40 --outdir results/combo_20260906"                results/combo_20260906/$tag.json $ng $nj
run2p scripts/mj_2phase_generic.py      "--kind method   $C --n 4 --order $LG_ORDER --harm $LG_HARM --n-items 40 --outdir results/method_baselines_v2_20260906"  results/method_baselines_v2_20260906/$tag.json $ng $nj
run2p scripts/triple_combo_eval.py      "$C --n 4 --order $LG_ORDER --harm $LG_HARM --n-items 40 --answer-lang Norwegian --outdir results/lingua_triple_20260907" results/lingua_triple_20260907/$tag.json $ng $nj
run2p scripts/triple_hien_eval.py       "$C --n 4 --order $LG_ORDER --harm $LG_HARM --n-items 40 --answer-lang Norwegian --outdir results/lingua_hien_20260908"   results/lingua_hien_20260908/$tag.json $ng $nj
run2p scripts/extra_arms.py             "$C --order $LG_ORDER --harm $LG_HARM --n-items 40 --answer-lang Norwegian --outdir results/extra_arms_lg_20260908"       results/extra_arms_lg_20260908/$tag.json $ng $nj
run_benign "$C --order $LG_ORDER --benign $LG_BEN --answer-lang Norwegian --n-items 40 --outdir results/benign_arms_20260907" results/benign_arms_20260907/$tag.json $ng
echo "QWEN3_8B_DONE $(date +%H:%M:%S)"
echo "NEXT: expand HELDOUT in heldout_selector.py + heldout_baseline_means.py to the full completed set and re-run both to fold qwen3_8b (and the expansion models) into the held-out tables at one consistent n."
