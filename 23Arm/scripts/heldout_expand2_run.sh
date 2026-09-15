#!/bin/bash
# Move-3 target expansion: add NEW vendor-disjoint open held-out models to raise held-out n (7 -> up to 11)
# for a better-powered transfer significance test. Full protocol per model: 14 harmful two-phase evals
# (7 MultiJail + 7 Lingua) + recon benign probe + signals + borderline probes (the frozen recipe needs
# borderline-detail on MJ and fiction_hold on LG). GPU-guarded, idempotent, set +e so one model's failure
# (exotic tokenizer/template) does not stop the rest. New file; does not touch jinkwon originals.
set +e
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ENFORCE_EAGER=1 VLLM_USE_FLASHINFER_SAMPLER=0
MJ_HARM=private_artifacts/multijail_v1/harm_grid.jsonl;   MJ_ORDER=private_artifacts/multijail_v1/resource_order.json
LG_HARM=private_artifacts/panel_v2/harm_grid.jsonl;       LG_ORDER=results/lang_rank_20260905/resource_order.json
MJ_BEN=private_artifacts/multijail_v1/benign_probe.jsonl; LG_BEN=private_artifacts/panel_v2/benign_probe.jsonl
pick_gpu(){ need=$1; for w in $(seq 1 2880); do
  while read idx free; do [ "$free" -ge "$need" ] && { echo "$idx"; return 0; }; done \
    < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  echo "[wait $(date +%H:%M:%S)] no GPU ${need}MB" >&2; sleep 30; done; return 1; }
run2p(){ local script="$1" extra="$2" outfile="$3" needg="$4" needj="$5"
  [ -f "$outfile" ] && { echo "OK $outfile"; return 0; }
  local g; g=$(pick_gpu "$needg")||return 1
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
sig(){ local extra="$1" outfile="$2" needg="$3"
  [ -f "$outfile" ] && { echo "OK $outfile"; return 0; }
  local g; g=$(pick_gpu "$needg")||return 1
  echo "[$(date +%H:%M:%S)] SIG GPU$g -> $outfile"
  for t in 1 2 3; do CUDA_VISIBLE_DEVICES=$g $VP/python scripts/benign_signals_probe.py $extra && [ -f "$outfile" ] && return 0; sleep 15; done; }
do_model(){ tag="$1"; ref="$2"; TR="$3"; NT="$4"; gu="$5"; ng="$6"; nj="$7"; mml="${8:-4096}"
  C="--target $ref --tag $tag $TR $NT --tokenizer-mode auto --util $gu --max-model-len $mml"
  echo "======== HELD-OUT+ $tag ($ref) ========"
  run2p scripts/amount_disorder_2phase.py "--kind amount   $C --n 4 --order $MJ_ORDER --harm $MJ_HARM --n-items 64 --outdir results/mj_sequential_20260906" results/mj_sequential_20260906/$tag.json $ng $nj
  run2p scripts/amount_disorder_2phase.py "--kind disorder $C --n 4 --order $MJ_ORDER --harm $MJ_HARM --n-items 64 --outdir results/mj_disorder_20260906"   results/mj_disorder_20260906/$tag.json $ng $nj
  run2p scripts/mj_2phase_generic.py      "--kind combo    $C --n 4 --order $MJ_ORDER --harm $MJ_HARM --n-items 64 --outdir results/mj_combo_20260906"      results/mj_combo_20260906/$tag.json $ng $nj
  run2p scripts/mj_2phase_generic.py      "--kind method   $C --n 4 --order $MJ_ORDER --harm $MJ_HARM --n-items 64 --outdir results/mj_method_20260906"     results/mj_method_20260906/$tag.json $ng $nj
  run2p scripts/triple_combo_eval.py      "$C --n 4 --order $MJ_ORDER --harm $MJ_HARM --n-items 64 --answer-lang Swahili --outdir results/mj_triple_20260906"  results/mj_triple_20260906/$tag.json $ng $nj
  run2p scripts/triple_hien_eval.py       "$C --n 4 --order $MJ_ORDER --harm $MJ_HARM --n-items 64 --answer-lang Swahili --outdir results/mj_hien_20260908"    results/mj_hien_20260908/$tag.json $ng $nj
  run2p scripts/extra_arms.py             "$C --order $MJ_ORDER --harm $MJ_HARM --n-items 64 --answer-lang Swahili --outdir results/extra_arms_mj_20260908"    results/extra_arms_mj_20260908/$tag.json $ng $nj
  run_benign "$C --order $MJ_ORDER --benign $MJ_BEN --answer-lang Swahili --n-items 64 --outdir results/benign_arms_mj_20260907" results/benign_arms_mj_20260907/$tag.json $ng
  run2p scripts/amount_disorder_2phase.py "--kind amount   $C --n 4 --order $LG_ORDER --harm $LG_HARM --n-items 40 --outdir results/sequential_resource_20260906" results/sequential_resource_20260906/$tag.json $ng $nj
  run2p scripts/amount_disorder_2phase.py "--kind disorder $C --n 4 --order $LG_ORDER --harm $LG_HARM --n-items 40 --outdir results/disorder_sweep_20260906"       results/disorder_sweep_20260906/$tag.json $ng $nj
  run2p scripts/mj_2phase_generic.py      "--kind combo    $C --n 4 --order $LG_ORDER --harm $LG_HARM --n-items 40 --outdir results/combo_20260906"                results/combo_20260906/$tag.json $ng $nj
  run2p scripts/mj_2phase_generic.py      "--kind method   $C --n 4 --order $LG_ORDER --harm $LG_HARM --n-items 40 --outdir results/method_baselines_v2_20260906"  results/method_baselines_v2_20260906/$tag.json $ng $nj
  run2p scripts/triple_combo_eval.py      "$C --n 4 --order $LG_ORDER --harm $LG_HARM --n-items 40 --answer-lang Norwegian --outdir results/lingua_triple_20260907" results/lingua_triple_20260907/$tag.json $ng $nj
  run2p scripts/triple_hien_eval.py       "$C --n 4 --order $LG_ORDER --harm $LG_HARM --n-items 40 --answer-lang Norwegian --outdir results/lingua_hien_20260908"   results/lingua_hien_20260908/$tag.json $ng $nj
  run2p scripts/extra_arms.py             "$C --order $LG_ORDER --harm $LG_HARM --n-items 40 --answer-lang Norwegian --outdir results/extra_arms_lg_20260908"       results/extra_arms_lg_20260908/$tag.json $ng $nj
  run_benign "$C --order $LG_ORDER --benign $LG_BEN --answer-lang Norwegian --n-items 40 --outdir results/benign_arms_20260907" results/benign_arms_20260907/$tag.json $ng
  # signals + borderline probes (both collections)
  P="--target $ref --tag $tag $TR $NT --tokenizer-mode auto --util $gu --max-model-len $mml"
  sig "$P --answer-lang Swahili   --outdir results/benign_signals_mj_20260907"    results/benign_signals_mj_20260907/$tag.json $ng
  sig "$P --answer-lang Norwegian --outdir results/benign_signals_20260907"       results/benign_signals_20260907/$tag.json $ng
  sig "$P --answer-lang Swahili   --request-set borderline --outdir results/benign_borderline_mj_20260907" results/benign_borderline_mj_20260907/$tag.json $ng
  sig "$P --answer-lang Norwegian --request-set borderline --outdir results/benign_borderline_20260907"    results/benign_borderline_20260907/$tag.json $ng
  echo "HELDOUT2_DONE $tag $(date +%H:%M:%S)"
}
do_model exaone35_8b   LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct   "--trust-remote-code" ""  0.45 40000 44000
do_model solar_11b     upstage/SOLAR-10.7B-Instruct-v1.0      "--trust-remote-code" ""  0.55 46000 44000
do_model baichuan2_7b  baichuan-inc/Baichuan2-7B-Chat         "--trust-remote-code" ""  0.45 40000 44000
do_model minicpm3_4b   openbmb/MiniCPM3-4B                    "--trust-remote-code" ""  0.45 40000 44000
echo "HELDOUT_EXPAND2_ALL_DONE $(date +%H:%M:%S)"
