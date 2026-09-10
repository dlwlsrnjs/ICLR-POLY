#!/bin/bash
# Panel + held-out DrAttack/FlipAttack under reconstruction-gated verified scoring, so they become
# headline rows (Table tab:main). Two-phase (gen target-only / judge judges-only) so every size fits one
# card. GPU-guarded + idempotent. Chained LAST (after the Qwen3-8B held-out job, which is after the live
# query-eff job, which is after the expansion) so nothing contends for the one freed GPU.
# After it finishes: sudo -n -u jinkwon python3 scripts/sota_baselines_means.py  (emits macros), then add
# the two rows to paper/tab_main_combined.tex.
set +e
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ENFORCE_EAGER=1
export VLLM_USE_FLASHINFER_SAMPLER=0
MJ_HARM=private_artifacts/multijail_v1/harm_grid.jsonl
LG_HARM=private_artifacts/panel_v2/harm_grid.jsonl
MJ_OUT=results/sota_mj_20260909; LG_OUT=results/sota_lg_20260909

pick_gpu(){ need=$1; for w in $(seq 1 5760); do
  while read idx free; do [ "$free" -ge "$need" ] && { echo "$idx"; return 0; }; done \
    < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  echo "[wait $(date +%H:%M:%S)] no GPU with ${need}MB free" >&2; sleep 30; done; return 1; }
run2p(){ local extra="$1" harm="$2" nit="$3" outdir="$4" tag="$5" needg="$6" needj="$7"
  local outfile="$outdir/$tag.json"
  [ -f "$outfile" ] && { echo "OK $outfile"; return 0; }
  local g; g=$(pick_gpu "$needg")||{ echo "no GPU $outfile"; return 1; }
  echo "[$(date +%H:%M:%S)] GEN GPU$g -> $outfile"
  for t in 1 2 3; do CUDA_VISIBLE_DEVICES=$g $VP/python scripts/sota_baselines_eval.py --phase gen $extra --harm $harm --n-items $nit --outdir $outdir && break; sleep 15; done
  g=$(pick_gpu "$needj")||return 1
  echo "[$(date +%H:%M:%S)] JUDGE GPU$g -> $outfile"
  for t in 1 2 3; do CUDA_VISIBLE_DEVICES=$g $VP/python scripts/sota_baselines_eval.py --phase judge $extra --harm $harm --n-items $nit --outdir $outdir && [ -f "$outfile" ] && return 0; sleep 15; done; }
# $1 tag  $2 ref  $3 nothinking-flag  $4 util  $5 needg  $6 needj  $7 mml
do_sota(){ local tag="$1" ref="$2" nt="$3" util="$4" ng="$5" nj="$6" mml="${7:-4096}"
  local C="--target $ref --tag $tag --trust-remote-code $nt --tokenizer-mode auto --util $util --max-model-len $mml"
  echo "======== SOTA $tag ($ref) ========"
  run2p "$C" "$MJ_HARM" 64 "$MJ_OUT" "$tag" "$ng" "$nj"
  run2p "$C" "$LG_HARM" 40 "$LG_OUT" "$tag" "$ng" "$nj"
  echo "SOTA_DONE $tag"
}

# --- chain LAST: wait for the Qwen3-8B held-out job to finish ---
echo "[$(date +%H:%M:%S)] sota-baselines queued; waiting for the Qwen3-8B held-out job to finish"
while :; do
  grep -q QWEN3_8B_DONE results/qwen3_8b.log 2>/dev/null && { echo "[$(date +%H:%M:%S)] qwen3 marker seen"; break; }
  pgrep -f qwen3_8b_run.sh >/dev/null 2>&1 || { echo "[$(date +%H:%M:%S)] qwen3 job gone; proceeding"; break; }
  sleep 60
done

echo "==== PANEL (held-in, 9) ===="
do_sota qwen25_3b     Qwen/Qwen2.5-3B-Instruct               ""             0.45 40000 44000
do_sota qwen25_7b     Qwen/Qwen2.5-7B-Instruct               ""             0.45 40000 44000
do_sota qwen25_14b    Qwen/Qwen2.5-14B-Instruct              ""             0.55 46000 44000
do_sota qwen25_32b    Qwen/Qwen2.5-32B-Instruct              ""             0.90 74000 44000 3072
do_sota llama32_3b_it meta-llama/Llama-3.2-3B-Instruct       ""             0.45 40000 44000
do_sota llama31_8b_it meta-llama/Llama-3.1-8B-Instruct       ""             0.45 40000 44000
do_sota gemma2_2b_it  google/gemma-2-2b-it                   ""             0.45 40000 44000
do_sota gemma2_9b_it  google/gemma-2-9b-it                   ""             0.55 46000 44000
do_sota gemma2_27b    google/gemma-2-27b-it                  ""             0.90 74000 44000 3072
echo "==== HELD-OUT (10) ===="
do_sota phi35_mini    microsoft/Phi-3.5-mini-instruct        ""             0.45 40000 44000
do_sota mistral7b     mistralai/Mistral-7B-Instruct-v0.3     ""             0.45 40000 44000
do_sota falcon3_7b    tiiuae/Falcon3-7B-Instruct             ""             0.45 40000 44000
do_sota glm4_9b       THUDM/glm-4-9b-chat-hf                 ""             0.55 46000 44000
do_sota mistral24b    mistralai/Mistral-Small-24B-Instruct-2501 ""          0.90 74000 44000 3072
do_sota yi34b         01-ai/Yi-1.5-34B-Chat                  ""             0.90 74000 44000 3072
do_sota olmo2_7b      allenai/OLMo-2-1124-7B-Instruct        ""             0.45 40000 44000
do_sota zephyr7b      HuggingFaceH4/zephyr-7b-beta           ""             0.45 40000 44000
do_sota qwen25_15b    Qwen/Qwen2.5-1.5B-Instruct             ""             0.30 24000 44000
do_sota qwen3_8b      Qwen/Qwen3-8B                          "--no-thinking" 0.45 40000 44000
echo "SOTA_ALL_DONE $(date +%H:%M:%S)"
echo "NEXT: sudo -n -u jinkwon python3 scripts/sota_baselines_means.py  then add DrAttack/FlipAttack rows to paper/tab_main_combined.tex."
