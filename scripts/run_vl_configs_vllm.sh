#!/usr/bin/env bash
# Configs B (VL+real image) and C (VL+blank image) via vLLM (fast path).
# Generation runs in the isolated .vllm_env; judges run in the system HF env.
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache
set -uo pipefail
VPY=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python
GAME=private_artifacts/full_textdom/slot_game.jsonl
BLANK=/home/ubuntu/342/jinkwon/datasets/blank_gray.png
log(){ echo "[$(date +%H:%M:%S)] $*"; }

gen_vllm(){  # $1=workdir  $2=extra args (e.g. --blank-image ...)
  local W="$1"; shift; local EXTRA="$*"
  CUDA_VISIBLE_DEVICES=0 $VPY scripts/run_vl_gated_vllm.py --data "$GAME" \
    --outdir "$W/gen_shard0" --interleave-ns 4 10 --translated-langs Finnish \
    --gpu-memory-utilization 0.90 --shard 0/2 $EXTRA > "$W/gen0.log" 2>&1 &
  local P0=$!
  CUDA_VISIBLE_DEVICES=1 $VPY scripts/run_vl_gated_vllm.py --data "$GAME" \
    --outdir "$W/gen_shard1" --interleave-ns 4 10 --translated-langs Finnish \
    --gpu-memory-utilization 0.90 --shard 1/2 $EXTRA > "$W/gen1.log" 2>&1 &
  local P1=$!; wait $P0 $P1
  cat "$W"/gen_shard0/restricted_target_outputs.jsonl "$W"/gen_shard1/restricted_target_outputs.jsonl > "$W/target_outputs.jsonl"
  chmod 600 "$W/target_outputs.jsonl"
  log "  $W merged $(wc -l < "$W/target_outputs.jsonl") gens"
}

judge_and_summary(){  # $1=workdir  $2=summary  $3=baseline
  local W="$1" S="$2" BASE="$3"
  CUDA_VISIBLE_DEVICES=0 python3 scripts/judge_reconstruction_equivalence.py \
    --input "$W/target_outputs.jsonl" --outdir "$W/recon" \
    --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --batch-size 48 > "$W/recon.log" 2>&1 &
  local A=$!
  CUDA_VISIBLE_DEVICES=1 python3 scripts/rejudge_qwen3guard_official.py \
    --input "$W/target_outputs.jsonl" --outdir "$W/guard" \
    --model Qwen/Qwen3Guard-Gen-8B --device cuda:0 --batch-size 48 \
    --assistant-field answer_section > "$W/guard.log" 2>&1 &
  local B=$!; wait $A $B
  python3 scripts/summarize_polyjig_comparison.py \
    --recon-audit "$W/recon/restricted_reconstruction_audit.jsonl" \
    --guard-audit "$W/guard/restricted_qwen3guard_audit.jsonl" \
    --baseline "$BASE" --output "$S" > "$W/summary.log" 2>&1
  python3 scripts/package_jailbreak_cases.py --target "$W/target_outputs.jsonl" \
    --recon-audit "$W/recon/restricted_reconstruction_audit.jsonl" \
    --guard-audit "$W/guard/restricted_qwen3guard_audit.jsonl" --outdir "$W/cases" > /dev/null 2>&1
  log "  summary -> $S"
}

W=private_artifacts/full_textdom/vl_real_vllm; mkdir -p "$W"
log "CONFIG B (vLLM VL + real image) gen"; gen_vllm "$W"
log "CONFIG B judging"; judge_and_summary "$W" results/full_textdom_vlreal_summary.json vl_english_direct

W=private_artifacts/full_textdom/vl_blank_vllm; mkdir -p "$W"
log "CONFIG C (vLLM VL + blank image) gen"; gen_vllm "$W" --blank-image "$BLANK"
log "CONFIG C judging"; judge_and_summary "$W" results/full_textdom_vlblank_summary.json vl_english_direct
log "DONE vLLM configs B+C"
