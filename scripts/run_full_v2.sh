#!/usr/bin/env bash
# Full Text-Dominant dev (2879) across 4 configs, for HF-vs-vLLM and text-vs-VL:
#   A_vllm : text model, vLLM
#   B_vllm : VL + real image, vLLM
#   C_vllm : VL + blank image, vLLM
#   A_hf   : text model, HF (non-vLLM) — small batch to avoid the earlier hang
# vLLM configs run first (fast); HF text config last (slow). Judges are HF.
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache
set -uo pipefail
VPY=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python
GAME=private_artifacts/full_textdom/slot_game.jsonl
BLANK=/home/ubuntu/342/jinkwon/datasets/blank_gray.png
log(){ echo "[$(date +%H:%M:%S)] $*"; }

judge_and_summary(){  # $1=workdir  $2=summary  $3=baseline
  local W="$1" S="$2" BASE="$3"
  cat "$W"/gen_shard0/restricted_target_outputs.jsonl "$W"/gen_shard1/restricted_target_outputs.jsonl > "$W/target_outputs.jsonl"
  chmod 600 "$W/target_outputs.jsonl"
  log "  $(basename "$W") merged $(wc -l < "$W/target_outputs.jsonl") gens; judging"
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

vllm_gen(){  # $1=workdir $2=runner $3=baseline-cond-unused $4...=extra
  local W="$1" RUNNER="$2"; shift 3; local EXTRA="$*"
  CUDA_VISIBLE_DEVICES=0 $VPY scripts/$RUNNER --data "$GAME" --outdir "$W/gen_shard0" \
    --interleave-ns 4 10 --translated-langs Finnish --gpu-memory-utilization 0.90 --shard 0/2 $EXTRA > "$W/gen0.log" 2>&1 &
  local P0=$!
  CUDA_VISIBLE_DEVICES=1 $VPY scripts/$RUNNER --data "$GAME" --outdir "$W/gen_shard1" \
    --interleave-ns 4 10 --translated-langs Finnish --gpu-memory-utilization 0.90 --shard 1/2 $EXTRA > "$W/gen1.log" 2>&1 &
  local P1=$!; wait $P0 $P1
  # vLLM 0.11 can orphan EngineCore workers on exit; clear them so judges get clean GPUs.
  pkill -9 -f "EngineCore" 2>/dev/null; pkill -9 -f "$RUNNER" 2>/dev/null; sleep 6
}

# ---- A_vllm: text model on vLLM ----
if [ ! -f results/full_textdom_textmodel_vllm_summary.json ]; then
  W=private_artifacts/full_textdom/text_vllm; mkdir -p "$W"
  log "A_vllm (text, vLLM) gen"; vllm_gen "$W" run_text_gated_vllm.py x
  judge_and_summary "$W" results/full_textdom_textmodel_vllm_summary.json english_direct
fi

# ---- B_vllm: VL + real image ----
if [ ! -f results/full_textdom_vlreal_summary.json ]; then
  W=private_artifacts/full_textdom/vl_real_vllm; mkdir -p "$W"
  log "B_vllm (VL real image) gen"; vllm_gen "$W" run_vl_gated_vllm.py x
  judge_and_summary "$W" results/full_textdom_vlreal_summary.json vl_english_direct
fi

# ---- C_vllm: VL + blank image ----
if [ ! -f results/full_textdom_vlblank_summary.json ]; then
  W=private_artifacts/full_textdom/vl_blank_vllm; mkdir -p "$W"
  log "C_vllm (VL blank image) gen"; vllm_gen "$W" run_vl_gated_vllm.py x --blank-image "$BLANK"
  judge_and_summary "$W" results/full_textdom_vlblank_summary.json vl_english_direct
fi

# ---- A_hf: text model on HF (small batch to avoid hang) ----
if [ ! -f results/full_textdom_textmodel_hf_summary.json ]; then
  W=private_artifacts/full_textdom/text_hf; mkdir -p "$W"
  log "A_hf (text, HF, batch 8) gen"
  CUDA_VISIBLE_DEVICES=0 python3 scripts/run_polyjig_gated.py --data "$GAME" \
    --outdir "$W/gen_shard0" --interleave-ns 4 10 --translated-langs Finnish \
    --shard 0/2 --batch-size 8 > "$W/gen0.log" 2>&1 &
  P0=$!
  CUDA_VISIBLE_DEVICES=1 python3 scripts/run_polyjig_gated.py --data "$GAME" \
    --outdir "$W/gen_shard1" --interleave-ns 4 10 --translated-langs Finnish \
    --shard 1/2 --batch-size 8 > "$W/gen1.log" 2>&1 &
  P1=$!; wait $P0 $P1
  judge_and_summary "$W" results/full_textdom_textmodel_hf_summary.json english_direct
fi

log "building combined table"
python3 scripts/compare_all_configs.py > /tmp/final_table.txt 2>&1 || true
log "ALL DONE (4 configs)"
