#!/usr/bin/env bash
# Full Text-Dominant dev (2879) across 3 target configs:
#   A) text-only model (Qwen2.5-7B)
#   B) VL model + blank image (Qwen2.5-VL-7B)
#   C) VL model + real original image (Qwen2.5-VL-7B)
# Conditions per item: english_direct, translated_direct_Finnish (reference),
# interleave_ordered_n4/n10, slot_k1/k2/k3. Both GPUs, item-sharded.
# Long job (~10-12h). Each config is independent; a failure in one is logged and
# the script continues.
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache
set -uo pipefail

PILOT=/home/ubuntu/342/jinkwon/datasets/lingua_safetybench_text/textdom_dev_full.jsonl
ALIGNED=/home/ubuntu/342/jinkwon/datasets/lingua_safetybench_text/lingua_aligned_text_all.jsonl
BLANK=/home/ubuntu/342/jinkwon/datasets/blank_gray.png
ROOT=private_artifacts/full_textdom
GAME="$ROOT/slot_game.jsonl"
mkdir -p "$ROOT"

log(){ echo "[$(date +%H:%M:%S)] $*"; }

# ---- Stage A: aligner (once) ----
if [ ! -f "$ROOT/align/restricted_official_slot_alignments.jsonl" ]; then
  log "ALIGN start (2879 items)"
  CUDA_VISIBLE_DEVICES=0 python3 scripts/align_official_multilingual_slots.py \
    --pilot "$PILOT" --aligned "$ALIGNED" --outdir "$ROOT/align" \
    --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --languages-per-item 4 --batch-size 24 \
    > "$ROOT/align.log" 2>&1
  log "ALIGN done"
fi

# ---- Stage B: slot game ----
if [ ! -f "$GAME" ]; then
  log "BUILD slot game"
  python3 scripts/build_official_slot_game.py --pilot "$PILOT" \
    --alignments "$ROOT/align/restricted_official_slot_alignments.jsonl" \
    --output "$GAME" --max-k 3 > "$ROOT/build_game.log" 2>&1
fi

judges_and_summary(){  # $1=work dir  $2=summary path  $3=baseline cond
  local W="$1" S="$2" BASE="$3"
  cat "$W"/gen_shard0/restricted_target_outputs.jsonl "$W"/gen_shard1/restricted_target_outputs.jsonl > "$W/target_outputs.jsonl"
  chmod 600 "$W/target_outputs.jsonl"
  log "  merged $(wc -l < "$W/target_outputs.jsonl") gens; judging"
  CUDA_VISIBLE_DEVICES=0 python3 scripts/judge_reconstruction_equivalence.py \
    --input "$W/target_outputs.jsonl" --outdir "$W/recon" \
    --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --batch-size 48 > "$W/recon.log" 2>&1 &
  local A=$!
  CUDA_VISIBLE_DEVICES=1 python3 scripts/rejudge_qwen3guard_official.py \
    --input "$W/target_outputs.jsonl" --outdir "$W/guard" \
    --model Qwen/Qwen3Guard-Gen-8B --device cuda:0 --batch-size 48 \
    --assistant-field answer_section > "$W/guard.log" 2>&1 &
  local B=$!
  wait $A $B
  python3 scripts/summarize_polyjig_comparison.py \
    --recon-audit "$W/recon/restricted_reconstruction_audit.jsonl" \
    --guard-audit "$W/guard/restricted_qwen3guard_audit.jsonl" \
    --baseline "$BASE" --output "$S" > "$W/summary.log" 2>&1
  python3 scripts/package_jailbreak_cases.py --target "$W/target_outputs.jsonl" \
    --recon-audit "$W/recon/restricted_reconstruction_audit.jsonl" \
    --guard-audit "$W/guard/restricted_qwen3guard_audit.jsonl" --outdir "$W/cases" > /dev/null 2>&1
  log "  summary -> $S"
}

# ---- Stage C: text-only model ----
W=$ROOT/text; mkdir -p "$W"
if [ ! -f results/full_textdom_textmodel_summary.json ]; then
  log "CONFIG A (text-only Qwen2.5-7B) generation"
  CUDA_VISIBLE_DEVICES=0 python3 scripts/run_polyjig_gated.py --data "$GAME" \
    --outdir "$W/gen_shard0" --interleave-ns 4 10 --translated-langs Finnish \
    --shard 0/2 --batch-size 32 > "$W/gen0.log" 2>&1 &
  P0=$!
  CUDA_VISIBLE_DEVICES=1 python3 scripts/run_polyjig_gated.py --data "$GAME" \
    --outdir "$W/gen_shard1" --interleave-ns 4 10 --translated-langs Finnish \
    --shard 1/2 --batch-size 32 > "$W/gen1.log" 2>&1 &
  P1=$!; wait $P0 $P1
  judges_and_summary "$W" results/full_textdom_textmodel_summary.json english_direct
fi

# ---- Stage D: VL + real image ----
W=$ROOT/vl_real; mkdir -p "$W"
if [ ! -f results/full_textdom_vlreal_summary.json ]; then
  log "CONFIG B (VL + real image) generation"
  CUDA_VISIBLE_DEVICES=0 python3 scripts/run_vl_gated.py --data "$GAME" \
    --outdir "$W/gen_shard0" --interleave-ns 4 10 --translated-langs Finnish \
    --shard 0/2 --batch-size 12 > "$W/gen0.log" 2>&1 &
  P0=$!
  CUDA_VISIBLE_DEVICES=1 python3 scripts/run_vl_gated.py --data "$GAME" \
    --outdir "$W/gen_shard1" --interleave-ns 4 10 --translated-langs Finnish \
    --shard 1/2 --batch-size 12 > "$W/gen1.log" 2>&1 &
  P1=$!; wait $P0 $P1
  judges_and_summary "$W" results/full_textdom_vlreal_summary.json vl_english_direct
fi

# ---- Stage E: VL + blank image ----
W=$ROOT/vl_blank; mkdir -p "$W"
if [ ! -f results/full_textdom_vlblank_summary.json ]; then
  log "CONFIG C (VL + blank image) generation"
  CUDA_VISIBLE_DEVICES=0 python3 scripts/run_vl_gated.py --data "$GAME" \
    --outdir "$W/gen_shard0" --interleave-ns 4 10 --translated-langs Finnish \
    --blank-image "$BLANK" --shard 0/2 --batch-size 12 > "$W/gen0.log" 2>&1 &
  P0=$!
  CUDA_VISIBLE_DEVICES=1 python3 scripts/run_vl_gated.py --data "$GAME" \
    --outdir "$W/gen_shard1" --interleave-ns 4 10 --translated-langs Finnish \
    --blank-image "$BLANK" --shard 1/2 --batch-size 12 > "$W/gen1.log" 2>&1 &
  P1=$!; wait $P0 $P1
  judges_and_summary "$W" results/full_textdom_vlblank_summary.json vl_english_direct
fi

log "ALL CONFIGS DONE"
