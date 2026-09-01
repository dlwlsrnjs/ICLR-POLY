#!/usr/bin/env bash
# Serial GPU queue for the 2026-09-01 session. Only GPU 1 is free on this box
# (GPU 0 is held by another user's job), so every stage runs one at a time.
#
#   0. wait for the interleaving language-load x ordering curve to finish
#   1. AttaQ-multilingual full (1402 items) — external-dataset generalisation
#   2. MD-Judge-v0.1 cross-validation of the curve rows — judge invariance
#   3. Mistral-7B-Instruct target — target-model generalisation (non-Qwen)
#   4. VL + real image shard 0 — repairs the coverage hole from the 0-byte PNG
#
# Each stage is skipped if its output already exists, so the script is resumable.
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache
set -uo pipefail

VPY=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python
GPU=${GPU:-1}
CURVE=results/qwen_interleaving_curve_summary.json
log(){ echo "[$(date +%H:%M:%S)] QUEUE: $*"; }

# ---- 0. wait for the curve run ----
if [ ! -f "$CURVE" ]; then
  log "waiting for $CURVE"
  until [ -f "$CURVE" ]; do sleep 60; done
fi
log "curve ready"

# ---- 1. AttaQ full ----
if [ ! -f results/attaq_full_summary.json ]; then
  log "stage 1: AttaQ full (1402 items x 15 conditions)"
  GPU=$GPU bash scripts/run_attaq_full.sh
  log "stage 1 done"
fi

# ---- 2. MD-Judge cross-validation on the curve rows ----
CW=private_artifacts/interleaving_curve
if [ ! -f results/judge_crossvalidation_curve_summary.json ]; then
  log "stage 2: MD-Judge-v0.1 on curve rows"
  if [ ! -f "$CW/mdjudge/restricted_mdjudge_audit.jsonl" ]; then
    CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/judge_mdjudge.py \
      --input "$CW/target_outputs.jsonl" --outdir "$CW/mdjudge" \
      --assistant-field answer_section --gpu-memory-utilization 0.90 \
      > "$CW/mdjudge.log" 2>&1 || { log "MDJUDGE FAILED"; tail -20 "$CW/mdjudge.log"; }
  fi
  python3 scripts/compare_judges.py \
    --recon-audit "$CW/recon/restricted_reconstruction_audit.jsonl" \
    --guard-audit "$CW/guard/restricted_qwen3guard_audit.jsonl" \
    --mdjudge-audit "$CW/mdjudge/restricted_mdjudge_audit.jsonl" \
    --baseline english_direct \
    --output results/judge_crossvalidation_curve_summary.json > "$CW/judge_compare.log" 2>&1
  log "stage 2 done"
fi

# ---- 3. Mistral-7B-Instruct as an independent target model ----
MW=private_artifacts/mistral_textdom320
GAME320=private_artifacts/textdom320/slot_game.jsonl
if [ ! -f results/mistral_textdom320_summary.json ]; then
  log "stage 3: Mistral-7B-Instruct-v0.3 target on the 320 stratified set"
  mkdir -p "$MW"
  if [ ! -f "$MW/target_outputs.jsonl" ]; then
    CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/run_text_gated_vllm.py \
      --data "$GAME320" --outdir "$MW/gen" \
      --target mistralai/Mistral-7B-Instruct-v0.3 \
      --interleave-ns 4 10 --interleave-conditions ordered shuffled \
      --translated-langs Finnish --no-slots \
      --gpu-memory-utilization 0.90 > "$MW/gen.log" 2>&1 \
      && { cp "$MW/gen/restricted_target_outputs.jsonl" "$MW/target_outputs.jsonl"; chmod 600 "$MW/target_outputs.jsonl"; } \
      || { log "MISTRAL GEN FAILED"; tail -20 "$MW/gen.log"; }
  fi
  if [ -f "$MW/target_outputs.jsonl" ]; then
    CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/judge_reconstruction_equivalence.py \
      --input "$MW/target_outputs.jsonl" --outdir "$MW/recon" \
      --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --batch-size 48 > "$MW/recon.log" 2>&1
    CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/rejudge_qwen3guard_official.py \
      --input "$MW/target_outputs.jsonl" --outdir "$MW/guard" \
      --model Qwen/Qwen3Guard-Gen-8B --device cuda:0 --batch-size 48 \
      --assistant-field answer_section > "$MW/guard.log" 2>&1
    CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/judge_mdjudge.py \
      --input "$MW/target_outputs.jsonl" --outdir "$MW/mdjudge" \
      --assistant-field answer_section --gpu-memory-utilization 0.90 > "$MW/mdjudge.log" 2>&1
    python3 scripts/summarize_polyjig_comparison.py \
      --recon-audit "$MW/recon/restricted_reconstruction_audit.jsonl" \
      --guard-audit "$MW/guard/restricted_qwen3guard_audit.jsonl" \
      --baseline english_direct --output results/mistral_textdom320_summary.json > "$MW/summary.log" 2>&1
    python3 scripts/compare_judges.py \
      --recon-audit "$MW/recon/restricted_reconstruction_audit.jsonl" \
      --guard-audit "$MW/guard/restricted_qwen3guard_audit.jsonl" \
      --mdjudge-audit "$MW/mdjudge/restricted_mdjudge_audit.jsonl" \
      --baseline english_direct \
      --output results/judge_crossvalidation_mistral320_summary.json > "$MW/judge_compare.log" 2>&1
  fi
  log "stage 3 done"
fi

# ---- 4. repair config B (VL + real image): the missing shard 0 ----
VW=private_artifacts/full_textdom/vl_real_vllm
if [ ! -f results/full_textdom_vlreal_repaired_summary.json ]; then
  log "stage 4: VL+real image shard 0 (image validation now skips the 0-byte PNG)"
  if [ ! -f "$VW/gen_shard0/restricted_target_outputs.jsonl" ]; then
    CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/run_vl_gated_vllm.py \
      --data private_artifacts/full_textdom/slot_game.jsonl \
      --outdir "$VW/gen_shard0" --interleave-ns 4 10 --translated-langs Finnish \
      --gpu-memory-utilization 0.90 --shard 0/2 > "$VW/gen0_repair.log" 2>&1 \
      || { log "VL SHARD0 FAILED"; tail -20 "$VW/gen0_repair.log"; }
  fi
  if [ -f "$VW/gen_shard0/restricted_target_outputs.jsonl" ]; then
    cat "$VW/gen_shard0/restricted_target_outputs.jsonl" \
        "$VW/gen_shard1/restricted_target_outputs.jsonl" > "$VW/target_outputs_full.jsonl"
    chmod 600 "$VW/target_outputs_full.jsonl"
    log "  merged $(wc -l < "$VW/target_outputs_full.jsonl") gens"
    CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/judge_reconstruction_equivalence.py \
      --input "$VW/target_outputs_full.jsonl" --outdir "$VW/recon_full" \
      --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --batch-size 48 > "$VW/recon_full.log" 2>&1
    CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/rejudge_qwen3guard_official.py \
      --input "$VW/target_outputs_full.jsonl" --outdir "$VW/guard_full" \
      --model Qwen/Qwen3Guard-Gen-8B --device cuda:0 --batch-size 48 \
      --assistant-field answer_section > "$VW/guard_full.log" 2>&1
    python3 scripts/summarize_polyjig_comparison.py \
      --recon-audit "$VW/recon_full/restricted_reconstruction_audit.jsonl" \
      --guard-audit "$VW/guard_full/restricted_qwen3guard_audit.jsonl" \
      --baseline vl_english_direct \
      --output results/full_textdom_vlreal_repaired_summary.json > "$VW/summary_full.log" 2>&1
  fi
  log "stage 4 done"
fi

log "QUEUE COMPLETE"
