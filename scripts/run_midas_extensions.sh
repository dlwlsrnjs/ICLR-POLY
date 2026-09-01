#!/usr/bin/env bash
# MIDAS-aligned extension experiments on the 320-item stratified Text-Dominant set.
# (See docs/MIDAS_ALIGNED_DESIGN.md.) Data/model/judges stay ours; only the logic
# and experiment structure follow MIDAS.
#
#   Generation: english + Finnish + interleave{ordered,shuffled} n{2,4,6,8,10}
#               + no-game ablation of each interleave condition  (MIDAS Table 5)
#   Judges:  reconstruction (gate) + Qwen3Guard (safety) + MD-Judge (cross)
#            + Harmfulness Rating 0-5                             (MIDAS ASR+HR)
#   Input:   prompt-stage detectability gap                      (MIDAS Fig.4a)
#   Defense: best interleave condition re-run under Self-Reminder (MIDAS Table 7/8)
#
# Single GPU; every stage skipped if its output exists (resumable).
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache
set -uo pipefail
VPY=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python
GAME=${GAME:-private_artifacts/textdom320/slot_game.jsonl}
W=${W:-private_artifacts/midas_ext320}
GPU=${GPU:-1}
mkdir -p "$W"
log(){ echo "[$(date +%H:%M:%S)] MIDAS-EXT: $*"; }

# ---- 1. generation with no-game ablation ----
if [ ! -f "$W/target_outputs.jsonl" ]; then
  log "[1] generation: interleave + no-game ablation, n2/4/6/8/10 x ordered/shuffled"
  CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/run_text_gated_vllm.py \
    --data "$GAME" --outdir "$W/gen" \
    --interleave-ns 2 4 6 8 10 --interleave-conditions ordered shuffled \
    --translated-langs Finnish --no-slots --with-nogame \
    --gpu-memory-utilization 0.90 > "$W/gen.log" 2>&1 || { log "GEN FAILED"; tail -20 "$W/gen.log"; exit 1; }
  cp "$W/gen/restricted_target_outputs.jsonl" "$W/target_outputs.jsonl"; chmod 600 "$W/target_outputs.jsonl"
  log "generated $(wc -l < "$W/target_outputs.jsonl") rows"
fi

# ---- 2. judges: reconstruction, Qwen3Guard, MD-Judge, HR ----
[ -f "$W/recon/restricted_reconstruction_audit.jsonl" ] || { log "[2a] reconstruction judge";
  CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/judge_reconstruction_equivalence.py \
    --input "$W/target_outputs.jsonl" --outdir "$W/recon" \
    --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --batch-size 48 > "$W/recon.log" 2>&1; }
[ -f "$W/guard/restricted_qwen3guard_audit.jsonl" ] || { log "[2b] Qwen3Guard judge";
  CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/rejudge_qwen3guard_official.py \
    --input "$W/target_outputs.jsonl" --outdir "$W/guard" \
    --model Qwen/Qwen3Guard-Gen-8B --device cuda:0 --batch-size 48 \
    --assistant-field answer_section > "$W/guard.log" 2>&1; }
[ -f "$W/mdjudge/restricted_mdjudge_audit.jsonl" ] || { log "[2c] MD-Judge";
  CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/judge_mdjudge.py \
    --input "$W/target_outputs.jsonl" --outdir "$W/mdjudge" \
    --assistant-field answer_section --gpu-memory-utilization 0.90 > "$W/mdjudge.log" 2>&1; }
[ -f "$W/hr/restricted_hr_audit.jsonl" ] || { log "[2d] Harmfulness Rating 0-5";
  CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/judge_harmfulness_rating.py \
    --input "$W/target_outputs.jsonl" --outdir "$W/hr" \
    --model Qwen/Qwen2.5-7B-Instruct --assistant-field answer_section \
    --gpu-memory-utilization 0.90 > "$W/hr.log" 2>&1; }

# ---- 3. input-stage detectability gap ----
[ -f "$W/input_detect/summary.json" ] || { log "[3] input detectability gap";
  CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/judge_input_detectability.py \
    --input "$W/target_outputs.jsonl" --outdir "$W/input_detect" \
    --model Qwen/Qwen3Guard-Gen-8B --device cuda:0 --batch-size 32 > "$W/input_detect.log" 2>&1; }

# ---- 4. summaries (gated ASR incl. no-game + judge cross-val) ----
log "[4] summaries"
python3 scripts/summarize_polyjig_comparison.py \
  --recon-audit "$W/recon/restricted_reconstruction_audit.jsonl" \
  --guard-audit "$W/guard/restricted_qwen3guard_audit.jsonl" \
  --baseline english_direct --output results/midas_ext320_gated_summary.json > "$W/summary.log" 2>&1
python3 scripts/summarize_interleaving_curve.py \
  --summary results/midas_ext320_gated_summary.json \
  --output results/midas_ext320_curve_matrix.json >> "$W/summary.log" 2>&1 || true
python3 scripts/compare_judges.py \
  --recon-audit "$W/recon/restricted_reconstruction_audit.jsonl" \
  --guard-audit "$W/guard/restricted_qwen3guard_audit.jsonl" \
  --mdjudge-audit "$W/mdjudge/restricted_mdjudge_audit.jsonl" \
  --baseline english_direct --output results/midas_ext320_judge_crossval.json >> "$W/summary.log" 2>&1

# ---- 5. defense-prompt run (Self-Reminder) on the strongest interleave load ----
DW=$W/defense_self_reminder
if [ ! -f results/midas_ext320_defense_summary.json ]; then
  log "[5] defense: Self-Reminder system prompt, interleave ordered n4/n6"
  if [ ! -f "$DW/target_outputs.jsonl" ]; then
    CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/run_text_gated_vllm.py \
      --data "$GAME" --outdir "$DW/gen" \
      --interleave-ns 4 6 --interleave-conditions ordered \
      --translated-langs Finnish --no-slots --system-prompt self_reminder \
      --gpu-memory-utilization 0.90 > "$DW/gen.log" 2>&1 \
      && { cp "$DW/gen/restricted_target_outputs.jsonl" "$DW/target_outputs.jsonl"; chmod 600 "$DW/target_outputs.jsonl"; } \
      || { log "DEFENSE GEN FAILED"; tail -20 "$DW/gen.log"; }
  fi
  if [ -f "$DW/target_outputs.jsonl" ]; then
    CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/judge_reconstruction_equivalence.py \
      --input "$DW/target_outputs.jsonl" --outdir "$DW/recon" \
      --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --batch-size 48 > "$DW/recon.log" 2>&1
    CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/rejudge_qwen3guard_official.py \
      --input "$DW/target_outputs.jsonl" --outdir "$DW/guard" \
      --model Qwen/Qwen3Guard-Gen-8B --device cuda:0 --batch-size 48 \
      --assistant-field answer_section > "$DW/guard.log" 2>&1
    python3 scripts/summarize_polyjig_comparison.py \
      --recon-audit "$DW/recon/restricted_reconstruction_audit.jsonl" \
      --guard-audit "$DW/guard/restricted_qwen3guard_audit.jsonl" \
      --baseline english_direct --output results/midas_ext320_defense_summary.json > "$DW/summary.log" 2>&1
  fi
fi

log "MIDAS-EXT COMPLETE"
