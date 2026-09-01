#!/usr/bin/env bash
# Language-load x ordering curve for the interleaving game.
#
# Fills the ablation the reports called for: n = 2/4/6/8/10 languages crossed
# with ordered vs shuffled fragment order, on the full Text-Dominant dev set,
# anchored by english_direct and the strongest translation baseline (Finnish).
# Single GPU (GPU 1): vLLM for generation, HF judges sequentially afterwards.
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache
set -uo pipefail

# NOTE: the system python's transformers was downgraded to 4.46.3, which no longer
# accepts the `dtype=` kwarg the judge scripts pass. .vllm_env carries transformers
# 4.57.1 + torch 2.8.0+cu128, so it drives both vLLM generation and the HF judges.
VPY=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python
GAME=${GAME:-private_artifacts/full_textdom/slot_game.jsonl}
W=${W:-private_artifacts/interleaving_curve}
SUMMARY=${SUMMARY:-results/qwen_interleaving_curve_summary.json}
GPU=${GPU:-1}
mkdir -p "$W"

log(){ echo "[$(date +%H:%M:%S)] CURVE: $*"; }

log "items: $(wc -l < "$GAME")  game: $GAME  gpu: $GPU"

if [ ! -f "$W/target_outputs.jsonl" ]; then
  log "[1/4] generation (vLLM): n2/4/6/8/10 x ordered/shuffled + english + Finnish"
  CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/run_text_gated_vllm.py \
    --data "$GAME" --outdir "$W/gen" \
    --interleave-ns 2 4 6 8 10 \
    --interleave-conditions ordered shuffled \
    --translated-langs Finnish --no-slots \
    --gpu-memory-utilization 0.90 > "$W/gen.log" 2>&1 || { log "GEN FAILED"; tail -20 "$W/gen.log"; exit 1; }
  cp "$W/gen/restricted_target_outputs.jsonl" "$W/target_outputs.jsonl"
  chmod 600 "$W/target_outputs.jsonl"
  log "generated $(wc -l < "$W/target_outputs.jsonl") rows"
fi

if [ ! -f "$W/recon/restricted_reconstruction_audit.jsonl" ]; then
  log "[2/4] reconstruction judge (Qwen2.5-7B)"
  CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/judge_reconstruction_equivalence.py \
    --input "$W/target_outputs.jsonl" --outdir "$W/recon" \
    --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --batch-size 48 > "$W/recon.log" 2>&1 \
    || { log "RECON FAILED"; tail -20 "$W/recon.log"; exit 1; }
fi

if [ ! -f "$W/guard/restricted_qwen3guard_audit.jsonl" ]; then
  log "[3/4] safety judge (Qwen3Guard-Gen-8B, official [ANSWER]-only protocol)"
  CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/rejudge_qwen3guard_official.py \
    --input "$W/target_outputs.jsonl" --outdir "$W/guard" \
    --model Qwen/Qwen3Guard-Gen-8B --device cuda:0 --batch-size 48 \
    --assistant-field answer_section > "$W/guard.log" 2>&1 \
    || { log "GUARD FAILED"; tail -20 "$W/guard.log"; exit 1; }
fi

log "[4/4] summarize"
python3 scripts/summarize_polyjig_comparison.py \
  --recon-audit "$W/recon/restricted_reconstruction_audit.jsonl" \
  --guard-audit "$W/guard/restricted_qwen3guard_audit.jsonl" \
  --baseline english_direct --output "$SUMMARY" > "$W/summary.log" 2>&1
python3 scripts/package_jailbreak_cases.py --target "$W/target_outputs.jsonl" \
  --recon-audit "$W/recon/restricted_reconstruction_audit.jsonl" \
  --guard-audit "$W/guard/restricted_qwen3guard_audit.jsonl" --outdir "$W/cases" > /dev/null 2>&1
log "DONE -> $SUMMARY"
