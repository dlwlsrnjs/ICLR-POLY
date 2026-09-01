#!/usr/bin/env bash
# Frontier target (GPT-4o-mini) on Lingua, headline conditions, TEST split only.
# Generation via OpenAI API (CPU); judges run on GPU (Qwen3Guard/MD/HR + recon gate).
# Requires OPENAI_API_KEY in the environment. Cost-controlled (see MAX_ITEMS).
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache
set -uo pipefail
VPY=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python
GPU=${GPU:-1}
MODEL=${MODEL:-gpt-4o-mini}
DATA=${DATA:-private_artifacts/paper_main/lingua_csrtmt.jsonl}
[ -f "$DATA" ] || DATA=private_artifacts/full_textdom/slot_game.jsonl
SPLIT=${SPLIT:-test}
MAXITEMS=${MAXITEMS:-400}          # cost cap; set to 0 for full split
W=private_artifacts/frontier_${MODEL//\//_}
mkdir -p "$W"
log(){ echo "[$(date +%H:%M:%S)] FRONTIER: $*"; }

if [ -z "${OPENAI_API_KEY:-}" ]; then log "OPENAI_API_KEY not set; aborting"; exit 2; fi

CAP=""; [ "$MAXITEMS" != "0" ] && CAP="--max-items $MAXITEMS"

if [ ! -f "$W/target_outputs.jsonl" ]; then
  log "[1/3] generation ($MODEL, split=$SPLIT, cap=$MAXITEMS): headline conditions"
  $VPY scripts/run_openai_gated.py --data "$DATA" --outdir "$W/gen" --model "$MODEL" \
    --interleave-ns 4 10 --interleave-conditions ordered \
    --translated-langs Finnish --with-csrt --csrt-ks 2 --with-nogame \
    --split-file experiments/lingua_splits.json --split "$SPLIT" $CAP \
    --concurrency 8 > "$W/gen.log" 2>&1 || { log "GEN FAILED"; tail -20 "$W/gen.log"; exit 1; }
  cp "$W/gen/restricted_target_outputs.jsonl" "$W/target_outputs.jsonl"; chmod 600 "$W/target_outputs.jsonl"
  log "generated $(wc -l < "$W/target_outputs.jsonl") rows"
fi

log "[2/3] judges (recon + Qwen3Guard + MD-Judge + HR) on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/judge_reconstruction_equivalence.py \
  --input "$W/target_outputs.jsonl" --outdir "$W/recon" \
  --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --batch-size 48 > "$W/recon.log" 2>&1
CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/rejudge_qwen3guard_official.py \
  --input "$W/target_outputs.jsonl" --outdir "$W/guard" \
  --model Qwen/Qwen3Guard-Gen-8B --device cuda:0 --batch-size 48 \
  --assistant-field answer_section > "$W/guard.log" 2>&1
CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/judge_mdjudge.py \
  --input "$W/target_outputs.jsonl" --outdir "$W/mdjudge" \
  --assistant-field answer_section --gpu-memory-utilization 0.90 > "$W/mdjudge.log" 2>&1
CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/judge_harmfulness_rating.py \
  --input "$W/target_outputs.jsonl" --outdir "$W/hr" \
  --model Qwen/Qwen2.5-7B-Instruct --assistant-field answer_section \
  --gpu-memory-utilization 0.90 > "$W/hr.log" 2>&1

log "[3/3] method comparison"
python3 scripts/compare_methods.py \
  --recon-audit "$W/recon/restricted_reconstruction_audit.jsonl" \
  --guard-audit "$W/guard/restricted_qwen3guard_audit.jsonl" \
  --mdjudge-audit "$W/mdjudge/restricted_mdjudge_audit.jsonl" \
  --hr-audit "$W/hr/restricted_hr_audit.jsonl" \
  --output "results/paper_frontier_${MODEL//\//_}_method_comparison.json" > "$W/summary.log" 2>&1
log "DONE -> results/paper_frontier_${MODEL//\//_}_method_comparison.json"
