#!/usr/bin/env bash
# Full AttaQ-multilingual (1402 items, 7 languages) gated-ASR run.
#
# External-dataset generalisation check for the interleaving game: same protocol
# as the Lingua-SafetyBench Text-Dominant runs (independent reconstruction judge
# + Qwen3Guard-Gen-8B safety judge, gated ASR = equivalent AND strict-unsafe),
# on a dataset we did not design the method around. AttaQ ships 7 aligned
# languages, so the language-load sweep tops out at n=7.
# Single GPU: vLLM generation, then the HF judges sequentially.
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache
set -uo pipefail

# .vllm_env carries transformers 4.57.1 (the system one was downgraded to 4.46.3,
# which rejects the `dtype=` kwarg the judges pass), so it drives judges too.
VPY=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python
DATA=${DATA:-/home/ubuntu/342/jinkwon/datasets/attaq/attaq_full_aligned.jsonl}
W=${W:-private_artifacts/attaq_full}
SUMMARY=${SUMMARY:-results/attaq_full_summary.json}
GPU=${GPU:-1}
mkdir -p "$W"
log(){ echo "[$(date +%H:%M:%S)] ATTAQ: $*"; }

log "items: $(wc -l < "$DATA")  gpu: $GPU"

if [ ! -f "$W/target_outputs.jsonl" ]; then
  log "[1/4] generation (vLLM): english + 6 translations + n2/4/6/7 x ordered/shuffled"
  CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/run_attaq_gated_vllm.py \
    --data "$DATA" --outdir "$W/gen" \
    --interleave-ns 2 4 6 7 --interleave-conditions ordered shuffled \
    --gpu-memory-utilization 0.90 > "$W/gen.log" 2>&1 || { log "GEN FAILED"; tail -20 "$W/gen.log"; exit 1; }
  cp "$W/gen/restricted_target_outputs.jsonl" "$W/target_outputs.jsonl"
  chmod 600 "$W/target_outputs.jsonl"
  log "generated $(wc -l < "$W/target_outputs.jsonl") rows"
fi

if [ ! -f "$W/recon/restricted_reconstruction_audit.jsonl" ]; then
  log "[2/4] reconstruction judge"
  CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/judge_reconstruction_equivalence.py \
    --input "$W/target_outputs.jsonl" --outdir "$W/recon" \
    --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --batch-size 48 > "$W/recon.log" 2>&1 \
    || { log "RECON FAILED"; tail -20 "$W/recon.log"; exit 1; }
fi

if [ ! -f "$W/guard/restricted_qwen3guard_audit.jsonl" ]; then
  log "[3/4] safety judge (Qwen3Guard-Gen-8B)"
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
