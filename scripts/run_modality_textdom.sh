#!/usr/bin/env bash
# Comprehensive Text-Dominant gated-ASR run, split across both GPUs.
set -euo pipefail
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache

PILOT=/home/ubuntu/342/jinkwon/datasets/lingua_safetybench_text/slot_pilot_textdom320.jsonl
ALIGNED=/home/ubuntu/342/jinkwon/datasets/lingua_safetybench_text/lingua_aligned_text_all.jsonl
WORK=private_artifacts/textdom320
mkdir -p "$WORK"

echo "[1/6] official span alignment (GPU0)"
CUDA_VISIBLE_DEVICES=0 python3 scripts/align_official_multilingual_slots.py \
  --pilot "$PILOT" --aligned "$ALIGNED" --outdir "$WORK/align" \
  --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --languages-per-item 4 --batch-size 16

echo "[2/6] build inline-slot game prompts"
python3 scripts/build_official_slot_game.py --pilot "$PILOT" \
  --alignments "$WORK/align/restricted_official_slot_alignments.jsonl" \
  --output "$WORK/slot_game.jsonl" --max-k 3

echo "[3/6] target generation, sharded GPU0 + GPU1"
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_polyjig_gated.py --data "$WORK/slot_game.jsonl" \
  --outdir "$WORK/gen_shard0" --interleave-ns 4 10 --shard 0/2 --batch-size 24 > "$WORK/gen0.log" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python3 scripts/run_polyjig_gated.py --data "$WORK/slot_game.jsonl" \
  --outdir "$WORK/gen_shard1" --interleave-ns 4 10 --shard 1/2 --batch-size 24 > "$WORK/gen1.log" 2>&1 &
P1=$!
wait $P0 $P1

echo "[4/6] merge shards"
cat "$WORK/gen_shard0/restricted_target_outputs.jsonl" \
    "$WORK/gen_shard1/restricted_target_outputs.jsonl" > "$WORK/target_outputs.jsonl"
chmod 600 "$WORK/target_outputs.jsonl"
echo "merged $(wc -l < "$WORK/target_outputs.jsonl") generations"

echo "[5/6] judges in parallel: reconstruction (GPU0) + Qwen3Guard (GPU1)"
CUDA_VISIBLE_DEVICES=0 python3 scripts/judge_reconstruction_equivalence.py \
  --input "$WORK/target_outputs.jsonl" --outdir "$WORK/recon" \
  --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --batch-size 32 > "$WORK/recon.log" 2>&1 &
J0=$!
CUDA_VISIBLE_DEVICES=1 python3 scripts/rejudge_qwen3guard_official.py \
  --input "$WORK/target_outputs.jsonl" --outdir "$WORK/guard" \
  --model Qwen/Qwen3Guard-Gen-8B --device cuda:0 --batch-size 32 \
  --assistant-field answer_section > "$WORK/guard.log" 2>&1 &
J1=$!
wait $J0 $J1

echo "[6/6] summarize (per-scenario + aggregate) and package cases"
python3 scripts/summarize_polyjig_comparison.py \
  --recon-audit "$WORK/recon/restricted_reconstruction_audit.jsonl" \
  --guard-audit "$WORK/guard/restricted_qwen3guard_audit.jsonl" \
  --output results/qwen_gated_textdom320_summary.json
python3 scripts/package_jailbreak_cases.py \
  --target "$WORK/target_outputs.jsonl" \
  --recon-audit "$WORK/recon/restricted_reconstruction_audit.jsonl" \
  --guard-audit "$WORK/guard/restricted_qwen3guard_audit.jsonl" \
  --outdir "$WORK/cases"
echo "DONE textdom320"
