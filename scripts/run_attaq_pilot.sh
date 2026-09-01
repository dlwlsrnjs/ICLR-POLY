#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache
DATA=/home/ubuntu/342/jinkwon/datasets/attaq/attaq_sample1_aligned.jsonl
WORK=private_artifacts/attaq_sample1
mkdir -p "$WORK"

echo "[1/4] generation, sharded GPU0 + GPU1"
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_attaq_gated.py --data "$DATA" \
  --outdir "$WORK/gen0" --interleave-ns 4 7 --shard 0/2 --batch-size 24 > "$WORK/gen0.log" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python3 scripts/run_attaq_gated.py --data "$DATA" \
  --outdir "$WORK/gen1" --interleave-ns 4 7 --shard 1/2 --batch-size 24 > "$WORK/gen1.log" 2>&1 &
P1=$!
wait $P0 $P1
cat "$WORK/gen0/restricted_target_outputs.jsonl" "$WORK/gen1/restricted_target_outputs.jsonl" > "$WORK/target_outputs.jsonl"
chmod 600 "$WORK/target_outputs.jsonl"; echo "merged $(wc -l < "$WORK/target_outputs.jsonl")"

echo "[2/4] judges parallel"
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

echo "[3/4] summarize"
python3 scripts/summarize_polyjig_comparison.py \
  --recon-audit "$WORK/recon/restricted_reconstruction_audit.jsonl" \
  --guard-audit "$WORK/guard/restricted_qwen3guard_audit.jsonl" \
  --output results/attaq_sample1_summary.json

echo "[4/4] package cases"
python3 scripts/package_jailbreak_cases.py --target "$WORK/target_outputs.jsonl" \
  --recon-audit "$WORK/recon/restricted_reconstruction_audit.jsonl" \
  --guard-audit "$WORK/guard/restricted_qwen3guard_audit.jsonl" --outdir "$WORK/cases"
echo "DONE attaq_sample1"
