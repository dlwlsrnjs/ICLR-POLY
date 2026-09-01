#!/usr/bin/env bash
# Text-Dominant comparison on the VL target (Qwen2.5-VL-7B, images attached),
# to compare against the Qwen2.5-7B text-model run. Both GPUs, item-sharded.
set -euo pipefail
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache

DATA=/home/ubuntu/342/jinkwon/datasets/lingua_safetybench_text/vl_textdom_game320.jsonl
WORK=private_artifacts/textdom_vl320
mkdir -p "$WORK"

echo "[1/4] VL target generation (images attached), sharded GPU0 + GPU1"
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_vl_gated.py --data "$DATA" \
  --outdir "$WORK/gen_shard0" --interleave-ns 4 10 \
  --translated-langs Chinese Arabic Finnish --shard 0/2 --batch-size 12 > "$WORK/gen0.log" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python3 scripts/run_vl_gated.py --data "$DATA" \
  --outdir "$WORK/gen_shard1" --interleave-ns 4 10 \
  --translated-langs Chinese Arabic Finnish --shard 1/2 --batch-size 12 > "$WORK/gen1.log" 2>&1 &
P1=$!
wait $P0 $P1

echo "[2/4] merge shards"
cat "$WORK/gen_shard0/restricted_target_outputs.jsonl" \
    "$WORK/gen_shard1/restricted_target_outputs.jsonl" > "$WORK/target_outputs.jsonl"
chmod 600 "$WORK/target_outputs.jsonl"
echo "merged $(wc -l < "$WORK/target_outputs.jsonl") generations"

echo "[3/4] judges in parallel: reconstruction (GPU0) + Qwen3Guard (GPU1)"
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

echo "[4/4] summarize (gated + per-scenario) + package cases"
python3 scripts/summarize_polyjig_comparison.py \
  --recon-audit "$WORK/recon/restricted_reconstruction_audit.jsonl" \
  --guard-audit "$WORK/guard/restricted_qwen3guard_audit.jsonl" \
  --baseline vl_english_direct \
  --output results/qwen_vl_textdom320_summary.json
python3 scripts/package_jailbreak_cases.py \
  --target "$WORK/target_outputs.jsonl" \
  --recon-audit "$WORK/recon/restricted_reconstruction_audit.jsonl" \
  --guard-audit "$WORK/guard/restricted_qwen3guard_audit.jsonl" \
  --outdir "$WORK/cases"
echo "DONE textdom_vl320"
