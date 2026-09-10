#!/usr/bin/env bash
# Launch the fragment-count factorial across 2 GPUs, 3 targets each, sequentially.
set -uo pipefail
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
ROOT=private_artifacts/frag_factorial_20260903
mkdir -p "$ROOT"
DRV=scripts/run_fragment_factorial.sh
# GPU0 queue
( for pair in "qwen25_3b:Qwen/Qwen2.5-3B-Instruct" "phi35:microsoft/Phi-3.5-mini-instruct" "qwen25_14b:Qwen/Qwen2.5-14B-Instruct"; do
    tag="${pair%%:*}"; model="${pair#*:}"
    bash "$DRV" "$tag" "$model" 0 >> "$ROOT/queue_gpu0.log" 2>&1
  done ) &
echo $! > "$ROOT/gpu0.pid"
# GPU1 queue
( for pair in "qwen25_7b:Qwen/Qwen2.5-7B-Instruct" "falcon3_7b:tiiuae/Falcon3-7B-Instruct" "qwen3_8b:Qwen/Qwen3-8B"; do
    tag="${pair%%:*}"; model="${pair#*:}"
    bash "$DRV" "$tag" "$model" 1 >> "$ROOT/queue_gpu1.log" 2>&1
  done ) &
echo $! > "$ROOT/gpu1.pid"
wait
echo "ALL FRAGMENT-FACTORIAL TARGETS DONE" >> "$ROOT/queue_gpu0.log"
