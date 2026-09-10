#!/usr/bin/env bash
# Capability-decomposition probe for every panel target, after the main queues finish.
# args: GPU "tag=MODEL[::EXTRA] ..."
set -uo pipefail
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
VPY=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python
GPU="$1"; SPEC="$2"
CB=private_artifacts/capability_probe; mkdir -p $CB
log(){ echo "[$(date +%H:%M:%S)] cap-gpu$GPU: $*"; }
until grep -q "ALL SCORING DONE" private_artifacts/alignment_probe/scoring_v2.log 2>/dev/null; do sleep 30; done
log "alignment scoring finished; starting capability probes"
for entry in $SPEC; do
  tag="${entry%%=*}"; rest="${entry#*=}"; model="${rest%%::*}"; extra=""
  [ "$rest" != "$model" ] && extra="${rest#*::}"
  W=$CB/$tag; mkdir -p "$W"
  if [ ! -f "$W/capability_probe_outputs.jsonl" ]; then
    log "$tag generate"
    CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/run_capability_probe.py \
      --data private_artifacts/panel_v2/benign_probe.jsonl --outdir "$W" --model "$model" $extra \
      --gpu-memory-utilization 0.90 > "$W/gen.log" 2>&1 \
      || { log "$tag GEN FAILED"; tail -5 "$W/gen.log"; continue; }
  fi
  if [ ! -f "$W/judge/restricted_reconstruction_audit.jsonl" ]; then
    log "$tag judge"
    CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/judge_reconstruction_equivalence.py \
      --input "$W/capability_probe_outputs.jsonl" --outdir "$W/judge" \
      --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --batch-size 64 > "$W/judge.log" 2>&1 \
      || { log "$tag JUDGE FAILED"; tail -5 "$W/judge.log"; continue; }
  fi
  log "$tag done"
done
log "CAPABILITY QUEUE COMPLETE"
