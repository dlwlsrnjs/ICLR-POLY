#!/usr/bin/env bash
# Run both alignment probes for a list of targets on one GPU.
# usage: run_alignment_panel.sh GPU "tag=MODEL[:extra] tag2=MODEL2 ..."
set -uo pipefail
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
VPY=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python
GPU="$1"; SPEC="$2"
BASE=private_artifacts/alignment_probe
log(){ echo "[$(date +%H:%M:%S)] ALIGN-gpu$GPU: $*"; }
for entry in $SPEC; do
  tag="${entry%%=*}"; rest="${entry#*=}"; model="${rest%%::*}"; extra=""
  [ "$rest" != "$model" ] && extra="${rest#*::}"
  W="$BASE/$tag"; mkdir -p "$W"
  for probe in overrefusal direct_harm; do
    if [ "$probe" = overrefusal ]; then data=$BASE/probe_overrefusal.jsonl; done_f="$W/over/probe_outputs.jsonl"; od="$W/over"
    else data=$BASE/probe_direct_harm.jsonl; done_f="$W/direct/restricted_probe_outputs.jsonl"; od="$W/direct"; fi
    if [ -f "$done_f" ]; then log "$tag/$probe already done"; continue; fi
    log "$tag/$probe start ($model $extra)"
    CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/run_alignment_probe.py --data "$data" --outdir "$od" \
      --model "$model" $extra --gpu-memory-utilization 0.90 > "$W/${probe}.log" 2>&1 \
      || { log "$tag/$probe FAILED"; tail -5 "$W/${probe}.log"; continue; }
    log "$tag/$probe done"
  done
done
log "ALL DONE"
