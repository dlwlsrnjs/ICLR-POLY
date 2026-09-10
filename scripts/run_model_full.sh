#!/usr/bin/env bash
# Full per-model pipeline on one GPU: panel grid (v2 inputs) + the four alignment probes.
# args: GPU "tag=MODEL[::EXTRA] tag2=MODEL2 ..."
set -uo pipefail
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
VPY=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python
GPU="$1"; SPEC="$2"
AB=private_artifacts/alignment_probe
log(){ echo "[$(date +%H:%M:%S)] gpu$GPU: $*"; }
for entry in $SPEC; do
  tag="${entry%%=*}"; rest="${entry#*=}"; model="${rest%%::*}"; extra=""
  [ "$rest" != "$model" ] && extra="${rest#*::}"
  mkdir -p "$AB/$tag"
  log "=== $tag ($model $extra) ==="
  bash scripts/run_panel_v2.sh "$tag" "$model" "$GPU" "$extra" 2>&1 | sed "s/^/  /"
  for probe in over:probe_overrefusal direct:probe_direct_harm multi:probe_direct_multi xsunsafe:probe_xstest_unsafe; do
    sub="${probe%%:*}"; file="${probe#*:}"
    od="$AB/$tag/$sub"
    if [ -f "$od/probe_outputs.jsonl" ] || [ -f "$od/restricted_probe_outputs.jsonl" ]; then
      log "$tag/$sub already done"; continue
    fi
    log "$tag/$sub"
    CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/run_alignment_probe.py --data "$AB/$file.jsonl" \
      --outdir "$od" --model "$model" $extra --gpu-memory-utilization 0.90 \
      > "$AB/$tag/${sub}.log" 2>&1 || { log "$tag/$sub FAILED"; tail -4 "$AB/$tag/${sub}.log"; }
  done
  log "=== $tag COMPLETE ==="
done
log "QUEUE COMPLETE"
