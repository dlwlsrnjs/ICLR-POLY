#!/bin/bash
# LIVE selector on HEAVY/more-aligned targets (narrower good-arm region -> query savings should show,
# unlike the easy 7B held-outs where uninformed GP also hits threshold in 1 query). Target on GPU0
# (util high), resident judges split to GPU1 (~31GB) so a 24-32B target + judges fit. Sequential
# (each run uses both cards). Idempotent.
set +e
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ENFORCE_EAGER=1 VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_VISIBLE_DEVICES=0,1   # target -> device0 (GPU0), judges -> cuda:1 (GPU1)
OUT=results/live_query_efficiency_20260909
one(){ tag="$1"; ref="$2"; util="$3"; mml="$4"
  [ -f "$OUT/$tag.json" ] && { echo "OK $tag"; return; }
  echo "[$(date +%H:%M:%S)] LIVE-HEAVY $tag (target GPU0 util $util, judges GPU1)"
  $VP/python scripts/live_query_efficiency.py --target "$ref" --tag "$tag" --exclude-target-tag "$tag" \
    --trust-remote-code --util "$util" --max-model-len "$mml" --judge-device cuda:1 \
    --budget 12 --calib 24 --outdir $OUT
  echo "[$(date +%H:%M:%S)] DONE_ONE $tag"
}
one qwen25_32b Qwen/Qwen2.5-32B-Instruct 0.85 3072
one mistral24b mistralai/Mistral-Small-24B-Instruct-2501 0.70 3072
echo "LIVE_QUERYEFF_HEAVY_DONE $(date +%H:%M:%S)"
