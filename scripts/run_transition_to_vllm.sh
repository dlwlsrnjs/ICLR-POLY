#!/usr/bin/env bash
# Wait for HF config A (text model) to finish, then stop the HF master and run
# configs B/C on vLLM. If the vLLM smoke test fails, fall back to the HF master.
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache
VPY=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python
GAME=private_artifacts/full_textdom/slot_game.jsonl
log(){ echo "[$(date +%H:%M:%S)] TRANSITION: $*"; }

log "waiting for config A (text model) summary..."
until [ -f results/full_textdom_textmodel_summary.json ]; do sleep 30; done
log "config A done. stopping HF master + any config B HF gen."
pkill -f run_full_textdom_3configs 2>/dev/null
pkill -f "run_vl_gated.py" 2>/dev/null
pkill -f "run_polyjig_gated.py" 2>/dev/null
sleep 20

# vLLM smoke test on 2 items, GPU0.
log "vLLM smoke test..."
head -2 "$GAME" > /tmp/vllm_smoke.jsonl
CUDA_VISIBLE_DEVICES=0 $VPY scripts/run_vl_gated_vllm.py --data /tmp/vllm_smoke.jsonl \
  --outdir /tmp/vllm_smoke_out --interleave-ns 4 --translated-langs Finnish \
  --gpu-memory-utilization 0.85 > /tmp/vllm_smoke.log 2>&1
if [ $? -eq 0 ] && [ -s /tmp/vllm_smoke_out/restricted_target_outputs.jsonl ]; then
  log "smoke test PASSED -> running configs B/C on vLLM"
  bash scripts/run_vl_configs_vllm.sh
else
  log "smoke test FAILED -> falling back to HF master for B/C"
  tail -5 /tmp/vllm_smoke.log
  nohup bash scripts/run_full_textdom_3configs.sh > /tmp/full_textdom_hf_bc.log 2>&1 &
fi
log "transition script done"
