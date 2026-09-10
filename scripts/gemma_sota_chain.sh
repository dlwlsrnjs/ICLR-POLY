#!/bin/bash
# Chain: wait until the 4 small held-out signal probes exist (so the signals-expand job has freed its
# GPU slot; yi34b may still be queued behind other users' jobs), then run the gemma-2 SOTA retry.
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
echo "[$(date +%H:%M:%S)] gemma-sota chain: waiting for 4 small held-out signal probes"
for i in $(seq 1 480); do
  n=$(ls results/benign_signals_mj_20260907/{olmo2_7b,zephyr7b,qwen25_15b,qwen3_8b}.json \
        results/benign_signals_20260907/{olmo2_7b,zephyr7b,qwen25_15b,qwen3_8b}.json \
        results/benign_borderline_mj_20260907/{olmo2_7b,zephyr7b,qwen25_15b,qwen3_8b}.json \
        results/benign_borderline_20260907/{olmo2_7b,zephyr7b,qwen25_15b,qwen3_8b}.json 2>/dev/null | wc -l)
  [ "$n" -ge 16 ] && { echo "[$(date +%H:%M:%S)] small probes ready ($n/16)"; break; }
  sleep 30
done
exec bash scripts/sota_gemma_retry_run.sh
