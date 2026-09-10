#!/bin/bash
# Queue the per-domain study behind the harmless probes so nothing contends for a GPU.
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
while pgrep -u jinkwon -f 'benign_chain.sh|benign_arms|benign_signals' > /dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] harmless probes done; starting per-domain study"
bash scripts/domain_breakdown_run.sh
echo CHAIN2_DONE
