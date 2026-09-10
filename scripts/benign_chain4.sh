#!/bin/bash
# Last harmless probe in the queue: the safety-adjacent request set.
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
while pgrep -u jinkwon -f 'benign_signals_run.sh|benign_arms_mj_run.sh|benign_chain3.sh' > /dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] starting borderline-request probe"
bash scripts/benign_borderline_run.sh
echo CHAIN4_DONE
