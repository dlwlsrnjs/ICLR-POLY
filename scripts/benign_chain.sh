#!/bin/bash
# Serialise the harmless probes so a large run is never starved by a small one:
#   Lingua reconstruction (32B/27B)  ->  MultiJail reconstruction (all 9)  ->  signal probe (all 9, both languages)
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
while pgrep -u jinkwon -f 'benign_arms_big.sh' > /dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] Lingua reconstruction probe done; starting MultiJail reconstruction probe"
bash scripts/benign_arms_mj_run.sh
echo "[$(date +%H:%M:%S)] MultiJail reconstruction probe done; starting signal probe"
bash scripts/benign_signals_run.sh
echo BENIGN_CHAIN_DONE
