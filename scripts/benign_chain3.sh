#!/bin/bash
# After the signal probe: run the MultiJail reconstruction probe against the newly built harmless
# MultiJail items, then hand over to the queued per-domain study.
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
while pgrep -u jinkwon -f 'benign_signals_run.sh|build_mj_benign_probe' > /dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] signals + benign build done; running MultiJail reconstruction probe"
bash scripts/benign_arms_mj_run.sh
echo CHAIN3_DONE
