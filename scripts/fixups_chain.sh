#!/bin/bash
# Re-run whatever the main queue left incomplete. Every runner skips finished targets, so this is a
# safe catch-all: run it after the main chain and it fills only the gaps.
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
while pgrep -u jinkwon -f 'full_rerun_chain.sh' > /dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] main queue done; filling gaps"
bash scripts/benign_arms_run.sh            # Lingua reconstruction probe
bash scripts/benign_arms_big.sh
bash scripts/benign_arms_mj_run.sh         # MultiJail reconstruction probe
bash scripts/benign_signals_run.sh
bash scripts/benign_borderline_run.sh
bash scripts/domain_breakdown_run.sh
bash scripts/domain_breakdown_mj_run.sh
echo FIXUPS_DONE
