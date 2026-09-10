#!/bin/bash
# One serial queue for the full re-run, so nothing contends for the single available GPU:
#   1. finish the harmless probes (MultiJail reconstruction, then the safety-adjacent request set)
#   2. per-domain study with per-item judgments, Lingua then MultiJail, all nine targets
# Every stage is idempotent, so this script can be restarted at any point.
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
step() { echo; echo "[$(date +%H:%M:%S)] === $* ==="; }
while pgrep -u jinkwon -f 'benign_arms_mj_run.sh' > /dev/null; do sleep 60; done
step "harmless probe: safety-adjacent requests"
bash scripts/benign_borderline_run.sh
step "per-domain study: Lingua, nine targets"
bash scripts/domain_breakdown_run.sh
step "per-domain study: MultiJail, nine targets"
bash scripts/domain_breakdown_mj_run.sh
echo "FULL_RERUN_CHAIN_DONE"
