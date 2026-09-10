#!/usr/bin/env bash
# After the full 32-arm factorial queue finishes: build the FULL contextual dataset,
# train the contextual selector (3 seeds) on an idle GPU, run the online loop. Logged.
set -uo pipefail
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
VPY=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python
QPID="${1:-3194097}"
DATA=results/context_selector_full_train_20260904
TRAIN=results/context_selector_full_run_20260904
ONLINE=results/online_adapt_full_20260904
log(){ echo "[$(date +%H:%M:%S)] FULLCHAIN: $*"; }

log "waiting for full factorial queue pid $QPID"
while kill -0 "$QPID" 2>/dev/null; do sleep 60; done
log "queue exited"; sleep 30

n=$(ls -d private_artifacts/frag_factorial_20260903/*/ 2>/dev/null | while read d; do [ -f "$d/harmful_summary.json" ] && echo x; done | wc -l)
log "targets with full 32-arm factorial: $n"

log "building FULL contextual dataset"
$VPY scripts/build_context_selector_data_full.py --outdir "$DATA" || { log "BUILD FAILED"; exit 1; }

log "refitting DOMAIN-STRUCTURED policy on full 32-arm data (exports warm-start prior)"
$VPY scripts/structured_policy.py --data "$DATA" --outdir results/structured_policy_full_20260904 \
  > results/structured_policy_full_20260904.log 2>&1 || { log "STRUCTURED FIT FAILED"; tail -8 results/structured_policy_full_20260904.log; }

GPU=$($VPY - <<'PY'
import subprocess
o=subprocess.run(["nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout
best=None
for line in o.strip().splitlines():
    i,f=[x.strip() for x in line.split(",")]
    if int(f)>8000 and (best is None or int(f)>best[1]): best=(int(i),int(f))
print(best[0] if best else -1)
PY
)
log "training GPU (most free) = $GPU"
if [ "$GPU" -ge 0 ]; then
  rm -rf "$TRAIN"
  CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/train_context_selector.py --data "$DATA" --outdir "$TRAIN" \
    --supervised-steps 800 --ppo-updates 600 --seeds 7 17 27 > "$TRAIN.log" 2>&1 || { log "TRAIN FAILED"; tail -15 "$TRAIN.log"; }
else log "no idle GPU; skipping training"; fi

log "online loop (replay, 32-arm)"
$VPY scripts/online_adapt.py --data "$DATA" --budget 8 --threshold 0.5 --outdir "$ONLINE" > "$ONLINE.log" 2>&1 || { log "ONLINE FAILED"; tail -10 "$ONLINE.log"; }
log "FULLCHAIN COMPLETE"; echo DONE > "$ONLINE/fullchain_done.marker"
