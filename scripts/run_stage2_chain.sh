#!/usr/bin/env bash
# After the panel expansion queue finishes: build the contextual dataset, train the
# contextual selector on an idle GPU (never GPU0 if another user holds it), then run the
# online adaptation loop in replay. Robust, logged, idempotent-ish.
set -uo pipefail
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
VPY=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python
QUEUE_PID="${1:-215942}"
DATA=results/context_selector_train_20260903
TRAIN=results/context_selector_run_20260903
ONLINE=results/online_adapt_20260903
log(){ echo "[$(date +%H:%M:%S)] STAGE2: $*"; }

log "waiting for collection queue pid $QUEUE_PID"
while kill -0 "$QUEUE_PID" 2>/dev/null; do sleep 30; done
log "collection queue exited"
sleep 25   # let the last vLLM release GPU memory

n=$(ls -d private_artifacts/panel_v2/*/ 2>/dev/null | while read d; do [ -f "$d/harmful_summary.json" ] && echo x; done | wc -l)
log "completed targets with joint observations: $n"

log "building contextual dataset"
$VPY scripts/build_context_selector_data.py --outdir "$DATA" || { log "BUILD FAILED"; exit 1; }

# pick the idle GPU (lowest memory used); refuse to contend with another user's job
GPU=$($VPY - <<'PY'
import subprocess
o=subprocess.run(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout
best=None
for line in o.strip().splitlines():
    i,u=[x.strip() for x in line.split(",")]
    if int(u)<3000 and (best is None or int(u)<best[1]): best=(int(i),int(u))
print(best[0] if best else -1)
PY
)
log "idle GPU = $GPU"
if [ "$GPU" -lt 0 ]; then log "no idle GPU; skipping training, running online replay only"; else
  log "training contextual selector on GPU $GPU"
  rm -rf "$TRAIN"
  CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/train_context_selector.py --data "$DATA" --outdir "$TRAIN" \
    > "$TRAIN.log" 2>&1 || { log "TRAIN FAILED"; tail -15 "$TRAIN.log"; }
fi

log "running online adaptation loop (replay)"
$VPY scripts/online_adapt.py --data "$DATA" --budget 8 --threshold 0.5 --outdir "$ONLINE" \
  > "$ONLINE.log" 2>&1 || { log "ONLINE FAILED"; tail -15 "$ONLINE.log"; }

log "STAGE2 COMPLETE"
echo "DONE" > "$ONLINE/stage2_done.marker"
