#!/usr/bin/env bash
# =====================================================================================
#  L40S-DEDICATED big/mid-model panel collection  (PolyJigsaw / ICLR-POLY)
#  Run this ONLY on the L40S box. It collects the C=160 harmful full-matrix + benign
#  probe for the 6 models in models.txt, for BOTH datasets (MultiJail + Lingua).
#  Results land under bigmodel_l40s/results/ (same layout as the shared box) so they
#  merge straight back into the panel. Safe to Ctrl-C and re-run: finished arms are
#  skipped via MANIFEST.json (resume). See README.md for the full guide.
# =====================================================================================
set -uo pipefail
cd "$(dirname "$0")/.."                       # repo root
HERE=bigmodel_l40s
VP="${VP:-python}"                            # set VP=/path/to/venv/bin/python if not activated
R="$HERE/results"                             # <-- L40S results live here, separate from the shared box
mkdir -p "$R"
log(){ echo "[l40s $(date +%F_%H:%M:%S)] $*"; }

# ---- required env (offline judges + target from local HF cache) ---------------------
: "${HF_HOME:?set HF_HOME to your HF cache dir, e.g. export HF_HOME=/data/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# VLLM_ENFORCE_EAGER=1 frees CUDA-graph memory for KV on the tightest fits (24B). On by default here.
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
# CUDA_VISIBLE_DEVICES orders the cards this run may use. Default: all. Target uses the first TP of
# them (cuda:0..TP-1); the judge pair goes on cuda:$TP. Override to pin, e.g. CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$($VP -c 'import torch;print(",".join(str(i) for i in range(torch.cuda.device_count())))')}"
NGPU=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
log "visible GPUs: $CUDA_VISIBLE_DEVICES  (count=$NGPU)  HF_HOME=$HF_HOME  VP=$VP"

MJ_ITEMS="${MJ_ITEMS:-64}"; LG_ITEMS="${LG_ITEMS:-40}"; JB="${JB:-8}"

grep -vE '^\s*#|^\s*$' "$HERE/models.txt" | while read -r HFID TP UTIL _rest; do
  [ -z "${HFID:-}" ] && continue
  JUDGE="cuda:${TP}"                          # judge card = right after the target shards
  if [ "$((TP+1))" -gt "$NGPU" ]; then
    log "SKIP $HFID : needs TP+1=$((TP+1)) GPUs but only $NGPU visible (see README 2-GPU note / AWQ fallback)"; continue
  fi
  log "=== $HFID  (TP=$TP util=$UTIL, target cuda:0..$((TP-1)), judge $JUDGE) ==="
  # MultiJail
  $VP experiments_suite/exp02_panel_collect/collect_mj.py both \
      --models "$HFID" --arm-space C --n-items "$MJ_ITEMS" --root "$R" \
      --tensor-parallel "$TP" --util "$UTIL" --judge-device "$JUDGE" --judge-batch-size "$JB" \
      || log "SKIP $HFID mj (see log above)"
  # Lingua
  $VP experiments_suite/exp02_panel_collect/collect_lg.py both \
      --models "$HFID" --arm-space C --n-items "$LG_ITEMS" --root "$R" \
      --tensor-parallel "$TP" --util "$UTIL" --judge-device "$JUDGE" --judge-batch-size "$JB" \
      || log "SKIP $HFID lg (see log above)"
done

$VP scripts/closed_compare.py repair-manifest --root "$R" || true
log "L40S_ALL_DONE  -> results in $R  (run bash $HERE/upload_results.sh to push back)"
