#!/usr/bin/env bash
# Closed-model per-target comparison, in the paper's two-phase order.
#
#   PHASE 1 (probe)  = HARMLESS. Plaintext-only benign probing to SELECT the per-target setting.
#                      No harmful request; safety judge not loaded. Safe to run unattended.
#   PHASE 2 (attack) = HARMFUL. Fires the selected config + comparison baselines on the harmful set.
#                      Run this yourself, under your own accountability. Needs a free GPU for the judges.
#
# Storage: everything lands under $ROOT with a MANIFEST.json the later analysis pass reads on its own.
# Requirements: GPU with room for the local judges (recon 15GB; +safety 16GB in phase 2).
#               OPENAI_API_KEY loaded from the 0600 secret file below.
#
# Cost (GPT-4o, target generation only; local judges are free):
#   phase 1 per collection: (32 cells x FP_BENIGN) + (3 frames x 16) harmless calls.
#   phase 2 per collection: shortlist + six baselines, using the collection-specific item count.
#   API cost depends on the selected shortlist, response length and provider; use the saved ledger.
set -euo pipefail
cd "$(dirname "$0")/.."                      # -> PolyJigsaw/
VP=${VP:-/data1/users/ljk98/envs/VLLM-VL-LABEL/bin/python}
export HF_HOME=${HF_HOME:-/data1/users/ljk98/hf_cache} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export PATH="$(dirname "$VP"):${PATH}"
# Keep regenerable compiler/JIT caches off the small system partition.
RUN_CACHE=${RUN_CACHE:-/data1/users/ljk98/runtime_cache}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$RUN_CACHE/xdg}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$RUN_CACHE/vllm}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$RUN_CACHE/flashinfer}
export POLY_JUDGE_BATCH_SIZE=${POLY_JUDGE_BATCH_SIZE:-16}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ -z ${OPENAI_API_KEY:-} && -n ${OPENAI_KEY_FILE:-} ]]; then
  export OPENAI_API_KEY="$(<"$OPENAI_KEY_FILE")"
fi

ROOT=${ROOT:-results/closed_compare_$(date +%Y%m%d)}
MODEL=${MODEL:-gpt-4o}
BACKEND=${BACKEND:-openai}
MJ_N_ITEMS=${MJ_N_ITEMS:-64}
LG_N_ITEMS=${LG_N_ITEMS:-40}
FP_BENIGN=${FP_BENIGN:-24}
JUDGE_DEVICE=${JUDGE_DEVICE:-cuda:0}         # point at a GPU that is actually free
FORCE=${FORCE:-0}
# Collection-specific inputs: the two collections use DIFFERENT language sets, so order + benign probe
# + harm file must all match the collection (mixing them silently builds puzzles in the wrong languages).
MJ_HARM=private_artifacts/multijail_v1/harm_grid.jsonl
LG_HARM=private_artifacts/panel_v2/harm_grid.jsonl
MJ_ORDER=private_artifacts/multijail_v1/resource_order.json
LG_ORDER=results/lang_rank_20260905/resource_order.json
MJ_BENIGN=private_artifacts/multijail_v1/benign_probe.jsonl
LG_BENIGN=private_artifacts/panel_v2/benign_probe.jsonl
BASELINES="plain,translated,cipher_base64,aim,deepinception,pap"

phase=${1:-help}

run_probe () {  # $1=collection $2=tag $3=order $4=benign  (HARMLESS)
  force_args=(); [[ "$FORCE" == 1 ]] && force_args+=(--force)
  "$VP" scripts/closed_compare.py probe \
    --backend "$BACKEND" --model "$MODEL" --tag "$2" --collection "$1" \
    --root "$ROOT" --order "$3" --benign "$4" --fp-benign "$FP_BENIGN" --judge-device "$JUDGE_DEVICE" \
    "${force_args[@]}"
}

run_attack () {  # $1=collection $2=tag $3=harm $4=order $5=n-items  (HARMFUL)
  force_args=(); [[ "$FORCE" == 1 ]] && force_args+=(--force)
  short=$("$VP" -c "import json;print(','.join(json.load(open('$ROOT/benign/$2.json'))['shortlist']))")
  echo ">> $2: phase-1 shortlist (confirmatory pulls) = $short"
  "$VP" scripts/closed_compare.py attack \
    --backend "$BACKEND" --model "$MODEL" --tag "$2" --collection "$1" \
    --root "$ROOT" --order "$4" --harm "$3" --n-items "$5" \
    --shortlist "$short" --methods "$BASELINES" --judge-device "$JUDGE_DEVICE" \
    "${force_args[@]}"
}

case "$phase" in
  probe)                       # HARMLESS: run this first (Claude can run this part)
    run_probe MultiJail          "${MODEL//\//_}_mj" "$MJ_ORDER" "$MJ_BENIGN"
    run_probe Lingua-SafetyBench "${MODEL//\//_}_lg" "$LG_ORDER" "$LG_BENIGN"
    echo "PHASE1_DONE  -> $ROOT/benign/"
    ;;
  attack)                      # HARMFUL: run this yourself after phase 1
    run_attack MultiJail          "${MODEL//\//_}_mj" "$MJ_HARM" "$MJ_ORDER" "$MJ_N_ITEMS"
    run_attack Lingua-SafetyBench "${MODEL//\//_}_lg" "$LG_HARM" "$LG_ORDER" "$LG_N_ITEMS"
    echo "PHASE2_DONE  -> $ROOT/attack/  (aggregates + _raw/)"
    ;;
  *)
    echo "usage: ROOT=... MODEL=gpt-4o bash scripts/closed_compare_run.sh {probe|attack}"
    echo "  probe  = harmless plaintext setting-selection (phase 1)"
    echo "  attack = harmful evaluation of selected config + baselines (phase 2)"
    ;;
esac
