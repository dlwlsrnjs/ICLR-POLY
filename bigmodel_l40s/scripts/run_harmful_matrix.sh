#!/usr/bin/env bash
# ==============================================================================================
#  L40S — HARMFUL full-matrix (C=160) for the 6 big/mid models.
#  Symmetric to run_benign_axes_only.sh, but phase=matrix (verified ASR) with BOTH judges resident.
#  The benign + willingness priors are already collected; this fills the harmful arms (0 -> 164).
#  Writes into the MAIN panel dir so it colocates with the benign priors already committed there.
#  Resumable: finished arms are skipped via MANIFEST.json. 4x L40S assumed (lanes 2,3 / 0,1 / 0,1,2).
# ==============================================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."
VP="${VP:-$PWD/.venv/bin/python}"
HF_HOME="${HF_HOME:-$PWD/bigmodel_l40s/hf_home}"
export HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_ENFORCE_EAGER=1 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT="experiments_suite/exp02_panel_collect/results"    # same dir as the committed benign priors
STATUS="$ROOT/harmful_matrix_status_l40s.tsv"
mkdir -p "$ROOT" logs
status(){ printf '%s\t%s\n' "$(date -Iseconds)" "$*" >> "$STATUS"; }

wait_free(){  # wait until every card in the csv has >=44GB free
  local csv="$1" query gpu free ready
  while true; do
    query=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    ready=1
    for gpu in ${csv//,/ }; do
      free=$(awk -F, -v w="$gpu" '$1+0==w{gsub(/ /,"",$2);print $2}' <<<"$query")
      { [ -z "$free" ] || [ "$free" -lt 44000 ]; } && ready=0
    done
    [ "$ready" -eq 1 ] && return 0; sleep 20
  done
}

matrix_model(){  # gpus judge model tag tp util
  local gpus="$1" judge="$2" model="$3" tag="$4" tp="$5" util="$6" collection driver items
  for collection in mj lg; do
    if [ "$collection" = mj ]; then driver=experiments_suite/exp02_panel_collect/collect_mj.py; items=64
    else driver=experiments_suite/exp02_panel_collect/collect_lg.py; items=40; fi
    status "START model=$model collection=$collection GPUs=$gpus phase=harmful_matrix"
    if CUDA_VISIBLE_DEVICES="$gpus" "$VP" "$driver" matrix \
        --models "$model" --arm-space C --n-items "$items" --root "$ROOT" \
        --tensor-parallel "$tp" --util "$util" --judge-device "$judge" --judge-batch-size 8; then
      status "COMPLETE model=$model collection=$collection"
    else
      status "FAILED model=$model collection=$collection"
    fi
  done
}

status "QUEUE_STARTED purpose=harmful_matrix space=C harmful=true"

# Lane A (cards 2,3): target cuda:0=phys2, judges cuda:1=phys3
( wait_free 2,3
  matrix_model 2,3 cuda:1 google/gemma-2-2b-it           gemma2_2b_it  1 0.30
  matrix_model 2,3 cuda:1 meta-llama/Llama-3.1-8B-Instruct llama31_8b_it 1 0.50
  matrix_model 2,3 cuda:1 Qwen/Qwen2.5-14B-Instruct       qwen25_14b    1 0.85
) & lane_a=$!

# Lane B (cards 0,1): target cuda:0=phys0, judges cuda:1=phys1
( wait_free 0,1
  matrix_model 0,1 cuda:1 tiiuae/Falcon3-7B-Instruct      falcon3_7b    1 0.45
  matrix_model 0,1 cuda:1 google/gemma-2-9b-it            gemma2_9b_it  1 0.55
) & lane_b=$!

wait "$lane_a"; a=$?; wait "$lane_b"; b=$?
status "ONE_GPU_LANES_DONE lane_a=$a lane_b=$b"

# 24B: TP=2 target on cards 0,1; judges on card 2
wait_free 0,1,2
matrix_model 0,1,2 cuda:2 mistralai/Mistral-Small-24B-Instruct-2501 mistral24b 2 0.85
status "QUEUE_FINISHED"

# rebuild the manifest so the new harmful arms register next to the benign priors
"$VP" scripts/closed_compare.py repair-manifest --root "$ROOT" || true
status "MANIFEST_REPAIRED"
