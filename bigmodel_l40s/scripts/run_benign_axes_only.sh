#!/usr/bin/env bash
# Inference-only collection of the other benign prior axis: multilingual
# comprehension. This invokes only the canonical `probe` phase; it never runs
# the harmful `matrix` phase and never loads the safety judge.
set -uo pipefail
cd "$(dirname "$0")/../.."

VP="${VP:-$PWD/.venv/bin/python}"
HF_HOME="${HF_HOME:-$PWD/bigmodel_l40s/hf_home}"
export HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_ENFORCE_EAGER=1
ROOT="bigmodel_l40s/results"
STATUS="$ROOT/benign_axes_status.tsv"
mkdir -p "$ROOT" logs

status() {
  printf '%s\t%s\n' "$(date -Iseconds)" "$*" >> "$STATUS"
}

wait_free() {
  local csv="$1" query gpu free ready
  while true; do
    query=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    ready=1
    for gpu in ${csv//,/ }; do
      free=$(awk -F, -v wanted="$gpu" '$1 + 0 == wanted {gsub(/ /,"",$2); print $2}' <<<"$query")
      if [ -z "$free" ] || [ "$free" -lt 44000 ]; then ready=0; fi
    done
    [ "$ready" -eq 1 ] && return 0
    sleep 20
  done
}

probe_model() {
  local gpus="$1" judge="$2" model="$3" tag="$4" tp="$5" util="$6" collection driver suffix out
  for collection in mj lg; do
    if [ "$collection" = mj ]; then
      driver="experiments_suite/exp02_panel_collect/collect_mj.py"
      suffix="mj"
    else
      driver="experiments_suite/exp02_panel_collect/collect_lg.py"
      suffix="lg"
    fi
    out="$ROOT/benign/${tag}_${suffix}.json"
    if [ -s "$out" ]; then
      status "SKIP model=$model collection=$collection complete"
      continue
    fi
    status "START model=$model collection=$collection GPUs=$gpus phase=benign_probe"
    if CUDA_VISIBLE_DEVICES="$gpus" "$VP" "$driver" probe \
        --models "$model" --root "$ROOT" --fp-benign 24 --util "$util" \
        --tensor-parallel "$tp" --judge-device "$judge" --judge-batch-size 8; then
      status "COMPLETE model=$model collection=$collection"
    else
      status "FAILED model=$model collection=$collection"
      return 1
    fi
  done
}

status "QUEUE_STARTED purpose=inference_only_axes axes=comprehension,willingness harmful=false"

# Lane A can start immediately on the currently free pair.
(
  wait_free 2,3
  probe_model 2,3 cuda:1 google/gemma-2-2b-it gemma2_2b_it 1 0.30
  probe_model 2,3 cuda:1 meta-llama/Llama-3.1-8B-Instruct llama31_8b_it 1 0.50
  probe_model 2,3 cuda:1 Qwen/Qwen2.5-14B-Instruct qwen25_14b 1 0.85
) & lane_a=$!

# Lane B begins as soon as the ongoing Mistral willingness run releases 0,1.
(
  wait_free 0,1
  probe_model 0,1 cuda:1 tiiuae/Falcon3-7B-Instruct falcon3_7b 1 0.45
  probe_model 0,1 cuda:1 google/gemma-2-9b-it gemma2_9b_it 1 0.55
) & lane_b=$!

wait "$lane_a"; a_rc=$?
wait "$lane_b"; b_rc=$?
status "ONE_GPU_LANES_DONE lane_a=$a_rc lane_b=$b_rc"

# The 24B target follows the repository's TP=2 setting; cuda:2 holds only the
# reconstruction judge used by the harmless probe.
wait_free 0,1,2
probe_model 0,1,2 cuda:2 mistralai/Mistral-Small-24B-Instruct-2501 mistral24b 2 0.85
status "QUEUE_FINISHED"
