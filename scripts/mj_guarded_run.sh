#!/bin/bash
# Waits for a GPU with >=NEED MiB free (stable over 3 checks) before each run, so runs don't
# OOM under intermittent contention. Runs the weaklang-strengthening test on strong models and
# the qwen25_14b MJ baselines. Resumable (skips existing jsons).
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
HARM=private_artifacts/multijail_v1/harm_grid.jsonl
ORDER=private_artifacts/multijail_v1/resource_order.json
NEED=68000
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wait_free_gpu() {  # echoes a gpu index once it has >=NEED free, stable x3
  while true; do
    for g in 1 0; do
      ok=1
      for c in 1 2 3; do
        f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)
        [ "$f" -lt "$NEED" ] && { ok=0; break; }
        sleep 6
      done
      [ "$ok" = 1 ] && { echo $g; return; }
    done
    sleep 30
  done
}

run() { OUT="$1"; GPU="$2"; BE="$3"; shift 3
  [ -f "$OUT" ] && { echo "have $OUT"; return; }
  export CUDA_VISIBLE_DEVICES=$GPU
  if [ "$BE" = triton ]; then export VLLM_ATTENTION_BACKEND=TRITON_ATTN; unset VLLM_USE_FLASHINFER_SAMPLER
  else export VLLM_USE_FLASHINFER_SAMPLER=0; unset VLLM_ATTENTION_BACKEND; fi
  echo "[$(date +%H:%M:%S)] GPU$GPU -> $OUT"; "$@"
}

# 1) weaklang strengthening (verify-EN, answer-in-weak-lang) on strong models
for spec in "qwen25_7b|Qwen/Qwen2.5-7B-Instruct|0.35|flash" \
            "llama31_8b_it|meta-llama/Llama-3.1-8B-Instruct|0.35|flash" \
            "gemma2_9b_it|google/gemma-2-9b-it|0.40|triton"; do
  IFS='|' read tag ref util be <<< "$spec"
  G=$(wait_free_gpu)
  run results/mj_weaklang_20260906/$tag.json $G $be \
    $VP/python scripts/weaklang_answer_eval.py --target "$ref" --tag $tag --order $ORDER --harm $HARM \
    --n 4 --n-items 64 --util $util
done
# 2) qwen25_14b MJ baselines (need clean GPU)
for spec in "method|method_baselines_eval|" "multijail|multijail_eval|" "combo|combo_eval|--order $ORDER --n 4"; do
  IFS='|' read d script extra <<< "$spec"
  G=$(wait_free_gpu)
  run results/mj_${d}_20260906/qwen25_14b.json $G flash \
    $VP/python scripts/$script.py --target Qwen/Qwen2.5-14B-Instruct --tag qwen25_14b --harm $HARM \
    --outdir results/mj_${d}_20260906 --n-items 64 --util 0.45 --max-model-len 4096 $extra
done
echo GUARDED_ALL_DONE
