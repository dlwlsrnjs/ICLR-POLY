#!/bin/bash
# Sequential retry (up to 5x each) for the 4 missing MJ evals, on GPU1 (more reliable).
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
HARM=private_artifacts/multijail_v1/harm_grid.jsonl
ORDER=private_artifacts/multijail_v1/resource_order.json
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1

run_until() { OUT="$1"; shift; for a in 1 2 3 4 5; do
  [ -f "$OUT" ] && { echo "OK $OUT"; return 0; }
  echo "[try $a $(date +%H:%M:%S)] $OUT"; "$@"; sleep 8
done; }

export VLLM_USE_FLASHINFER_SAMPLER=0
run_until results/mj_method_20260906/qwen25_14b.json \
  $VP/python scripts/method_baselines_eval.py --target Qwen/Qwen2.5-14B-Instruct --tag qwen25_14b \
  --harm $HARM --outdir results/mj_method_20260906 --n-items 64 --util 0.40
run_until results/mj_multijail_20260906/qwen25_14b.json \
  $VP/python scripts/multijail_eval.py --target Qwen/Qwen2.5-14B-Instruct --tag qwen25_14b \
  --harm $HARM --outdir results/mj_multijail_20260906 --n-items 64 --util 0.40
run_until results/mj_combo_20260906/qwen25_14b.json \
  $VP/python scripts/combo_eval.py --target Qwen/Qwen2.5-14B-Instruct --tag qwen25_14b \
  --harm $HARM --order $ORDER --n 4 --outdir results/mj_combo_20260906 --n-items 64 --util 0.40

unset VLLM_USE_FLASHINFER_SAMPLER; export VLLM_ATTENTION_BACKEND=TRITON_ATTN
run_until results/mj_multijail_20260906/gemma2_9b_it.json \
  $VP/python scripts/multijail_eval.py --target google/gemma-2-9b-it --tag gemma2_9b_it \
  --harm $HARM --outdir results/mj_multijail_20260906 --n-items 64 --util 0.40
echo ALLFINAL_DONE
