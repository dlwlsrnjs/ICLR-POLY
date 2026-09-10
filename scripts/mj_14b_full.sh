#!/bin/bash
# Fill ALL missing MJ evals for qwen25_14b on a fully-free GPU. Single-phase for method/multijail/combo
# (fits: 14B ~28GB + judges 31GB < 81GB at util 0.45); 2-phase for triple/hipuzzle/weaklang.
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
HARM=private_artifacts/multijail_v1/harm_grid.jsonl
ORDER=private_artifacts/multijail_v1/resource_order.json
REF=Qwen/Qwen2.5-14B-Instruct; TAG=qwen25_14b
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${GPU:-1} VLLM_USE_FLASHINFER_SAMPLER=0
U=0.45; ML=4096
run_until() { OUT="$1"; shift; [ -f "$OUT" ] && { echo "OK $OUT"; return 0; }
  for a in 1 2 3; do echo "[sp try $a $(date +%H:%M:%S)] $OUT"; "$@" && [ -f "$OUT" ] && return 0; sleep 10; done; }
# --- single-phase ---
run_until results/mj_method_20260906/$TAG.json \
  $VP/python scripts/method_baselines_eval.py --target $REF --tag $TAG --harm $HARM \
  --outdir results/mj_method_20260906 --n-items 64 --util $U --max-model-len $ML
run_until results/mj_multijail_20260906/$TAG.json \
  $VP/python scripts/multijail_eval.py --target $REF --tag $TAG --harm $HARM \
  --outdir results/mj_multijail_20260906 --n-items 64 --util $U --max-model-len $ML
run_until results/mj_combo_20260906/$TAG.json \
  $VP/python scripts/combo_eval.py --target $REF --tag $TAG --harm $HARM --order $ORDER --n 4 \
  --outdir results/mj_combo_20260906 --n-items 64 --util $U --max-model-len $ML
# --- 2-phase (gen target-only -> judge judges-only) ---
two_phase() { SCRIPT="$1"; OUT="$2"; EXTRA="$3"
  [ -f "$OUT" ] && { echo "OK $OUT"; return 0; }
  for a in 1 2 3; do echo "[gen try $a $(date +%H:%M:%S)] $OUT"
    $VP/python scripts/$SCRIPT --phase gen --target $REF --tag $TAG --order $ORDER --harm $HARM \
      --n 4 --n-items 64 --util $U --max-model-len $ML $EXTRA && break; sleep 10; done
  for a in 1 2 3; do echo "[judge try $a $(date +%H:%M:%S)] $OUT"
    $VP/python scripts/$SCRIPT --phase judge --tag $TAG --order $ORDER --harm $HARM $EXTRA && [ -f "$OUT" ] && return 0; sleep 10; done; }
two_phase triple_combo_eval.py  results/mj_triple_20260906/$TAG.json   "--answer-lang Swahili --outdir results/mj_triple_20260906"
two_phase hipuzzle_weaklang.py  results/mj_hipuzzle_20260906/$TAG.json "--answer-lang Swahili --outdir results/mj_hipuzzle_20260906"
two_phase weaklang_twophase.py  results/mj_weaklang_20260906/$TAG.json "--outdir results/mj_weaklang_20260906"
echo MJ_14B_FULL_DONE
