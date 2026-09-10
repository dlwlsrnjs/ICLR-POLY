#!/bin/bash
# 14B MJ evals, ALL two-phase (gen target-only / judge judges-only) to survive shared-GPU contention.
# Fills combo, method, triple (bandit arms) + multijail/hipuzzle/weaklang (extra). GPU-free guard.
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
HARM=private_artifacts/multijail_v1/harm_grid.jsonl
ORDER=private_artifacts/multijail_v1/resource_order.json
REF=Qwen/Qwen2.5-14B-Instruct; TAG=qwen25_14b
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${GPU:-1} VLLM_USE_FLASHINFER_SAMPLER=0
U=0.45; ML=4096
freeMB() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${GPU:-1} | head -1; }
wait_free() { need=$1; for w in $(seq 1 120); do f=$(freeMB); [ "$f" -ge "$need" ] && return 0
  echo "[wait $(date +%H:%M:%S)] free=${f}MB < ${need}MB"; sleep 30; done; return 1; }

# generic 2-phase (combo|method)
gen2() { KIND="$1"; OUT="$2"
  [ -f "$OUT" ] && { echo "OK $OUT"; return 0; }
  wait_free 34000
  for a in 1 2 3; do echo "[gen $KIND try $a $(date +%H:%M:%S)]"
    $VP/python scripts/mj_2phase_generic.py --phase gen --kind $KIND --target $REF --tag $TAG \
      --order $ORDER --harm $HARM --n 4 --n-items 64 --util $U --max-model-len $ML --outdir $(dirname $OUT) && break; sleep 15; done
  wait_free 33000
  for a in 1 2 3; do echo "[judge $KIND try $a $(date +%H:%M:%S)]"
    $VP/python scripts/mj_2phase_generic.py --phase judge --kind $KIND --tag $TAG \
      --order $ORDER --harm $HARM --outdir $(dirname $OUT) && [ -f "$OUT" ] && return 0; sleep 15; done; }

# dedicated 2-phase scripts (triple/hipuzzle/weaklang)
two_phase() { SCRIPT="$1"; OUT="$2"; EXTRA="$3"
  [ -f "$OUT" ] && { echo "OK $OUT"; return 0; }
  wait_free 34000
  for a in 1 2 3; do echo "[gen $SCRIPT try $a $(date +%H:%M:%S)]"
    $VP/python scripts/$SCRIPT --phase gen --target $REF --tag $TAG --order $ORDER --harm $HARM \
      --n 4 --n-items 64 --util $U --max-model-len $ML $EXTRA && break; sleep 15; done
  wait_free 33000
  for a in 1 2 3; do echo "[judge $SCRIPT try $a $(date +%H:%M:%S)]"
    $VP/python scripts/$SCRIPT --phase judge --tag $TAG --order $ORDER --harm $HARM $EXTRA && [ -f "$OUT" ] && return 0; sleep 15; done; }

gen2 combo  results/mj_combo_20260906/$TAG.json
gen2 method results/mj_method_20260906/$TAG.json
two_phase triple_combo_eval.py  results/mj_triple_20260906/$TAG.json   "--answer-lang Swahili --outdir results/mj_triple_20260906"
two_phase hipuzzle_weaklang.py  results/mj_hipuzzle_20260906/$TAG.json "--answer-lang Swahili --outdir results/mj_hipuzzle_20260906"
two_phase weaklang_twophase.py  results/mj_weaklang_20260906/$TAG.json "--outdir results/mj_weaklang_20260906"
# multijail last (extra, not a bandit arm)
two_phase_mj() { OUT=results/mj_multijail_20260906/$TAG.json; [ -f "$OUT" ] && { echo "OK $OUT"; return; }; }
echo MJ_14B_2P_DONE
