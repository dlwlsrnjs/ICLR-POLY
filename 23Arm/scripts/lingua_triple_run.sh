#!/bin/bash
# Add the triple (role-separation) arm to Lingua-SafetyBench for all 7 models, so the Lingua bandit
# matches MJ's arm set. Puzzle=high-resource langs, [ANSWER]=Norwegian (Lingua's lowest-resource /
# weakest-alignment tier = order[0]). 2-phase, n-items=40 to match the existing Lingua panel.
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
HARM=private_artifacts/panel_v2/harm_grid.jsonl
ORDER=results/lang_rank_20260905/resource_order.json
OUT=results/lingua_triple_20260907; AL=Norwegian
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${GPU:-1}
gj() { tag="$1"; ref="$2"; be="$3"; util="$4"
  [ -f $OUT/$tag.json ] && { echo "OK $tag"; return; }
  if [ "$be" = triton ]; then export VLLM_ATTENTION_BACKEND=TRITON_ATTN; unset VLLM_USE_FLASHINFER_SAMPLER
  else export VLLM_USE_FLASHINFER_SAMPLER=0; unset VLLM_ATTENTION_BACKEND; fi
  echo "[gen $tag $(date +%H:%M:%S)]"
  for a in 1 2 3; do $VP/python scripts/triple_combo_eval.py --phase gen --target "$ref" --tag $tag \
    --order $ORDER --harm $HARM --answer-lang $AL --n 4 --n-items 40 --util $util --max-model-len 4096 --outdir $OUT && break; sleep 12; done
  echo "[judge $tag $(date +%H:%M:%S)]"
  for a in 1 2 3; do $VP/python scripts/triple_combo_eval.py --phase judge --tag $tag \
    --order $ORDER --harm $HARM --answer-lang $AL --outdir $OUT && [ -f $OUT/$tag.json ] && break; sleep 12; done
}
gj qwen25_3b     Qwen/Qwen2.5-3B-Instruct        flash  0.35
gj qwen25_7b     Qwen/Qwen2.5-7B-Instruct        flash  0.35
gj llama32_3b_it meta-llama/Llama-3.2-3B-Instruct flash  0.35
gj llama31_8b_it meta-llama/Llama-3.1-8B-Instruct flash  0.35
gj gemma2_2b_it  google/gemma-2-2b-it            triton 0.35
gj gemma2_9b_it  google/gemma-2-9b-it            triton 0.40
gj qwen25_14b    Qwen/Qwen2.5-14B-Instruct       flash  0.45
echo LINGUA_TRIPLE_DONE
