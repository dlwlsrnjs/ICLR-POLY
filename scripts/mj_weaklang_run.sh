#!/bin/bash
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0
/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python scripts/weaklang_answer_eval.py --target Qwen/Qwen2.5-7B-Instruct --tag qwen25_7b --order private_artifacts/multijail_v1/resource_order.json --harm private_artifacts/multijail_v1/harm_grid.jsonl --n 4 --n-items 64 --util 0.35
/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python scripts/weaklang_answer_eval.py --target meta-llama/Llama-3.1-8B-Instruct --tag llama31_8b_it --order private_artifacts/multijail_v1/resource_order.json --harm private_artifacts/multijail_v1/harm_grid.jsonl --n 4 --n-items 64 --util 0.35
export VLLM_ATTENTION_BACKEND=TRITON_ATTN; unset VLLM_USE_FLASHINFER_SAMPLER
/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python scripts/weaklang_answer_eval.py --target google/gemma-2-9b-it --tag gemma2_9b_it --order private_artifacts/multijail_v1/resource_order.json --harm private_artifacts/multijail_v1/harm_grid.jsonl --n 4 --n-items 64 --util 0.40
echo WEAKLANG_DONE
