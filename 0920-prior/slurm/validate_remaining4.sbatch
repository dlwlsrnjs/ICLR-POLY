#!/bin/bash
#SBATCH -J prior331-qa4
#SBATCH -p h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -t 01:00:00
#SBATCH -D /home/ljk98/POLY
#SBATCH -o logs/slurm-%j-%x.out
set -euo pipefail
P=/home/ljk98/POLY;R=$P/ICLR-POLY-0920;GPU=${CUDA_VISIBLE_DEVICES:?}
export HF_HOME=$P/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=0
CODE=$R/experiments_suite/exp08_willingness394_milmmt_v3/aligned/model_specific/review_translation_failures.py
srun "$P/.venv-gen/bin/python" "$CODE" \
 --source "$P/prior331_runs/translation_repair_20260920/gpt56sol_remaining4_v4_gpt_backtranslation/translations.jsonl" \
 --out "$P/prior331_runs/translation_repair_20260920/gpt56sol_remaining4_v4_qwen32_validation" \
 --model-path "$P/hf-cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd" \
 --gpu "$GPU" --memory-utilization 0.90 --expected-count 4
