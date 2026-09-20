#!/bin/bash
#SBATCH -J prior331-back17
#SBATCH -p rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH -t 01:00:00
#SBATCH -D /home/ljk98/POLY
#SBATCH -o logs/slurm-%j-%x.out
set -euo pipefail
P=/home/ljk98/POLY
R=$P/ICLR-POLY-0920
GPU=${CUDA_VISIBLE_DEVICES:?Slurm did not assign CUDA_VISIBLE_DEVICES}
export HF_HOME=$P/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CODE=$R/experiments_suite/exp08_willingness394_milmmt_v3/aligned/model_specific/backtranslate_openai_repairs.py
srun "$P/.venv-judge-bw/bin/python" "$CODE" \
  --source "$P/prior331_runs/translation_repair_20260920/gpt56sol_remaining17_v3/translations.jsonl" \
  --out "$P/prior331_runs/translation_repair_20260920/gpt56sol_remaining17_v3_milmmt_backtranslation" \
  --model-path "$P/hf-cache/hub/models--xiaomi-research--MiLMMT-46-12B-v1.0/snapshots/a27dbbb37142ff076990820a1c9f0827beb5d6ea" \
  --gpu "$GPU" --expected-count 17
