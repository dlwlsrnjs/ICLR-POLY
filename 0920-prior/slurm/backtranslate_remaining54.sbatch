#!/bin/bash
#SBATCH -J prior331-back54
#SBATCH -p rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH -t 01:00:00
#SBATCH -D /home/ljk98/POLY
#SBATCH -o logs/slurm-%j-%x.out
set -euo pipefail

POLY_ROOT=/home/ljk98/POLY
REPO=$POLY_ROOT/ICLR-POLY-0920
PY=$POLY_ROOT/.venv-judge-bw/bin/python
SOURCE=$POLY_ROOT/prior331_runs/translation_repair_20260920/gpt56sol_remaining54_v2/translations.jsonl
OUT=$POLY_ROOT/prior331_runs/translation_repair_20260920/gpt56sol_remaining54_v2_milmmt_backtranslation
MODEL=$POLY_ROOT/hf-cache/hub/models--xiaomi-research--MiLMMT-46-12B-v1.0/snapshots/a27dbbb37142ff076990820a1c9f0827beb5d6ea
CODE=$REPO/experiments_suite/exp08_willingness394_milmmt_v3/aligned/model_specific/backtranslate_openai_repairs.py
GPU=${CUDA_VISIBLE_DEVICES:?Slurm did not assign CUDA_VISIBLE_DEVICES}
case "$GPU" in *','*) echo "Expected one GPU, got $GPU" >&2; exit 2;; esac
export HF_HOME=$POLY_ROOT/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "time=$(date -Is) host=$(hostname) job=$SLURM_JOB_ID gpu=$GPU"
sha256sum "$CODE" "$SOURCE"
srun "$PY" "$CODE" --source "$SOURCE" --out "$OUT" --model-path "$MODEL" --gpu "$GPU" --expected-count 54
exit 0
