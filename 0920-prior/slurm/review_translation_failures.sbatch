#!/bin/bash
#SBATCH -J prior331-transqa
#SBATCH -p h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -t 02:00:00
#SBATCH -D /home/ljk98/POLY
#SBATCH -o logs/slurm-%j-%x.out
set -euo pipefail

POLY_ROOT=/home/ljk98/POLY
REPO=$POLY_ROOT/ICLR-POLY-0920
PY=$POLY_ROOT/.venv-gen/bin/python
SOURCE=$POLY_ROOT/0920-prior-overrefusal331-multilingual/data/translations_needs_review.jsonl
OUT=$POLY_ROOT/prior331_runs/translation_repair_20260920/rereview_v1
MODEL=$POLY_ROOT/hf-cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd
CODE=$REPO/experiments_suite/exp08_willingness394_milmmt_v3/aligned/model_specific/review_translation_failures.py
GPU=${CUDA_VISIBLE_DEVICES:?Slurm did not assign CUDA_VISIBLE_DEVICES}
case "$GPU" in *','*) echo "Expected one GPU, got $GPU" >&2; exit 2;; esac

export HF_HOME=$POLY_ROOT/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "time=$(date -Is) host=$(hostname) job=$SLURM_JOB_ID gpu=$GPU"
git -C "$REPO" rev-parse HEAD
sha256sum "$CODE" "$REPO/0920-prior/UNDERSTANDING_AXIS_V2.md"
nvidia-smi --id="$GPU" --query-gpu=index,name,driver_version,memory.total,memory.used --format=csv
srun "$PY" "$CODE" --source "$SOURCE" --out "$OUT" --model-path "$MODEL" --gpu "$GPU" --memory-utilization 0.90
exit 0
