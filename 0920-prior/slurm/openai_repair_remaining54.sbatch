#!/bin/bash
#SBATCH -J prior331-gptrepair54
#SBATCH -p all
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -D /home/ljk98/POLY
#SBATCH -o logs/slurm-%j-%x.out
set -euo pipefail

POLY_ROOT=/home/ljk98/POLY
REPO=$POLY_ROOT/ICLR-POLY-0920
PY=$POLY_ROOT/.venv-transfer/bin/python
SOURCE=$POLY_ROOT/prior331_runs/translation_repair_20260920/gpt56sol_retranslated_430_v1_milmmt_backtranslation/translations.jsonl
VALIDATION=$POLY_ROOT/prior331_runs/translation_repair_20260920/gpt56sol_retranslated_430_v1_qwen32_validation/judgments.jsonl
OUT=$POLY_ROOT/prior331_runs/translation_repair_20260920/gpt56sol_remaining54_v2
CODE=$REPO/experiments_suite/exp08_willingness394_milmmt_v3/aligned/model_specific/openai_repair_remaining.py

set -a
source "$POLY_ROOT/.secrets/tokens.env"
set +a
: "${OPENAI_API_KEY:?OPENAI_API_KEY is not set}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "time=$(date -Is) host=$(hostname) job=$SLURM_JOB_ID"
git -C "$REPO" rev-parse HEAD
sha256sum "$CODE" "$SOURCE" "$VALIDATION"
srun "$PY" "$CODE" --source "$SOURCE" --validation "$VALIDATION" --out "$OUT" --expected-source-count 430 --expected-rejected-count 54 --workers 8
exit 0
