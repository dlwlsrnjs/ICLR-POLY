#!/bin/bash
#SBATCH -J prior331-gptrepair17
#SBATCH -p all
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -D /home/ljk98/POLY
#SBATCH -o logs/slurm-%j-%x.out
set -euo pipefail
P=/home/ljk98/POLY
R=$P/ICLR-POLY-0920
set -a; source "$P/.secrets/tokens.env"; set +a
: "${OPENAI_API_KEY:?OPENAI_API_KEY is not set}"
CODE=$R/experiments_suite/exp08_willingness394_milmmt_v3/aligned/model_specific/openai_repair_remaining.py
srun "$P/.venv-transfer/bin/python" "$CODE" \
  --source "$P/prior331_runs/translation_repair_20260920/gpt56sol_remaining54_v2_milmmt_backtranslation/translations.jsonl" \
  --validation "$P/prior331_runs/translation_repair_20260920/gpt56sol_remaining54_v2_qwen32_validation/judgments.jsonl" \
  --out "$P/prior331_runs/translation_repair_20260920/gpt56sol_remaining17_v3" \
  --expected-source-count 54 --expected-rejected-count 17 --workers 8
