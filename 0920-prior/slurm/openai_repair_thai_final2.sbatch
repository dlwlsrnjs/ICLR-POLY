#!/bin/bash
#SBATCH -J prior331-thai-final2
#SBATCH -p all
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH -t 01:00:00
#SBATCH -D /home/ljk98/POLY
#SBATCH -o logs/slurm-%j-%x.out
set -euo pipefail
P=/home/ljk98/POLY;R=$P/ICLR-POLY-0920;set -a;source "$P/.secrets/tokens.env";set +a
srun "$P/.venv-transfer/bin/python" "$R/experiments_suite/exp08_willingness394_milmmt_v3/aligned/model_specific/openai_repair_remaining.py" \
 --source "$P/prior331_runs/translation_repair_20260920/gpt56sol_remaining1_v6_gpt_backtranslation/translations.jsonl" \
 --validation "$P/prior331_runs/translation_repair_20260920/gpt56sol_remaining1_v6_qwen32_validation/judgments.jsonl" \
 --out "$P/prior331_runs/translation_repair_20260920/gpt56sol_remaining1_v7" --expected-source-count 1 --expected-rejected-count 1 --workers 1
