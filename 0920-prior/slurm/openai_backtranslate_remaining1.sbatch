#!/bin/bash
#SBATCH -J prior331-gptback1
#SBATCH -p all
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH -t 01:00:00
#SBATCH -D /home/ljk98/POLY
#SBATCH -o logs/slurm-%j-%x.out
set -euo pipefail
P=/home/ljk98/POLY;R=$P/ICLR-POLY-0920
set -a;source "$P/.secrets/tokens.env";set +a
CODE=$R/experiments_suite/exp08_willingness394_milmmt_v3/aligned/model_specific/openai_backtranslate_remaining.py
srun "$P/.venv-transfer/bin/python" "$CODE" \
 --source "$P/prior331_runs/translation_repair_20260920/gpt56sol_remaining1_v5/translations.jsonl" \
 --all-rows --out "$P/prior331_runs/translation_repair_20260920/gpt56sol_remaining1_v5_gpt_backtranslation" \
 --expected-source-count 1 --expected-rejected-count 1
