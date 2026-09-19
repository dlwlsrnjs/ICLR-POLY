#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CONFIG="${1:-configs/legacy_394.json}"
bash run.sh align-qa --config "$CONFIG"
../../.venv_prior085/bin/python code/manage.py build --config "$CONFIG"
bash run.sh collect --config "$CONFIG"
bash judge.sh "$CONFIG" wildguard
bash judge.sh "$CONFIG" behavior
../../.venv_prior085/bin/python code/summarize.py --config "$CONFIG"
