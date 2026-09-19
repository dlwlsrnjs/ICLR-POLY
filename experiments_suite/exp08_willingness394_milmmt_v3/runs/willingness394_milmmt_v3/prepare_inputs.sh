#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
bash translate.sh
bash translation_qa.sh
bash run.sh align-qa --config configs/expanded_394.json
../../.venv_prior085/bin/python code/manage.py build --config configs/legacy_394.json
../../.venv_prior085/bin/python code/manage.py plan --config configs/expanded_394.json
