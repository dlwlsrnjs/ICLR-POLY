#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
exec 9>../willingness394_scenarios_v2/cache/gpu1.lock
flock -n 9 || { echo 'GPU1 collection lock is held'; exit 1; }
../../.venv_prior085/bin/python code/translate.py "$@"
