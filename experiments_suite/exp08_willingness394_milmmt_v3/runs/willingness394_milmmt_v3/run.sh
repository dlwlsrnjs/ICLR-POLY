#!/usr/bin/env bash
set -euo pipefail
umask 077
cd "$(dirname "$0")"
source env.sh
exec 9>../willingness394_scenarios_v2/cache/gpu1.lock
flock -n 9
../../.venv_prior085/bin/python code/check_layout.py "$@"
../../.venv_prior085/bin/python code/run.py "$@"
