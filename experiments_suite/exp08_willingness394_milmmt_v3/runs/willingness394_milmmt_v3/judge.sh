#!/usr/bin/env bash
set -euo pipefail
umask 077
cd "$(dirname "$0")"
source env.sh
exec 9>../willingness394_scenarios_v2/cache/gpu1.lock
flock -n 9
PYTHON=../../.venv_prior085/bin/python
CONFIG="$1"
STAGE="${2:-wildguard}"
JUDGE_ROOT=$("$PYTHON" code/prepare_judges.py --config "$CONFIG")
if [ "$STAGE" = wildguard ]; then
 MODEL="$(realpath ../../model_cache/models--allenai--wildguard/snapshots/cbba4823f3e8020e5a74a5e29bf85072def6f2ff)"
elif [ "$STAGE" = behavior ]; then
 MODEL=/home/ubuntu/342/jinkwon/hf_cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd
else
 exit 2
fi
"$PYTHON" "$JUDGE_ROOT/code/batch_label.py" "$STAGE" --model "$MODEL"
