#!/usr/bin/env bash
# Wait for an already-running L40S matrix collector, then copy its complete
# aggregate and restricted raw outputs to the owner-only HF bucket.
set -euo pipefail
cd "$(dirname "$0")/../.."

MATRIX_PID="${1:?usage: sync_private_after_matrix.sh <matrix-pid>}"
HF_BIN="${HF_BIN:-hf}"
SOURCE="experiments_suite/exp02_panel_collect/results"
DESTINATION="hf://buckets/jin-kwon/poly/PolyJigsaw/experiments_suite/exp02_panel_collect/results"
STATUS="$SOURCE/private_bucket_sync_status_l40s.tsv"

printf '%s\tWAIT matrix_pid=%s\n' "$(date -Iseconds)" "$MATRIX_PID" >> "$STATUS"
while kill -0 "$MATRIX_PID" 2>/dev/null; do
  sleep 60
done

if grep -q $'\tQUEUE_FINISHED$' "$SOURCE/harmful_matrix_status_l40s.tsv"; then
  state="complete"
else
  state="partial_after_runner_exit"
fi
printf '%s\tSTART state=%s destination=%s\n' "$(date -Iseconds)" "$state" "$DESTINATION" >> "$STATUS"
"$HF_BIN" sync "$SOURCE" "$DESTINATION"
printf '%s\tCOMPLETE state=%s destination=%s\n' "$(date -Iseconds)" "$state" "$DESTINATION" >> "$STATUS"
