#!/usr/bin/env bash
set -uo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/../.." && pwd)
RUN_ROOT=${POLY_RUN_ROOT:-$ROOT/target_prior_only_20260916/full_grid_5models_bf16}
PIPELINE=${POLY_PIPELINE_SCRIPT:-$HERE/run_five_models.sh}
PIPELINE_NEEDLE=${POLY_PIPELINE_NEEDLE:-$(basename "$PIPELINE")}
SYNC=${POLY_SYNC_SCRIPT:-$HERE/run_hf_sync.sh}
INTERVAL=${POLY_WATCH_INTERVAL:-30}
PIPELINE_PIDFILE=$RUN_ROOT/daemon.pid
SYNC_PIDFILE=$RUN_ROOT/hf-sync.pid
WATCH_STATUS=$RUN_ROOT/status/watchdog.status
WATCH_LOG=$RUN_ROOT/logs/watchdog.log
LOCK_FILE=$RUN_ROOT/watchdog.lock
FINAL_RECEIPT=$RUN_ROOT/hf_sync/FINAL_RECEIPT.json
MODELS=(qwen25_14b qwen25_32b gemma2_27b mistral24b phi3_medium_14b)
DATASETS=(lg mj)

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/status"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf 'watchdog already active\n' >&2
  exit 0
fi

log_event() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$WATCH_LOG"
}

pid_matches() {
  local pidfile=$1 needle=$2 pid command
  [[ -s "$pidfile" ]] || return 1
  pid=$(sed -n '1p' "$pidfile")
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  command=$(tr '\000' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)
  [[ "$command" == *"$needle"* ]]
}

compute_complete() {
  local model dataset count
  for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      count=$(find "$RUN_ROOT/raw/$dataset" -maxdepth 1 -type f -name "${model}_${dataset}__*.jsonl" 2>/dev/null | wc -l)
      ((count == 160)) || return 1
      count=$(find "$RUN_ROOT/judged/$dataset" -maxdepth 1 -type f -name "${model}_${dataset}__*.jsonl" 2>/dev/null | wc -l)
      ((count == 160)) || return 1
      count=$(find "$RUN_ROOT/judged/$dataset" -maxdepth 1 -type f -name "${model}_${dataset}__*.jsonl.meta.json" 2>/dev/null | wc -l)
      ((count == 160)) || return 1
    done
  done
}

start_pipeline() {
  rm -f "$PIPELINE_PIDFILE"
  start-stop-daemon --start --background --make-pidfile \
    --pidfile "$PIPELINE_PIDFILE" --startas "$PIPELINE" \
    --chdir "$ROOT" --output "$RUN_ROOT/logs/orchestrator-daemon.log"
  sleep 2
  if pid_matches "$PIPELINE_PIDFILE" "$PIPELINE_NEEDLE"; then
    log_event "pipeline restarted pid=$(sed -n '1p' "$PIPELINE_PIDFILE")"
    return 0
  fi
  log_event "pipeline restart failed"
  return 1
}

start_sync() {
  rm -f "$SYNC_PIDFILE"
  start-stop-daemon --start --background --make-pidfile \
    --pidfile "$SYNC_PIDFILE" --startas "$SYNC" \
    --chdir "$ROOT" --output "$RUN_ROOT/logs/hf-sync.log"
  sleep 2
  if pid_matches "$SYNC_PIDFILE" sync_hf_bucket.py; then
    log_event "HF sync restarted pid=$(sed -n '1p' "$SYNC_PIDFILE")"
    return 0
  fi
  log_event "HF sync restart failed"
  return 1
}

write_status() {
  local state=$1 pipeline_pid=none sync_pid=none tmp
  pid_matches "$PIPELINE_PIDFILE" "$PIPELINE_NEEDLE" && pipeline_pid=$(sed -n '1p' "$PIPELINE_PIDFILE")
  pid_matches "$SYNC_PIDFILE" sync_hf_bucket.py && sync_pid=$(sed -n '1p' "$SYNC_PIDFILE")
  tmp=$WATCH_STATUS.tmp.$$
  {
    printf 'state=%s\n' "$state"
    printf 'updated_at=%s\n' "$(date -Is)"
    printf 'pipeline_pid=%s\n' "$pipeline_pid"
    printf 'hf_sync_pid=%s\n' "$sync_pid"
  } >"$tmp"
  mv "$tmp" "$WATCH_STATUS"
}

log_event "watchdog started; adopting live processes when present"
while true; do
  state=running
  if ! pid_matches "$PIPELINE_PIDFILE" "$PIPELINE_NEEDLE"; then
    if compute_complete; then
      state=compute_complete
    else
      state=restarting_pipeline
      start_pipeline || state=pipeline_restart_failed
    fi
  fi

  if ! pid_matches "$SYNC_PIDFILE" sync_hf_bucket.py; then
    if [[ -s "$FINAL_RECEIPT" ]]; then
      state=${state}_sync_complete
    else
      start_sync || state=${state}_sync_restart_failed
    fi
  fi

  if compute_complete && [[ -s "$FINAL_RECEIPT" ]]; then
    write_status complete
    log_event "pipeline and HF Bucket receipt complete; watchdog exiting"
    exit 0
  fi

  write_status "$state"
  [[ ${POLY_WATCH_ONCE:-0} == 1 ]] && exit 0
  sleep "$INTERVAL"
done
