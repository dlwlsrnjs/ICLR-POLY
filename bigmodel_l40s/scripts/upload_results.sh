#!/usr/bin/env bash
# Push the L40S results back so they merge into the panel. Two sinks:
#   (1) the PRIVATE HF bucket jin-kwon/poly  (recommended: keeps harmful raw outputs OUT of git)
#   (2) a local rsync target, if you prefer to copy onto the shared box directly.
# Raw per-response outputs (results/attack/_raw/*.jsonl, mode 0600) are RESTRICTED — bucket only,
# never git. Run from repo root (script cd's there).
set -euo pipefail
cd "$(dirname "$0")/../.."
R=bigmodel_l40s/results
[ -d "$R" ] || { echo "no $R yet - run run_l40s.sh first"; exit 1; }

echo "[upload] summary of what will be shipped:"
find "$R" -name '*.json' ! -path '*/_raw/*' | wc -l | xargs echo "  score json files:"
find "$R" -path '*/_raw/*' -name '*.jsonl'  | wc -l | xargs echo "  raw jsonl (RESTRICTED):"

if [ "${SINK:-bucket}" = "bucket" ]; then
  : "${HF_TOKEN:?export HF_TOKEN=<token with write access to jin-kwon/poly>}"
  export HF_TOKEN
  echo "[upload] -> bucket jin-kwon/poly/PolyJigsaw/$R"
  hf sync "./$R" "hf://buckets/jin-kwon/poly/PolyJigsaw/$R"
  echo "[upload] done. On the shared box, pull with:"
  echo "  hf sync hf://buckets/jin-kwon/poly/PolyJigsaw/$R ./bigmodel_l40s/results"
  echo "  then MERGE into the panel:"
  echo "  rsync -a bigmodel_l40s/results/ experiments_suite/exp02_panel_collect/results/"
  echo "  python scripts/closed_compare.py repair-manifest --root experiments_suite/exp02_panel_collect/results"
else
  : "${RSYNC_DEST:?export RSYNC_DEST=user@sharedbox:/path/to/ICLR-POLY/experiments_suite/exp02_panel_collect/results/}"
  echo "[upload] rsync -> $RSYNC_DEST"
  rsync -a "$R"/ "$RSYNC_DEST"
  echo "[upload] then on the shared box: python scripts/closed_compare.py repair-manifest --root experiments_suite/exp02_panel_collect/results"
fi
