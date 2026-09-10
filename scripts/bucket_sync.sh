#!/usr/bin/env bash
# Sync the PolyJigsaw working tree to the PRIVATE Hugging Face bucket so a session on another machine
# can pull code + data and continue. The bucket is PRIVATE (jin-kwon/poly); it holds restricted
# research artifacts for the owner's own reuse, not public redistribution.
#
# NEVER uploaded (safety / hygiene): API keys and secrets, HF/token caches, python caches, git internals.
# Uploaded: code (scripts/, experiments_suite/), paper/, docs/, results/ aggregates AND raw, and
# private_artifacts/ (harm grids + benign probes). Because the bucket is private and owner-only, the
# restricted files travel with it; keep the bucket private.
#
# Prereqs (run once, interactive -- only you can authenticate):
#   hf auth login            # paste your HF token
# The bucket "sync" verb ships with the newer hf CLI shown in the bucket UI. If your hf lacks `sync`,
# install/upgrade:  pip install -U "huggingface_hub[cli]"   (or the bucket page's install line)
set -euo pipefail
cd "$(dirname "$0")/.."                      # -> PolyJigsaw/
BUCKET=${BUCKET:-hf://buckets/jin-kwon/poly}

# Build a clean staging copy that excludes secrets/caches, then sync that.
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
rsync -a --delete \
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude '.secrets' --exclude '*.key' --exclude '*api_key*' --exclude '*token*' \
  --exclude '.venv*' --exclude '.vllm_env' --exclude '.internlm_env' \
  --exclude 'hf_cache' --exclude '.cache' \
  ./ "$STAGE/PolyJigsaw/"

# Safety scan: refuse if anything that looks like a key slipped into staging.
if grep -rIl -E 'sk-proj-|sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}' "$STAGE" 2>/dev/null | head -1 | grep -q .; then
  echo "ABORT: possible secret found in staging; not uploading." >&2; exit 3
fi

echo ">> syncing $STAGE/PolyJigsaw -> $BUCKET/PolyJigsaw"
if hf sync --help >/dev/null 2>&1; then
  hf sync "$STAGE/PolyJigsaw" "$BUCKET/PolyJigsaw"
else
  echo "NOTE: this hf CLI has no 'sync'. Use the bucket page's CLI, or upload-large-folder to a dataset repo." >&2
  echo "Manual bucket command (from the bucket UI):" >&2
  echo "  hf sync ./PolyJigsaw $BUCKET/PolyJigsaw" >&2
  exit 4
fi
echo "BUCKET_SYNC_DONE"
