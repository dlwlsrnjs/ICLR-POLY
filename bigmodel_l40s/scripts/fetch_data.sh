#!/usr/bin/env bash
# Fetch the dataset inputs the collection needs (harmful text is NOT in the git repo, by policy).
# Source of truth: the PRIVATE HF bucket  jin-kwon/poly . You need a bucket-capable hf CLI and a
# token with access. Run from the repo root (or anywhere; it cd's to the repo root).
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${HF_TOKEN:?export HF_TOKEN=<your huggingface token with access to jin-kwon/poly>}"

pip install -U "huggingface_hub[cli]" >/dev/null   # gives `hf sync` / `hf buckets` (huggingface_hub >= 1.30)
export HF_TOKEN
echo "[fetch] pulling dataset inputs from private bucket jin-kwon/poly ..."

# Only the small inputs the collection reads (weights are re-downloaded separately, see README step 2):
for sub in \
  private_artifacts/multijail_v1 \
  private_artifacts/panel_v2 \
  results/lang_rank_20260905 ; do
  echo "  - $sub"
  hf sync "hf://buckets/jin-kwon/poly/PolyJigsaw/$sub" "./$sub"
done

echo "[fetch] verifying required files ..."
req=(
 private_artifacts/multijail_v1/harm_grid.jsonl
 private_artifacts/multijail_v1/benign_probe.jsonl
 private_artifacts/multijail_v1/resource_order.json
 private_artifacts/panel_v2/harm_grid.jsonl
 private_artifacts/panel_v2/benign_probe.jsonl
 results/lang_rank_20260905/resource_order.json
)
miss=0
for f in "${req[@]}"; do [ -s "$f" ] && printf "  OK  %s\n" "$f" || { printf "  MISSING %s\n" "$f"; miss=1; }; done
[ "$miss" = 0 ] && echo "[fetch] all dataset inputs present." || { echo "[fetch] MISSING files above"; exit 1; }

cat <<'EOF'

If you do NOT have bucket access, obtain the two datasets officially instead:
  * MultiJail (Deng et al.)          - official source per its license
  * Lingua-SafetyBench (Text partition) - official source per its license
then rebuild these local files with the pilot builders documented in docs/REPRODUCE.md
(scripts/prepare_lingua_text.py etc.). The collection reads exactly the 6 files listed above.
EOF
