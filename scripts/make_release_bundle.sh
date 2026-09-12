#!/bin/bash
# Freeze everything needed to reproduce the paper's selector results into one directory.
#
#   bash scripts/make_release_bundle.sh [DEST]
#
# Copies the code that produces each number, the aggregate results those numbers come from, the
# generated tables and figures, and the manuscript. Raw generations and harmful items are NOT copied:
# they stay access-restricted, and the bundle records their paths and hashes instead so a holder of
# the restricted data can verify they match. Re-running the pipeline from the bundle reproduces every
# table without a GPU, because the expensive per-(target, configuration) results are included.
set -u
SRC=/home/ubuntu/342/jinkwon/poly/PolyJigsaw
DEST=${1:-/home/ubuntu/342/jinkwon/poly/release_20260908}
cd "$SRC" || exit 1
mkdir -p "$DEST"/{code,results,paper,docs}

say() { echo "=== $*"; }

# ---------------------------------------------------------------- code
say "code"
CODE=(
  # scoring rule and the two bandit loaders that every table reads
  arm_scoring.py mj_bandit_full.py lingua_bandit_full.py gp_bai.py
  # analysis that produces the paper's numbers
  bandit_bootstrap_ci.py query_efficiency.py benign_prior_selection.py
  make_selector_tables.py make_paper_figs.py audit_panel_integrity.py
  # harmless probes (the warm start)
  benign_arms_probe.py benign_signals_probe.py benign_compliance_probe.py
  build_mj_benign_probe.py
  benign_arms_run.sh benign_arms_big.sh benign_arms_mj_run.sh
  benign_signals_run.sh benign_borderline_run.sh
  # harmful evaluation of the configuration space
  sequential_add.py disorder_sweep.py combo_eval.py method_baselines_eval.py
  triple_combo_eval.py amount_disorder_2phase.py mj_2phase_generic.py
  run_mj_gpu.py big_models_run.sh lingua_triple_run.sh
  # judging
  online_live.py judge_mdjudge.py judge_crosscheck.py judge_crosscheck_rows.py
  judge_crosscheck_run.sh judge_crosscheck_rows_run.sh mdjudge_template_check.py
  # per-domain study
  domain_breakdown.py domain_breakdown_run.sh
  # shared construction helpers
  run_qwen_interleaving_probe.py run_polyjig_gated.py polyjig_pilot.py
)
for f in "${CODE[@]}"; do
  if [ -f "scripts/$f" ]; then cp -p "scripts/$f" "$DEST/code/"; else echo "  MISSING scripts/$f"; fi
done
cp -p requirements.txt "$DEST/code/" 2>/dev/null
cp -p requirements-collection.lock.txt "$DEST/code/" 2>/dev/null

# ---------------------------------------------------------------- results
say "aggregate results"
# per-(configuration family, target) aggregates: the 90 files behind every table
for d in mj_sequential_20260906 mj_disorder_20260906 mj_combo_20260906 mj_method_20260906 \
         mj_triple_20260906 sequential_resource_20260906 disorder_sweep_20260906 combo_20260906 \
         method_baselines_v2_20260906 lingua_triple_20260907; do
  mkdir -p "$DEST/results/$d"
  for f in results/$d/*.json; do
    b=$(basename "$f")
    case "$b" in _raw*) continue;; esac          # raw generations stay restricted
    cp -p "$f" "$DEST/results/$d/" 2>/dev/null
  done
done
# harmless probes, selector summaries, figures
for d in benign_arms_20260907 benign_arms_mj_20260907 benign_signals_20260907 \
         benign_signals_mj_20260907 benign_borderline_20260907 benign_borderline_mj_20260907 \
         domain_breakdown_20260907; do
  [ -d "results/$d" ] || continue
  mkdir -p "$DEST/results/$d"
  for f in results/$d/*.json; do
    b=$(basename "$f"); case "$b" in _raw*) continue;; esac
    cp -p "$f" "$DEST/results/$d/" 2>/dev/null
  done
done
for f in mj_bandit_full_20260906.json lingua_bandit_full_20260906.json bandit_ci_20260907.json \
         ablation_corr_20260907.json query_efficiency_20260907.json \
         benign_prior_selection_20260907.json; do
  [ -f "results/$f" ] && cp -p "results/$f" "$DEST/results/"
done
mkdir -p "$DEST/results/figs" && cp -p results/figs/*.png "$DEST/results/figs/" 2>/dev/null

# ---------------------------------------------------------------- paper
say "paper"
cp -p paper/polyjigsaw_iclr2026.tex paper/references.bib "$DEST/paper/"
cp -p paper/tab_*.tex paper/app_*.tex paper/selector_numbers.tex paper/query_numbers.tex "$DEST/paper/" 2>/dev/null
cp -p paper/iclr2026_conference.sty paper/iclr2026_conference.bst paper/math_commands.tex \
      paper/fancyhdr.sty paper/natbib.sty "$DEST/paper/" 2>/dev/null
mkdir -p "$DEST/paper/figures" && cp -p paper/figures/*.png paper/figures/*.pdf "$DEST/paper/figures/" 2>/dev/null
[ -f paper/build/polyjigsaw_iclr2026.pdf ] && cp -p paper/build/polyjigsaw_iclr2026.pdf "$DEST/paper/"

# ---------------------------------------------------------------- docs
say "docs"
for f in DATA_AUDIT_2026-09-07.md JUDGE_CROSSCHECK_2026-09-07.md PANEL9_RESULTS_2026-09-07.md \
         PAPER_TABLE_DRAFT_2026-09-07.md; do
  [ -f "docs/$f" ] && cp -p "docs/$f" "$DEST/docs/"
done

# ---------------------------------------------------------------- provenance
say "provenance"
{
  echo "# provenance for the release bundle"
  echo "built: $(date -Is)"
  echo "host: $(hostname)  user: $(id -un)"
  echo
  echo "## restricted inputs (not copied; hashes so a holder can verify)"
  for f in private_artifacts/multijail_v1/harm_grid.jsonl private_artifacts/panel_v2/harm_grid.jsonl \
           private_artifacts/multijail_v1/benign_probe.jsonl private_artifacts/panel_v2/benign_probe.jsonl \
           private_artifacts/multijail_v1/resource_order.json results/lang_rank_20260905/resource_order.json; do
    if [ -r "$f" ]; then echo "$(sha256sum "$f")"; else echo "UNREADABLE  $f"; fi
  done
  echo
  echo "## copied result files"
  find "$DEST/results" -name '*.json' | sort | while read -r f; do echo "$(sha256sum "$f" | sed "s|$DEST/||")"; done
} > "$DEST/PROVENANCE.txt"

say "done -> $DEST"
find "$DEST" -type f | wc -l | xargs echo "files:"
du -sh "$DEST" | cut -f1 | xargs echo "size:"
