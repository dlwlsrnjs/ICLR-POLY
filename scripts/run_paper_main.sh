#!/usr/bin/env bash
# PolyJigsaw paper-grade MAIN experiment (frozen config: experiments/paper_protocol.json).
#
# One unified head-to-head generation per (dataset, target) so every number in the
# main comparison table comes from an identical run. Attack methods compared:
#   english_direct                              (no-attack reference)
#   translated_direct_<all langs>               (multilingual translation baseline)
#   csrt_k1/k2/k3 + csrt_all                    (CSRT code-switching baseline)
#   interleave_ordered_n4/n6, shuffled_n4       (ours: deinterleaving game)
#   slot_k1/k2/k3                               (ours: inline-tile game; matched to csrt_k)
#   nogame_ordered_n4                           (ablation: MIDAS w/o Game-Style)
# Judges: reconstruction gate + Qwen3Guard (primary) + MD-Judge (cross) + HR(0-5).
#
# Single GPU, serial. Every stage skipped if its output exists (resumable).
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache
set -uo pipefail
VPY=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python
GPU=${GPU:-1}
LINGUA_BASE=private_artifacts/full_textdom/slot_game.jsonl   # 2879, carries official_slot_alignments
LINGUA=private_artifacts/paper_main/lingua_csrtmt.jsonl      # base + full-coverage MT-CSRT prompts
ATTAQ=/home/ubuntu/342/jinkwon/datasets/attaq/attaq_full_aligned.jsonl  # 1402, no span alignments
ROOT=private_artifacts/paper_main
mkdir -p "$ROOT"
log(){ echo "[$(date +%H:%M:%S)] PAPER: $*"; }

# ---- Stage 0: full-coverage MT-CSRT baseline prompts (NLLB) ----
if [ ! -f "$LINGUA" ]; then
  log "[stage0] building full-coverage MT-CSRT prompts (NLLB-1.3B)"
  CUDA_VISIBLE_DEVICES=$GPU python3 scripts/build_csrt_mt.py \
    --data "$LINGUA_BASE" --output "$LINGUA" --ks 2 3 \
    --model facebook/nllb-200-distilled-1.3B --device cuda:0 --batch-size 64 \
    > "$ROOT/csrt_mt.log" 2>&1 || { log "[stage0] MT-CSRT build failed; falling back to base data"; LINGUA="$LINGUA_BASE"; }
fi

# Full method set (Lingua has span alignments -> csrt/slot apply).
LINGUA_TRANS="Arabic Chinese Finnish French German Japanese Norwegian Russian Spanish"
# AttaQ has 6 non-English official translations, no span alignments -> no csrt/slot.
ATTAQ_TRANS="French Spanish German Czech Slovenian Valencian"

judges(){  # $1=workdir  (runs recon, guard, md, hr on target_outputs.jsonl)
  local W="$1"
  [ -f "$W/recon/restricted_reconstruction_audit.jsonl" ] || { log "  recon judge";
    CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/judge_reconstruction_equivalence.py \
      --input "$W/target_outputs.jsonl" --outdir "$W/recon" \
      --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --batch-size 48 > "$W/recon.log" 2>&1; }
  [ -f "$W/guard/restricted_qwen3guard_audit.jsonl" ] || { log "  Qwen3Guard judge";
    CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/rejudge_qwen3guard_official.py \
      --input "$W/target_outputs.jsonl" --outdir "$W/guard" \
      --model Qwen/Qwen3Guard-Gen-8B --device cuda:0 --batch-size 48 \
      --assistant-field answer_section > "$W/guard.log" 2>&1; }
  [ -f "$W/mdjudge/restricted_mdjudge_audit.jsonl" ] || { log "  MD-Judge";
    CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/judge_mdjudge.py \
      --input "$W/target_outputs.jsonl" --outdir "$W/mdjudge" \
      --assistant-field answer_section --gpu-memory-utilization 0.90 > "$W/mdjudge.log" 2>&1; }
  [ -f "$W/hr/restricted_hr_audit.jsonl" ] || { log "  Harmfulness Rating";
    CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/judge_harmfulness_rating.py \
      --input "$W/target_outputs.jsonl" --outdir "$W/hr" \
      --model Qwen/Qwen2.5-7B-Instruct --assistant-field answer_section \
      --gpu-memory-utilization 0.90 > "$W/hr.log" 2>&1; }
}

run_cell(){  # $1=name $2=data $3=target $4=trans $5=extra_flags $6=summary_prefix
  local NAME="$1" DATA="$2" TARGET="$3" TRANS="$4" EXTRA="$5" PFX="$6"
  local W="$ROOT/$NAME"; mkdir -p "$W"
  if [ ! -f "$W/target_outputs.jsonl" ]; then
    log "[$NAME] generation ($TARGET)"
    CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/run_text_gated_vllm.py \
      --data "$DATA" --outdir "$W/gen" --target "$TARGET" \
      --interleave-ns 4 6 --interleave-conditions ordered shuffled \
      --translated-langs $TRANS $EXTRA \
      --gpu-memory-utilization 0.90 > "$W/gen.log" 2>&1 \
      || { log "[$NAME] GEN FAILED"; tail -20 "$W/gen.log"; return 1; }
    cp "$W/gen/restricted_target_outputs.jsonl" "$W/target_outputs.jsonl"; chmod 600 "$W/target_outputs.jsonl"
    log "[$NAME] $(wc -l < "$W/target_outputs.jsonl") gens"
  fi
  judges "$W"
  log "[$NAME] summaries"
  python3 scripts/summarize_polyjig_comparison.py \
    --recon-audit "$W/recon/restricted_reconstruction_audit.jsonl" \
    --guard-audit "$W/guard/restricted_qwen3guard_audit.jsonl" \
    --baseline english_direct --output "results/${PFX}_gated_summary.json" > "$W/summary.log" 2>&1
  python3 scripts/compare_methods.py \
    --recon-audit "$W/recon/restricted_reconstruction_audit.jsonl" \
    --guard-audit "$W/guard/restricted_qwen3guard_audit.jsonl" \
    --mdjudge-audit "$W/mdjudge/restricted_mdjudge_audit.jsonl" \
    --hr-audit "$W/hr/restricted_hr_audit.jsonl" \
    --output "results/${PFX}_method_comparison.json" >> "$W/summary.log" 2>&1
  # Config-freeze reporting: main table on held-out TEST split, dev reported separately.
  local SPLITFILE=experiments/lingua_splits.json
  case "$NAME" in attaq*) SPLITFILE=experiments/attaq_splits.json;; esac
  for SP in test dev; do
    python3 scripts/compare_methods.py \
      --recon-audit "$W/recon/restricted_reconstruction_audit.jsonl" \
      --guard-audit "$W/guard/restricted_qwen3guard_audit.jsonl" \
      --mdjudge-audit "$W/mdjudge/restricted_mdjudge_audit.jsonl" \
      --hr-audit "$W/hr/restricted_hr_audit.jsonl" \
      --split-file "$SPLITFILE" --split "$SP" \
      --output "results/${PFX}_method_comparison_${SP}.json" >> "$W/summary.log" 2>&1 || true
  done
  log "[$NAME] done -> results/${PFX}_* (all/test/dev)"
}

# ---- Lingua (primary benchmark): full method set incl. CSRT + slot ----
run_cell lingua_qwen   "$LINGUA" Qwen/Qwen2.5-7B-Instruct          "$LINGUA_TRANS" \
  "--with-csrt --csrt-ks 1 2 3 --with-nogame" paper_lingua_qwen
run_cell lingua_mistral "$LINGUA" mistralai/Mistral-7B-Instruct-v0.3 "$LINGUA_TRANS" \
  "--with-csrt --csrt-ks 1 2 3 --with-nogame" paper_lingua_mistral
run_cell lingua_qwen3   "$LINGUA" Qwen/Qwen3-8B                        "$LINGUA_TRANS" \
  "--with-csrt --csrt-ks 1 2 3 --with-nogame" paper_lingua_qwen3

# ---- AttaQ (external generalisation): reuse the queue's AttaQ full generation ----
# (run_attaq_gated_vllm uses AttaQ's own 7-language set; the text runner's fixed
# Lingua language set would skip every interleave condition here.) We only add the
# cross judge + HR + method comparison on top of the queue's recon+guard audits.
AQW=private_artifacts/attaq_full
if [ -f "$AQW/target_outputs.jsonl" ]; then
  ln -sf ../attaq_full "$ROOT/attaq_qwen" 2>/dev/null || true
  [ -f "$AQW/mdjudge/restricted_mdjudge_audit.jsonl" ] || { log "[attaq] MD-Judge";
    CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/judge_mdjudge.py \
      --input "$AQW/target_outputs.jsonl" --outdir "$AQW/mdjudge" \
      --assistant-field answer_section --gpu-memory-utilization 0.90 > "$AQW/mdjudge.log" 2>&1; }
  [ -f "$AQW/hr/restricted_hr_audit.jsonl" ] || { log "[attaq] Harmfulness Rating";
    CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/judge_harmfulness_rating.py \
      --input "$AQW/target_outputs.jsonl" --outdir "$AQW/hr" \
      --model Qwen/Qwen2.5-7B-Instruct --assistant-field answer_section \
      --gpu-memory-utilization 0.90 > "$AQW/hr.log" 2>&1; }
  python3 scripts/compare_methods.py \
    --recon-audit "$AQW/recon/restricted_reconstruction_audit.jsonl" \
    --guard-audit "$AQW/guard/restricted_qwen3guard_audit.jsonl" \
    --mdjudge-audit "$AQW/mdjudge/restricted_mdjudge_audit.jsonl" \
    --hr-audit "$AQW/hr/restricted_hr_audit.jsonl" \
    --output results/paper_attaq_qwen_method_comparison.json > "$AQW/paper_compare.log" 2>&1
  log "[attaq] done -> results/paper_attaq_qwen_method_comparison.json"
else
  log "[attaq] SKIP: queue AttaQ output not found ($AQW/target_outputs.jsonl)"
fi

# ---- input detectability + defense on the primary cell ----
DW="$ROOT/lingua_qwen"
[ -f "$DW/input_detect/summary.json" ] || { log "[detectability] input-stage guard on Lingua/Qwen prompts";
  CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/judge_input_detectability.py \
    --input "$DW/target_outputs.jsonl" --outdir "$DW/input_detect" \
    --model Qwen/Qwen3Guard-Gen-8B --device cuda:0 --batch-size 32 \
    > "$DW/input_detect.log" 2>&1; }

DEF="$ROOT/lingua_qwen_defense"; mkdir -p "$DEF"
if [ ! -f results/paper_lingua_qwen_defense_summary.json ]; then
  log "[defense] Self-Reminder system prompt, interleave ordered n4/n6 + csrt_all + slot on Lingua/Qwen"
  if [ ! -f "$DEF/target_outputs.jsonl" ]; then
    CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/run_text_gated_vllm.py \
      --data "$LINGUA" --outdir "$DEF/gen" --target Qwen/Qwen2.5-7B-Instruct \
      --interleave-ns 4 6 --interleave-conditions ordered \
      --translated-langs Finnish --with-csrt --csrt-ks 2 --system-prompt self_reminder \
      --gpu-memory-utilization 0.90 > "$DEF/gen.log" 2>&1 \
      && { cp "$DEF/gen/restricted_target_outputs.jsonl" "$DEF/target_outputs.jsonl"; chmod 600 "$DEF/target_outputs.jsonl"; } \
      || { log "[defense] GEN FAILED"; tail -20 "$DEF/gen.log"; }
  fi
  if [ -f "$DEF/target_outputs.jsonl" ]; then
    judges "$DEF"
    python3 scripts/summarize_polyjig_comparison.py \
      --recon-audit "$DEF/recon/restricted_reconstruction_audit.jsonl" \
      --guard-audit "$DEF/guard/restricted_qwen3guard_audit.jsonl" \
      --baseline english_direct --output results/paper_lingua_qwen_defense_summary.json > "$DEF/summary.log" 2>&1
  fi
fi

log "PAPER MAIN COMPLETE"
