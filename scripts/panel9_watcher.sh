#!/bin/bash
# When the 32B/27B panel finishes, extend BOTH bandits to 9 models and regenerate CI + figures.
# Purely offline once big_models is done. Idempotent.
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
NINE='["qwen25_3b","qwen25_7b","qwen25_14b","qwen25_32b","llama32_3b_it","llama31_8b_it","gemma2_2b_it","gemma2_9b_it","gemma2_27b"]'
for w in $(seq 1 720); do
  if grep -q BIG_MODELS_ALL_DONE logs/big_models.log 2>/dev/null && [ ! -f results/PANEL9_DONE.marker ]; then
    # verify all 9-model arm files exist for both datasets before extending
    ok=1
    for t in qwen25_32b gemma2_27b; do
      for d in mj_sequential mj_disorder mj_combo mj_method mj_triple sequential_resource disorder_sweep combo method_baselines_v2 lingua_triple; do
        [ -f results/${d}_20260906/$t.json ] || [ -f results/${d}_20260907/$t.json ] || ok=0
      done
    done
    if [ "$ok" = 1 ]; then
      echo "[$(date +%H:%M:%S)] extending bandits to 9 models"
      sed -i "s/^STRONG = \[.*/STRONG = $NINE/" scripts/mj_bandit_full.py
      sed -i "s/^MODELS = \[.*/MODELS = $NINE/" scripts/lingua_bandit_full.py
      $VP/python scripts/mj_bandit_full.py > logs/panel9_mj.log 2>&1
      $VP/python scripts/lingua_bandit_full.py > logs/panel9_lingua.log 2>&1
      $VP/python scripts/bandit_bootstrap_ci.py > logs/panel9_ci.log 2>&1
      $VP/python scripts/make_paper_figs.py > logs/panel9_figs.log 2>&1
      touch results/PANEL9_DONE.marker
      echo "PANEL9_DONE"; break
    else
      echo "[$(date +%H:%M:%S)] big_models done but some 9-model arm files missing; waiting"
    fi
  fi
  grep -q BIG_MODELS_ALL_DONE logs/big_models.log 2>/dev/null || echo "[$(date +%H:%M:%S)] waiting for 32B/27B panel"
  sleep 60
done
