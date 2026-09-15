#!/bin/bash
# Regenerate the selector pipeline after restricting LOTO to deployable candidates
# (benign_prior_selection filter). Order handles the circular dep (selection <- bandit via prior_for).
# Run as jinkwon (reads per-model 0700 dirs, writes its own jinkwon-owned paper/*.tex). CPU, deterministic.
set -e
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
echo "[1/5] benign_prior_selection"; python3 scripts/benign_prior_selection.py >/dev/null
echo "[2/5] mj_bandit_full + lingua_bandit_full"; python3 scripts/mj_bandit_full.py >/dev/null; python3 scripts/lingua_bandit_full.py >/dev/null
echo "[3/5] bandit_bootstrap_ci"; python3 scripts/bandit_bootstrap_ci.py >/dev/null
echo "[4/5] make_selector_tables"; python3 scripts/make_selector_tables.py >/dev/null
echo "[5/5] query_efficiency"; python3 scripts/query_efficiency.py >/dev/null 2>&1 || echo "  (query_efficiency err/skip)"
echo "REGEN_DONE $(date +%H:%M:%S)"
