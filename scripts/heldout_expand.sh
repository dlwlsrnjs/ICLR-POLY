#!/bin/bash
# QUEUED held-out expansion (generation only). Reuses heldout_run.sh's env + do_model.
# GPU-guarded (pick_gpu waits for free memory) + idempotent (skips finished outputs), so it can sit
# waiting until a GPU frees and then generate. Does NOT touch paper tables; run heldout_selector.py
# afterwards to fold the new models into the held-out analysis.
set +e
source /home/ubuntu/342/jinkwon/poly/PolyJigsaw/scripts/heldout_run.sh   # env, do_model, re-verifies the original 5 (idempotent)

echo "==== EXPANSION MODELS ===="
do_model internlm25 internlm/internlm2_5-7b-chat      "--trust-remote-code" 0.45 40000 44000
do_model olmo2_7b   allenai/OLMo-2-1124-7B-Instruct   "--trust-remote-code" 0.45 40000 44000
do_model zephyr7b   HuggingFaceH4/zephyr-7b-beta      "--trust-remote-code" 0.45 40000 44000
do_model qwen25_15b Qwen/Qwen2.5-1.5B-Instruct        "--trust-remote-code" 0.30 24000 44000
do_model yi34b      01-ai/Yi-1.5-34B-Chat             "--trust-remote-code" 0.90 74000 44000 3072
echo "EXPAND_GEN_DONE $(date +%H:%M:%S)"
echo "NEXT: run  sudo -n -u jinkwon python3 scripts/heldout_selector.py  to fold these into the held-out tables."
