#!/bin/bash
# When lingua_hien finishes: swap the bandit loaders to English-answer role-separation arms, then
# regenerate every downstream artifact so the paper's numbers use only judge-reliable answers.
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
while pgrep -u jinkwon -f 'triple_hien_run.sh|post_leak_chain.sh' > /dev/null; do sleep 60; done
for t in qwen25_3b qwen25_7b qwen25_14b qwen25_32b llama32_3b_it llama31_8b_it gemma2_2b_it gemma2_9b_it gemma2_27b; do
  [ -f results/lingua_hien_20260908/$t.json ] || { echo "missing lingua_hien/$t; abort"; exit 1; }
done
echo "[$(date +%H:%M:%S)] applying English-answer arm swap"
$VP/python /tmp/claude-1001/-home-ubuntu-342-jinkwon-poly/752f3a14-e385-4430-bb88-44741d35c051/scratchpad/apply_hien.py || exit 1
echo "[$(date +%H:%M:%S)] recomputing"
$VP/python scripts/benign_prior_selection.py > /home/ubuntu/342/jinkwon/poly/fin_prior.log 2>&1
$VP/python scripts/bandit_bootstrap_ci.py    > /home/ubuntu/342/jinkwon/poly/fin_ci.log 2>&1
$VP/python scripts/query_efficiency.py       > /home/ubuntu/342/jinkwon/poly/fin_qe.log 2>&1
$VP/python scripts/probe_batch_size.py       >> /home/ubuntu/342/jinkwon/poly/fin_qe.log 2>&1
$VP/python scripts/robustness_checks.py      >> /home/ubuntu/342/jinkwon/poly/fin_qe.log 2>&1
$VP/python scripts/winners_curse.py          >> /home/ubuntu/342/jinkwon/poly/fin_qe.log 2>&1
$VP/python scripts/family_permutation.py     >> /home/ubuntu/342/jinkwon/poly/fin_qe.log 2>&1
$VP/python scripts/make_paper_figs.py        >> /home/ubuntu/342/jinkwon/poly/fin_qe.log 2>&1
$VP/python scripts/make_selector_tables.py   >> /home/ubuntu/342/jinkwon/poly/fin_qe.log 2>&1
echo FINALIZE_EN_DONE
