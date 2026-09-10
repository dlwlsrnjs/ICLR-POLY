#!/bin/bash
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
while pgrep -u jinkwon -f 'post_extra_chain.sh|extra_arms_run.sh|judge_leak_test.py' > /dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] judge-leak done; hi_en generation"
bash scripts/triple_hien_run.sh
echo POST_LEAK_DONE
