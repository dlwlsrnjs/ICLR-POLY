#!/bin/bash
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -u jinkwon -f 'extra_arms_run.sh' > /dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] extra arms done; judge-leak test"
pick() { while :; do g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk '$2>=40000{gsub(/,/,"",$1);print $1;exit}'); [ -n "$g" ] && { echo $g; return; }; sleep 45; done; }
g=$(pick); CUDA_VISIBLE_DEVICES=$g $VP/python scripts/judge_leak_test.py
echo POST_EXTRA_DONE
