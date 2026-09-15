#!/bin/bash
# Held-out warm-start SIGNAL probes: the answer-frame signals on the innocuous set and the
# safety-adjacent (borderline) set, for both collections, so the held-out prior uses the SAME
# candidate recipe the nine-panel LOO selection chose (MJ: benign-recon x borderline-persona_hold;
# Lingua: benign-recon x fiction_hold). Cheap: 16 harmless requests per frame, no judge. Idempotent.
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ENFORCE_EAGER=1 VLLM_USE_FLASHINFER_SAMPLER=0
pick(){ need=$1; while :; do
  g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -v n=$need '$2>=n{gsub(/,/,"",$1); print $1; exit}')
  [ -n "$g" ] && { echo $g; return 0; }; sleep 30; done; }
# $1 tag $2 ref $3 trust $4 util $5 need $6 answer-lang $7 outdir $8 mml $9 request-set-flag
go(){ tag="$1"; ref="$2"; TR="$3"; util="$4"; need="$5"; al="$6"; outd="$7"; mml="${8:-4096}"; rs="$9"
  [ -f $outd/$tag.json ] && { echo "OK $outd/$tag"; return; }
  for a in 1 2 3; do g=$(pick $need); echo "[$(date +%H:%M:%S)] $outd $tag ($al) GPU$g try $a"
    CUDA_VISIBLE_DEVICES=$g $VP/python scripts/benign_signals_probe.py --target "$ref" --tag $tag \
      $TR --tokenizer-mode auto --answer-lang $al --util $util --max-model-len $mml --outdir $outd $rs && break
    sleep 20; done; }
model(){ tag="$1"; ref="$2"; TR="$3"; util="$4"; need="$5"; mml="${6:-4096}"
  # innocuous signals
  go $tag "$ref" "$TR" $util $need Swahili   results/benign_signals_mj_20260907 $mml ""
  go $tag "$ref" "$TR" $util $need Norwegian results/benign_signals_20260907    $mml ""
  # safety-adjacent (borderline) signals
  go $tag "$ref" "$TR" $util $need Swahili   results/benign_borderline_mj_20260907 $mml "--request-set borderline"
  go $tag "$ref" "$TR" $util $need Norwegian results/benign_borderline_20260907    $mml "--request-set borderline"
}
model phi35_mini microsoft/Phi-3.5-mini-instruct          "--trust-remote-code" 0.45 40000
model mistral7b  mistralai/Mistral-7B-Instruct-v0.3        "--trust-remote-code" 0.45 40000
model falcon3_7b tiiuae/Falcon3-7B-Instruct                "--trust-remote-code" 0.45 40000
model glm4_9b    THUDM/glm-4-9b-chat-hf                    "--trust-remote-code" 0.55 46000
model mistral24b mistralai/Mistral-Small-24B-Instruct-2501 "--trust-remote-code" 0.90 74000 3072
echo HELDOUT_SIGNALS_DONE
