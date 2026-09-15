#!/bin/bash
# Missing warm-start SIGNAL + BORDERLINE (and yi34b's MJ recon) probes for the FIVE newest held-out
# models (yi34b, olmo2_7b, zephyr7b, qwen25_15b, qwen3_8b). The expansion + qwen3 held-out runs only
# produced the benign_arms (recon) probe; the frozen panel recipe also needs the signals probe (Lingua:
# benign-recon x fiction_hold) and the borderline probe (MultiJail: benign-recon x borderline-detail),
# so those five could not be scored by heldout_selector (they fell into MISSING(probe)). This fills them
# at the SAME n and frames as the original five (heldout_signals_run.sh). Cheap: 16 harmless req/frame,
# no judge. GPU-guarded, idempotent. New file (does not touch jinkwon's originals).
VP=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ENFORCE_EAGER=1 VLLM_USE_FLASHINFER_SAMPLER=0
MJ_ORDER=private_artifacts/multijail_v1/resource_order.json
MJ_BEN=private_artifacts/multijail_v1/benign_probe.jsonl
pick(){ need=$1; while :; do
  g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -v n=$need '$2>=n{gsub(/,/,"",$1); print $1; exit}')
  [ -n "$g" ] && { echo $g; return 0; }; sleep 30; done; }
# signals/borderline: $1 tag $2 ref $3 trust $4 nothink $5 util $6 need $7 al $8 outd $9 mml $10 rs
go(){ tag="$1"; ref="$2"; TR="$3"; NT="$4"; util="$5"; need="$6"; al="$7"; outd="$8"; mml="${9:-4096}"; rs="${10}"
  [ -f $outd/$tag.json ] && { echo "OK $outd/$tag"; return; }
  for a in 1 2 3; do g=$(pick $need); echo "[$(date +%H:%M:%S)] $outd $tag ($al) GPU$g try $a"
    CUDA_VISIBLE_DEVICES=$g $VP/python scripts/benign_signals_probe.py --target "$ref" --tag $tag \
      $TR $NT --tokenizer-mode auto --answer-lang $al --util $util --max-model-len $mml --outdir $outd $rs && break
    sleep 20; done; }
model(){ tag="$1"; ref="$2"; TR="$3"; NT="$4"; util="$5"; need="$6"; mml="${7:-4096}"
  go $tag "$ref" "$TR" "$NT" $util $need Swahili   results/benign_signals_mj_20260907    $mml ""
  go $tag "$ref" "$TR" "$NT" $util $need Norwegian results/benign_signals_20260907       $mml ""
  go $tag "$ref" "$TR" "$NT" $util $need Swahili   results/benign_borderline_mj_20260907 $mml "--request-set borderline"
  go $tag "$ref" "$TR" "$NT" $util $need Norwegian results/benign_borderline_20260907    $mml "--request-set borderline"
}
# yi34b lost its MultiJail recon probe (benign_arms_mj); regenerate it (Swahili answer, same as MJ run).
recon_mj(){ tag="$1"; ref="$2"; TR="$3"; NT="$4"; util="$5"; need="$6"; mml="${7:-4096}"
  outd=results/benign_arms_mj_20260907
  [ -f $outd/$tag.json ] && { echo "OK $outd/$tag"; return; }
  for a in 1 2 3; do g=$(pick $need); echo "[$(date +%H:%M:%S)] RECON_MJ $tag GPU$g try $a"
    CUDA_VISIBLE_DEVICES=$g $VP/python scripts/benign_arms_probe.py --target "$ref" --tag $tag \
      $TR $NT --tokenizer-mode auto --order $MJ_ORDER --benign $MJ_BEN --answer-lang Swahili \
      --n-items 64 --util $util --max-model-len $mml --outdir $outd && break
    sleep 20; done; }

model olmo2_7b   allenai/OLMo-2-1124-7B-Instruct       "--trust-remote-code" ""              0.45 40000
model zephyr7b   HuggingFaceH4/zephyr-7b-beta          "--trust-remote-code" ""              0.45 40000
model qwen25_15b Qwen/Qwen2.5-1.5B-Instruct            "--trust-remote-code" ""              0.30 24000
model qwen3_8b   Qwen/Qwen3-8B                         "--trust-remote-code" "--no-thinking" 0.45 40000
echo HELDOUT_SIGNALS_EXPAND_DONE $(date +%H:%M:%S)
