#!/usr/bin/env bash
# Panel run on the rebuilt, shared inputs (private_artifacts/panel_v2).
# args: TAG MODEL GPU [EXTRA_FLAGS]
set -uo pipefail
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
VPY=/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python
TAG="$1"; MODEL="$2"; GPU="$3"; EXTRA="${4:-}"
IN=private_artifacts/panel_v2
W=$IN/$TAG; mkdir -p "$W"
log(){ echo "[$(date +%H:%M:%S)] PANELv2-$TAG: $*"; }

if [ ! -f "$W/benign_recon.json" ]; then
  log "benign probe grid"
  CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/run_text_gated_vllm.py --data $IN/benign_probe.jsonl \
    --outdir "$W/benign_gen" --target "$MODEL" --interleave-ns 2 4 6 8 10 \
    --interleave-conditions ordered shuffled --translated-langs Finnish --no-slots $EXTRA \
    --gpu-memory-utilization ${GPU_MEM_UTIL:-0.90} > "$W/benign_gen.log" 2>&1 || { log "BENIGN GEN FAILED"; tail -5 "$W/benign_gen.log"; exit 1; }
  CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/judge_reconstruction_equivalence.py \
    --input "$W/benign_gen/restricted_target_outputs.jsonl" --outdir "$W/benign_recon" \
    --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --batch-size 64 > "$W/benign_recon.log" 2>&1 \
    || { log "BENIGN JUDGE FAILED"; tail -5 "$W/benign_recon.log"; exit 1; }
  $VPY -c "
import json
from collections import defaultdict
R=[json.loads(l) for l in open('$W/benign_recon/restricted_reconstruction_audit.jsonl')]
a=defaultdict(lambda:[0,0])
for r in R:
    c=r['condition']
    if not c.startswith('interleave_'): continue
    a[c][0]+=1; a[c][1]+= 1 if r['semantic_reconstruction_equivalent'] else 0
json.dump({c:v[1]/v[0] for c,v in a.items()},open('$W/benign_recon.json','w'),indent=2)
"
fi
log "benign done"

if [ ! -f "$W/harmful_summary.json" ]; then
  log "harmful grid"
  CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/run_text_gated_vllm.py --data $IN/harm_grid.jsonl \
    --outdir "$W/harm_gen" --target "$MODEL" --interleave-ns 2 4 6 8 10 \
    --interleave-conditions ordered shuffled --translated-langs Finnish --no-slots $EXTRA \
    --gpu-memory-utilization ${GPU_MEM_UTIL:-0.90} > "$W/harm_gen.log" 2>&1 || { log "HARM GEN FAILED"; tail -5 "$W/harm_gen.log"; exit 1; }
  CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/judge_reconstruction_equivalence.py \
    --input "$W/harm_gen/restricted_target_outputs.jsonl" --outdir "$W/harm_recon" \
    --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --batch-size 64 > "$W/harm_recon.log" 2>&1 \
    || { log "HARM RECON JUDGE FAILED"; exit 1; }
  CUDA_VISIBLE_DEVICES=$GPU $VPY scripts/rejudge_qwen3guard_official.py \
    --input "$W/harm_gen/restricted_target_outputs.jsonl" --outdir "$W/harm_guard" \
    --model Qwen/Qwen3Guard-Gen-8B --device cuda:0 --batch-size 64 \
    --assistant-field answer_section > "$W/harm_guard.log" 2>&1 || { log "GUARD JUDGE FAILED"; exit 1; }
  $VPY -c "
import json
from collections import defaultdict
def load(f): return [json.loads(l) for l in open(f)]
R={(r['item_id'],r['condition']):bool(r['semantic_reconstruction_equivalent']) for r in load('$W/harm_recon/restricted_reconstruction_audit.jsonl')}
G={(r['item_id'],r['condition']):bool(r['guard_strict_unsafe']) for r in load('$W/harm_guard/restricted_qwen3guard_audit.jsonl')}
a=defaultdict(lambda:[0,0,0])
for k in G:
    c=k[1]; a[c][0]+=1; rec=R.get(k,False)
    if rec: a[c][1]+=1
    if rec and G[k]: a[c][2]+=1
json.dump({c:{'n':v[0],'recon':v[1]/v[0],'gated':v[2]/v[0]} for c,v in a.items()},open('$W/harmful_summary.json','w'),indent=2)
"
fi
log "PANELv2-$TAG COMPLETE"
