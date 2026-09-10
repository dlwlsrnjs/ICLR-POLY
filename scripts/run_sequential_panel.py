#!/usr/bin/env python3
"""Run stage-2 sequential-add across the valid panel using the fixed universal order."""
import json, re, subprocess, os, time
from pathlib import Path
ROOT = Path("/home/ubuntu/342/jinkwon/poly/PolyJigsaw")
OUT = ROOT / "results/sequential_add_20260905"; OUT.mkdir(parents=True, exist_ok=True)
DL = {re.sub(r'[^a-z0-9]+','_',t['model_id'].split('/')[-1].lower()).strip('_'): t
      for t in json.loads((ROOT/"private_artifacts/selector_gpu_20260903/downloads.json").read_text())["targets"]}
PANEL = [
    ("qwen25_7b", "Qwen/Qwen2.5-7B-Instruct", ""),
    ("phi35", "microsoft/Phi-3.5-mini-instruct", ""),
    ("mistral7b", "mistralai/Mistral-7B-Instruct-v0.3", ""),
    ("granite_3_3_8b_instruct", DL["granite_3_3_8b_instruct"]["path"], ""),
    ("yi_1_5_6b_chat", DL["yi_1_5_6b_chat"]["path"], "--max-model-len 4096"),
    ("qwen3_4b", DL["qwen3_4b"]["path"], "--no-thinking"),
]
NEED = 58000
def free_gpu():
    o = subprocess.run(["nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"],
                       capture_output=True,text=True).stdout
    best=(-1,-1)
    for l in o.strip().splitlines():
        i,f=[int(x.strip()) for x in l.split(",")]
        if f>best[1]: best=(i,f)
    return best
def main():
    for tag, ref, extra in PANEL:
        if (OUT/f"{tag}.json").exists():
            print(f"skip {tag}",flush=True); continue
        gpu=-1
        for _ in range(180):
            i,f=free_gpu()
            if f>=NEED: gpu=i; break
            time.sleep(60)
        if gpu<0: print(f"no GPU {tag}",flush=True); continue
        env=dict(os.environ, HF_HOME="/home/ubuntu/342/jinkwon/hf_cache", HF_HUB_OFFLINE="1",
                 TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES=str(gpu),
                 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", VLLM_USE_FLASHINFER_SAMPLER="0")
        cmd=["/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python","scripts/sequential_add.py",
             "--target",ref,"--tag",tag,"--n-items","40","--util","0.35"]+extra.split()
        print(f"[{time.strftime('%H:%M:%S')}] {tag} on GPU{gpu} (free~{f})",flush=True)
        subprocess.run(cmd,cwd=ROOT,env=env,stdout=open(OUT/f"panel_{tag}.log","w"),stderr=subprocess.STDOUT)
        print(f"  -> {'OK' if (OUT/f'{tag}.json').exists() else 'FAILED'}",flush=True)
    (OUT/"PANEL_DONE.marker").write_text("done\n")
if __name__=="__main__": main()
