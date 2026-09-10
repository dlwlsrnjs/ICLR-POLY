#!/usr/bin/env python3
"""Run the benign per-language+complexity decode sweep across a diverse target panel (all
have harmful fragment data), on an idle GPU. Powers the paper claim: the optimal
language/complexity is target-dependent and readable from benign decode."""
import json, re, subprocess, os, time
from pathlib import Path
ROOT = Path("/home/ubuntu/342/jinkwon/poly/PolyJigsaw")
OUT = ROOT / "results/benign_recon_sweep_20260904"
DL = {re.sub(r'[^a-z0-9]+','_',t['model_id'].split('/')[-1].lower()).strip('_'): t
      for t in json.loads((ROOT/"private_artifacts/selector_gpu_20260903/downloads.json").read_text())["targets"]}
# (tag, model ref, extra flags) — diverse alignment/capability, all have harmful data
PANEL = [
    ("phi35", "microsoft/Phi-3.5-mini-instruct", ""),
    ("qwen25_14b", "Qwen/Qwen2.5-14B-Instruct", ""),
    ("mistral7b", "mistralai/Mistral-7B-Instruct-v0.3", ""),
    ("zephyr7b", "HuggingFaceH4/zephyr-7b-beta", ""),
    ("falcon3_7b", "tiiuae/Falcon3-7B-Instruct", ""),
    ("granite_3_3_8b_instruct", DL["granite_3_3_8b_instruct"]["path"], ""),
    ("yi_1_5_6b_chat", DL["yi_1_5_6b_chat"]["path"], "--max-model-len 4096"),
    ("qwen3_4b", DL["qwen3_4b"]["path"], "--no-thinking"),
]

def idle_gpu(thr=45000):
    o = subprocess.run(["nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout
    for l in o.strip().splitlines():
        i,f=[x.strip() for x in l.split(",")]
        if int(f)>thr: return int(i)
    return -1

def main():
    for tag, ref, extra in PANEL:
        if (OUT/f"{tag}.json").exists():
            print(f"skip {tag} (done)"); continue
        gpu=-1
        for _ in range(120):
            gpu=idle_gpu()
            if gpu>=0: break
            time.sleep(60)
        if gpu<0:
            print(f"no idle GPU for {tag}"); continue
        env=dict(os.environ, HF_HOME="/home/ubuntu/342/jinkwon/hf_cache", HF_HUB_OFFLINE="1",
                 TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES=str(gpu),
                 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        cmd=["/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python","scripts/benign_recon_sweep.py",
             "--target",ref,"--tag",tag,"--n-items","40","--util","0.40"]+extra.split()
        print(f"[{time.strftime('%H:%M:%S')}] {tag} on GPU{gpu}", flush=True)
        subprocess.run(cmd,cwd=ROOT,env=env,stdout=open(OUT/f"panel_{tag}.log","w"),stderr=subprocess.STDOUT)
        print(f"  -> {'OK' if (OUT/f'{tag}.json').exists() else 'FAILED'}", flush=True)
    (OUT/"PANEL_DONE.marker").write_text("done\n")

if __name__=="__main__":
    main()
