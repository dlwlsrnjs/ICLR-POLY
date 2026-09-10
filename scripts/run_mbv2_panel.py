#!/usr/bin/env python3
"""Combination attack across the 7 amount+disorder models (strong+weak). Runs now (no dep)."""
import glob, subprocess, os, time
from pathlib import Path
ROOT = Path("/home/ubuntu/342/jinkwon/poly/PolyJigsaw")
OUT = ROOT / "results/method_baselines_v2_20260906"; OUT.mkdir(parents=True, exist_ok=True)
HUB = "/home/ubuntu/342/jinkwon/hf_cache/hub"
PANEL = [
    ("qwen25_7b", "Qwen/Qwen2.5-7B-Instruct", "models--Qwen--Qwen2.5-7B-Instruct", "", "0.35", None),
    ("qwen25_14b", "Qwen/Qwen2.5-14B-Instruct", "models--Qwen--Qwen2.5-14B-Instruct", "", "0.45", None),
    ("llama31_8b_it", "meta-llama/Llama-3.1-8B-Instruct", "models--meta-llama--Llama-3.1-8B-Instruct", "", "0.35", None),
    ("gemma2_9b_it", "google/gemma-2-9b-it", "models--google--gemma-2-9b-it", "", "0.40", "TRITON_ATTN"),
    ("qwen25_3b", "Qwen/Qwen2.5-3B-Instruct", "models--Qwen--Qwen2.5-3B-Instruct", "", "0.35", None),
    ("llama32_3b_it", "meta-llama/Llama-3.2-3B-Instruct", "models--meta-llama--Llama-3.2-3B-Instruct", "", "0.35", None),
    ("gemma2_2b_it", "google/gemma-2-2b-it", "models--google--gemma-2-2b-it", "", "0.35", "TRITON_ATTN"),
]
def weights(repo): return len(glob.glob(f"{HUB}/{repo}/snapshots/*/*.safetensors")) > 0
def free_gpu():
    o = subprocess.run(["nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout
    best=(-1,-1)
    for l in o.strip().splitlines():
        i,f=[int(x.strip()) for x in l.split(",")]
        if f>best[1]: best=(i,f)
    return best
def main():
    for tag, ref, repo, extra, util, be in PANEL:
        if (OUT/f"{tag}.json").exists(): print(f"skip {tag}",flush=True); continue
        if not weights(repo): print(f"no weights {tag}",flush=True); continue
        ok=False
        for attempt in range(3):
            gpu=-1
            for _ in range(120):
                i,f=free_gpu()
                if f>=58000: gpu=i; break
                time.sleep(30)
            if gpu<0: break
            env=dict(os.environ,HF_HOME="/home/ubuntu/342/jinkwon/hf_cache",HF_HUB_OFFLINE="1",
                     TRANSFORMERS_OFFLINE="1",CUDA_VISIBLE_DEVICES=str(gpu),
                     PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            if be: env["VLLM_ATTENTION_BACKEND"]=be
            else: env["VLLM_USE_FLASHINFER_SAMPLER"]="0"
            cmd=["/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python","scripts/method_baselines_eval.py",
                 "--target",ref,"--tag",tag,"--outdir","results/method_baselines_v2_20260906","--n-items","40","--util",util]+(extra.split() if extra else [])
            print(f"[{time.strftime('%H:%M:%S')}] {tag} GPU{gpu} be={be or 'default'}",flush=True)
            subprocess.run(cmd,cwd=ROOT,env=env,stdout=open(OUT/f"panel_{tag}.log","w"),stderr=subprocess.STDOUT)
            if (OUT/f"{tag}.json").exists(): ok=True; break
            time.sleep(30)
        print(f"  -> {tag} {'OK' if ok else 'FAILED'}",flush=True)
    (OUT/"MBV2_DONE.marker").write_text("done\n")
if __name__=="__main__": main()
