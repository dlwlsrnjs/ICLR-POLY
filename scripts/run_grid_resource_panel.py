#!/usr/bin/env python3
"""Family x size grid (Qwen2.5 / Llama-3 / Gemma-2 at ~3/7-9/14B) for resource-order
sequential-add. Runs each model only if its weights are present (pending downloads auto-skip
and are picked up on a later resumed run). Waits for the existing-6 resource panel first."""
import glob, subprocess, os, time
from pathlib import Path
ROOT = Path("/home/ubuntu/342/jinkwon/poly/PolyJigsaw")
OUT = ROOT / "results/sequential_resource_20260906"; OUT.mkdir(parents=True, exist_ok=True)
ORDER = "results/lang_rank_20260905/resource_order.json"
HUB = "/home/ubuntu/342/jinkwon/hf_cache/hub"
# (tag, hf_ref, cache_repo_dir, extra, util)
GRID = [
    ("qwen25_3b",          "Qwen/Qwen2.5-3B-Instruct",        "models--Qwen--Qwen2.5-3B-Instruct",        "", "0.35"),
    ("qwen25_14b",         "Qwen/Qwen2.5-14B-Instruct",       "models--Qwen--Qwen2.5-14B-Instruct",       "", "0.45"),
    ("gemma2_2b_it",       "google/gemma-2-2b-it",            "models--google--gemma-2-2b-it",            "", "0.35"),
    ("gemma2_9b_it",       "google/gemma-2-9b-it",            "models--google--gemma-2-9b-it",            "", "0.40"),
    ("llama32_3b_it",      "meta-llama/Llama-3.2-3B-Instruct","models--meta-llama--Llama-3.2-3B-Instruct","", "0.35"),
    ("llama31_8b_it",      "meta-llama/Llama-3.1-8B-Instruct","models--meta-llama--Llama-3.1-8B-Instruct","", "0.35"),
    ("internlm2_5_7b_chat","internlm/internlm2_5-7b-chat",    "models--internlm--internlm2_5-7b-chat",    "--trust-remote-code", "0.35"),
]
def weights_present(repo):
    return len(glob.glob(f"{HUB}/{repo}/snapshots/*/*.safetensors")) > 0
def free_gpu():
    o = subprocess.run(["nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"],
                       capture_output=True,text=True).stdout
    best=(-1,-1)
    for l in o.strip().splitlines():
        i,f=[int(x.strip()) for x in l.split(",")]
        if f>best[1]: best=(i,f)
    return best
def main():
    for _ in range(300):
        if (OUT/"PANEL_DONE.marker").exists(): break
        time.sleep(60)
    print(f"[{time.strftime('%H:%M:%S')}] existing-6 done; starting grid",flush=True)
    for tag, ref, repo, extra, util in GRID:
        if (OUT/f"{tag}.json").exists():
            print(f"skip {tag} (done)",flush=True); continue
        if not weights_present(repo):
            print(f"DEFER {tag} (weights not present)",flush=True); continue
        gpu=-1
        for _ in range(240):
            i,f=free_gpu()
            if f>=58000: gpu=i; break
            time.sleep(60)
        if gpu<0: print(f"no GPU {tag}",flush=True); continue
        env=dict(os.environ, HF_HOME="/home/ubuntu/342/jinkwon/hf_cache", HF_HUB_OFFLINE="1",
                 TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES=str(gpu),
                 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", VLLM_USE_FLASHINFER_SAMPLER="0")
        cmd=["/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python","scripts/sequential_add.py",
             "--target",ref,"--tag",tag,"--order",ORDER,"--outdir",str(OUT),
             "--n-items","40","--util",util]+(extra.split() if extra else [])
        print(f"[{time.strftime('%H:%M:%S')}] {tag} on GPU{gpu} util={util} (free~{f})",flush=True)
        subprocess.run(cmd,cwd=ROOT,env=env,stdout=open(OUT/f"panel_{tag}.log","w"),stderr=subprocess.STDOUT)
        print(f"  -> {'OK' if (OUT/f'{tag}.json').exists() else 'FAILED'}",flush=True)
    (OUT/"GRID_DONE.marker").write_text("done\n")
if __name__=="__main__": main()
