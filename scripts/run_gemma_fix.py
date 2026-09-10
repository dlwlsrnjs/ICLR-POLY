#!/usr/bin/env python3
"""Re-run the 2 Gemma-2 models that failed on flash-attn softcapping, using an attention
backend that supports tanh logit soft-capping (Gemma-2 requirement). Tries TRITON_ATTN,
then FLEX_ATTENTION, then FLASHINFER if importable. Waits for the main grid to finish."""
import glob, subprocess, os, time
from pathlib import Path
ROOT = Path("/home/ubuntu/342/jinkwon/poly/PolyJigsaw")
OUT = ROOT / "results/sequential_resource_20260906"
ORDER = "results/lang_rank_20260905/resource_order.json"
GEMMAS = [("gemma2_2b_it","google/gemma-2-2b-it","0.35"),
          ("gemma2_9b_it","google/gemma-2-9b-it","0.40")]
def flashinfer_ok():
    try:
        import flashinfer  # noqa
        return True
    except Exception:
        return False
def backends():
    b=["TRITON_ATTN","FLEX_ATTENTION"]
    if flashinfer_ok(): b.append("FLASHINFER")
    return b
def free_gpu():
    o=subprocess.run(["nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"],
                     capture_output=True,text=True).stdout
    best=(-1,-1)
    for l in o.strip().splitlines():
        i,f=[int(x.strip()) for x in l.split(",")]
        if f>best[1]: best=(i,f)
    return best
def main():
    for _ in range(300):
        if (OUT/"GRID_DONE.marker").exists(): break
        time.sleep(60)
    print(f"[{time.strftime('%H:%M:%S')}] grid done; fixing gemma",flush=True)
    for tag, ref, util in GEMMAS:
        if (OUT/f"{tag}.json").exists():
            print(f"skip {tag}",flush=True); continue
        for be in backends():
            gpu=-1
            for _ in range(120):
                i,f=free_gpu()
                if f>=58000: gpu=i; break
                time.sleep(30)
            if gpu<0: print(f"no GPU {tag}",flush=True); break
            env=dict(os.environ, HF_HOME="/home/ubuntu/342/jinkwon/hf_cache", HF_HUB_OFFLINE="1",
                     TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES=str(gpu),
                     PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                     VLLM_ATTENTION_BACKEND=be)
            cmd=["/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python","scripts/sequential_add.py",
                 "--target",ref,"--tag",tag,"--order",ORDER,"--outdir",str(OUT),
                 "--n-items","40","--util",util]
            print(f"[{time.strftime('%H:%M:%S')}] {tag} backend={be} GPU{gpu}",flush=True)
            subprocess.run(cmd,cwd=ROOT,env=env,stdout=open(OUT/f"panel_{tag}.log","w"),stderr=subprocess.STDOUT)
            if (OUT/f"{tag}.json").exists():
                print(f"  -> OK with {be}",flush=True); break
            print(f"  -> FAILED with {be}, trying next backend",flush=True)
    (OUT/"GEMMAFIX_DONE.marker").write_text("done\n")
if __name__=="__main__": main()
