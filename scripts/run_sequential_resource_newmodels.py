#!/usr/bin/env python3
"""Resource-order sequential-add for the NEW panel additions (Gemma-2-9b, internlm2_5-7b,
Qwen2.5-14B, and Llama-3.1-8B when available). Waits for the existing 6-model resource panel
to finish (shared GPU1), then runs. Resumable. Per-model util (14B needs more)."""
import json, subprocess, os, time
from pathlib import Path
ROOT = Path("/home/ubuntu/342/jinkwon/poly/PolyJigsaw")
OUT = ROOT / "results/sequential_resource_20260906"; OUT.mkdir(parents=True, exist_ok=True)
ORDER = "results/lang_rank_20260905/resource_order.json"
EXISTING_MARKER = OUT / "PANEL_DONE.marker"
LLAMA = "/home/ubuntu/342/jinkwon/hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct"
# (tag, ref, extra flags, util)
PANEL = [
    ("gemma2_9b_it", "google/gemma-2-9b-it", "", "0.40"),
    ("internlm2_5_7b_chat", "internlm/internlm2_5-7b-chat", "--trust-remote-code", "0.35"),
    ("qwen25_14b", "Qwen/Qwen2.5-14B-Instruct", "", "0.45"),
    ("llama31_8b_it", "meta-llama/Llama-3.1-8B-Instruct", "", "0.35"),   # only if downloaded
]
def gemma_ok(tag,ref):
    if tag=="llama31_8b_it":
        # only run if weights actually present
        import glob
        return len(glob.glob(LLAMA+"/snapshots/*/*.safetensors"))>0
    return True
def free_gpu():
    o = subprocess.run(["nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"],
                       capture_output=True,text=True).stdout
    best=(-1,-1)
    for l in o.strip().splitlines():
        i,f=[int(x.strip()) for x in l.split(",")]
        if f>best[1]: best=(i,f)
    return best
def main():
    # wait for the existing 6-model resource panel to complete (shared GPU)
    for _ in range(240):
        if EXISTING_MARKER.exists(): break
        time.sleep(60)
    print(f"[{time.strftime('%H:%M:%S')}] existing panel done={EXISTING_MARKER.exists()}; starting new models",flush=True)
    for tag, ref, extra, util in PANEL:
        if (OUT/f"{tag}.json").exists():
            print(f"skip {tag}",flush=True); continue
        if not gemma_ok(tag,ref):
            print(f"defer {tag} (weights not present yet)",flush=True); continue
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
             "--n-items","40","--util",util]+ (extra.split() if extra else [])
        print(f"[{time.strftime('%H:%M:%S')}] {tag} on GPU{gpu} util={util} (free~{f})",flush=True)
        subprocess.run(cmd,cwd=ROOT,env=env,stdout=open(OUT/f"panel_{tag}.log","w"),stderr=subprocess.STDOUT)
        print(f"  -> {'OK' if (OUT/f'{tag}.json').exists() else 'FAILED'}",flush=True)
    (OUT/"NEWPANEL_DONE.marker").write_text("done\n")
if __name__=="__main__": main()
