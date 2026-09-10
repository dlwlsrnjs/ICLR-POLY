#!/usr/bin/env python3
"""Seed robustness: re-run the resource-order sequential-add for representative models at
2 extra seeds (arrangement + item resample). Combined with the main run (seed 0) this gives
mean +/- sd on the per-step joint and on the greedy-selected config -- addressing the 40-item
noise and showing the adaptation is stable. Waits for the disorder sweep to finish. Resumable."""
import glob, subprocess, os, time
from pathlib import Path
ROOT = Path("/home/ubuntu/342/jinkwon/poly/PolyJigsaw")
OUT = ROOT / "results/seed_robustness_20260906"; OUT.mkdir(parents=True, exist_ok=True)
ORDER = "results/lang_rank_20260905/resource_order.json"
DEP = ROOT / "results/disorder_sweep_20260906/DISORDER_DONE.marker"
HUB = "/home/ubuntu/342/jinkwon/hf_cache/hub"
# representative: strong-decode/strong-align (qwen14b), strong-decode (qwen7b, llama8b),
# weak-decode (phi35, gemma2b), plus a Gemma mid (gemma9b). (tag, ref, repo, extra, util, be)
MODELS = [
    ("qwen25_7b", "Qwen/Qwen2.5-7B-Instruct", "models--Qwen--Qwen2.5-7B-Instruct", "", "0.35", None),
    ("phi35", "microsoft/Phi-3.5-mini-instruct", "models--microsoft--Phi-3.5-mini-instruct", "", "0.35", None),
    ("llama31_8b_it", "meta-llama/Llama-3.1-8B-Instruct", "models--meta-llama--Llama-3.1-8B-Instruct", "", "0.35", None),
    ("gemma2_9b_it", "google/gemma-2-9b-it", "models--google--gemma-2-9b-it", "", "0.40", "TRITON_ATTN"),
]
SEEDS = [1, 2]
def weights(repo):
    return len(glob.glob(f"{HUB}/{repo}/snapshots/*/*.safetensors")) > 0
def free_gpu():
    o = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout
    best = (-1, -1)
    for l in o.strip().splitlines():
        i, f = [int(x.strip()) for x in l.split(",")]
        if f > best[1]:
            best = (i, f)
    return best
def run(tag, ref, extra, util, be, seed):
    gpu = -1
    for _ in range(120):
        i, f = free_gpu()
        if f >= 58000:
            gpu = i; break
        time.sleep(30)
    if gpu < 0:
        return False
    env = dict(os.environ, HF_HOME="/home/ubuntu/342/jinkwon/hf_cache", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES=str(gpu),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    if be:
        env["VLLM_ATTENTION_BACKEND"] = be
    else:
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    cmd = ["/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python", "scripts/sequential_add.py",
           "--target", ref, "--tag", f"{tag}_s{seed}", "--order", ORDER, "--outdir", str(OUT),
           "--seed", str(seed), "--n-items", "40", "--util", util] + (extra.split() if extra else [])
    print(f"[{time.strftime('%H:%M:%S')}] {tag} seed={seed} GPU{gpu}", flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, stdout=open(OUT / f"panel_{tag}_s{seed}.log", "w"), stderr=subprocess.STDOUT)
    return (OUT / f"{tag}_s{seed}.json").exists()
def main():
    for _ in range(600):
        if DEP.exists():
            break
        time.sleep(60)
    print(f"[{time.strftime('%H:%M:%S')}] disorder done; starting seed robustness", flush=True)
    for seed in SEEDS:
        for tag, ref, repo, extra, util, be in MODELS:
            if (OUT / f"{tag}_s{seed}.json").exists():
                print(f"skip {tag}_s{seed}", flush=True); continue
            if not weights(repo):
                print(f"no weights {tag}", flush=True); continue
            ok = False
            for attempt in range(3):
                if run(tag, ref, extra, util, be, seed):
                    ok = True; break
                time.sleep(30)
            print(f"  -> {tag}_s{seed} {'OK' if ok else 'FAILED'}", flush=True)
    (OUT / "SEED_DONE.marker").write_text("done\n")
if __name__ == "__main__":
    main()
