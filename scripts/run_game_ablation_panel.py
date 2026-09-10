#!/usr/bin/env python3
"""Game-framing ablation across the family x size grid at n=4 resource order. Waits for the
seed-robustness stage to finish (shared GPU1). Gemma uses a softcapping backend. Resumable."""
import glob, subprocess, os, time
from pathlib import Path
ROOT = Path("/home/ubuntu/342/jinkwon/poly/PolyJigsaw")
OUT = ROOT / "results/game_ablation_20260906"; OUT.mkdir(parents=True, exist_ok=True)
DEP = ROOT / "results/seed_robustness_20260906/SEED_DONE.marker"
HUB = "/home/ubuntu/342/jinkwon/hf_cache/hub"
PANEL = [
    ("qwen25_3b", "Qwen/Qwen2.5-3B-Instruct", "models--Qwen--Qwen2.5-3B-Instruct", "", "0.35", None),
    ("qwen25_7b", "Qwen/Qwen2.5-7B-Instruct", "models--Qwen--Qwen2.5-7B-Instruct", "", "0.35", None),
    ("qwen25_14b", "Qwen/Qwen2.5-14B-Instruct", "models--Qwen--Qwen2.5-14B-Instruct", "", "0.45", None),
    ("llama32_3b_it", "meta-llama/Llama-3.2-3B-Instruct", "models--meta-llama--Llama-3.2-3B-Instruct", "", "0.35", None),
    ("llama31_8b_it", "meta-llama/Llama-3.1-8B-Instruct", "models--meta-llama--Llama-3.1-8B-Instruct", "", "0.35", None),
    ("gemma2_2b_it", "google/gemma-2-2b-it", "models--google--gemma-2-2b-it", "", "0.35", "TRITON_ATTN"),
    ("gemma2_9b_it", "google/gemma-2-9b-it", "models--google--gemma-2-9b-it", "", "0.40", "TRITON_ATTN"),
    ("phi35", "microsoft/Phi-3.5-mini-instruct", "models--microsoft--Phi-3.5-mini-instruct", "", "0.35", None),
    ("mistral7b", "mistralai/Mistral-7B-Instruct-v0.3", "models--mistralai--Mistral-7B-Instruct-v0.3", "", "0.35", None),
]
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
def main():
    for _ in range(600):
        if DEP.exists():
            break
        time.sleep(60)
    print(f"[{time.strftime('%H:%M:%S')}] seed done; starting game ablation", flush=True)
    for tag, ref, repo, extra, util, be in PANEL:
        if (OUT / f"{tag}.json").exists():
            print(f"skip {tag}", flush=True); continue
        if not weights(repo):
            print(f"no weights {tag}", flush=True); continue
        ok = False
        for attempt in range(3):
            gpu = -1
            for _ in range(120):
                i, f = free_gpu()
                if f >= 58000:
                    gpu = i; break
                time.sleep(30)
            if gpu < 0:
                break
            env = dict(os.environ, HF_HOME="/home/ubuntu/342/jinkwon/hf_cache", HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES=str(gpu),
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            if be:
                env["VLLM_ATTENTION_BACKEND"] = be
            else:
                env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
            cmd = ["/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python", "scripts/game_ablation.py",
                   "--target", ref, "--tag", tag, "--n", "4", "--n-items", "40", "--util", util] + (extra.split() if extra else [])
            print(f"[{time.strftime('%H:%M:%S')}] {tag} GPU{gpu} be={be or 'default'}", flush=True)
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=open(OUT / f"panel_{tag}.log", "w"), stderr=subprocess.STDOUT)
            if (OUT / f"{tag}.json").exists():
                ok = True; break
            time.sleep(30)
        print(f"  -> {tag} {'OK' if ok else 'FAILED'}", flush=True)
    (OUT / "GAME_DONE.marker").write_text("done\n")
if __name__ == "__main__":
    main()
