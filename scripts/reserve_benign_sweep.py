#!/usr/bin/env python3
"""Watch BOTH GPUs; run the live benign per-language decode sweep the moment there is a
REAL free event (>=52GB free, stable) — i.e. a neighbor vLLM server actually released memory.
Launching into the tight ~33GB left while the 8B/14B servers actively serve keeps getting
OOM-killed (their KV cache grows during requests and pushes total over capacity, so the OS
kills the newest process = ours). So we only fire on genuine headroom."""
import subprocess, time, os
from pathlib import Path
ROOT = Path("/home/ubuntu/342/jinkwon/poly/PolyJigsaw")
OUT = ROOT / "results/benign_recon_sweep_20260904"
LOG = ROOT / "results/reserve_benign_sweep.log"
TARGET = "Qwen/Qwen2.5-7B-Instruct"; TAG = "qwen25_7b"
NEED_FREE = 52000     # MiB — only fires when a big server frees up

def log(m):
    with open(LOG, "a") as f:
        f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {m}\n")

def gpu(i):
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,utilization.gpu",
                        "--format=csv,noheader,nounits", "-i", str(i)], capture_output=True, text=True).stdout.strip()
    f, u = [int(x) for x in o.split(",")]
    return f, u

def stable(i, checks=6, gap=15):
    for _ in range(checks):
        f, _ = gpu(i)
        if f < NEED_FREE:
            return False
        time.sleep(gap)
    return True

def launch(i):
    env = dict(os.environ, HF_HOME="/home/ubuntu/342/jinkwon/hf_cache", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES=str(i),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ERROR.txt").unlink(missing_ok=True)
    subprocess.run(["/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python",
                    "scripts/benign_recon_sweep.py", "--target", TARGET, "--tag", TAG,
                    "--n-items", "40", "--util", "0.35"], cwd=ROOT, env=env,
                   stdout=open(OUT / f"reserve_gpu{i}.log", "w"), stderr=subprocess.STDOUT)
    return (OUT / f"{TAG}.json").exists()

def main():
    log(f"reservation armed: watching GPU0+GPU1 for a REAL free event (>={NEED_FREE}MiB free, stable x6)")
    deadline = time.time() + 36 * 3600
    while time.time() < deadline:
        if (OUT / f"{TAG}.json").exists():
            log("result present; done"); return
        for i in (0, 1):
            f, u = gpu(i)
            if f >= NEED_FREE and stable(i):
                log(f"GPU{i} freed up (free~{f}MiB). launching sweep.")
                if launch(i):
                    log(f"SUCCESS on GPU{i}"); return
                log(f"GPU{i} attempt failed; cooldown 5min then keep watching.")
                time.sleep(300); break
        else:
            time.sleep(90)
    log("reservation expired (36h) without success")

if __name__ == "__main__":
    main()
