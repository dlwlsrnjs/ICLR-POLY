#!/usr/bin/env python3
"""Run a bounded model collection queue on one explicitly selected GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--targets", nargs="+", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cohort", type=Path)
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(a.gpu), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HOME="/home/ubuntu/342/jinkwon/hf_cache", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="8", VLLM_WORKER_MULTIPROC_METHOD="spawn")
    manifest = json.loads((a.data/"manifest.json").read_text())
    states = []
    for target in a.targets:
        cmd = [sys.executable, str(Path(__file__).with_name("collect_selector_gpu.py")),
               "--data", str(a.data), "--outdir", str(a.outdir/f"target_{target:02d}"),
               "--target-index", str(target)]
        if a.cohort:
            cmd += ["--cohort", str(a.cohort)]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        target_env = env.copy()
        if manifest["targets"][target]["family"] == "internlm":
            compat = str((a.data.parent/"internlm_compat").resolve())
            target_env["PYTHONPATH"] = compat + os.pathsep + env.get("PYTHONPATH", "")
        started = time.time()
        log = a.outdir/f"target_{target:02d}.log"
        with log.open("a") as handle:
            result = subprocess.run(cmd, env=target_env, stdout=handle, stderr=subprocess.STDOUT)
        states.append({"target_index": target, "return_code": result.returncode,
                       "elapsed_seconds": time.time()-started, "log": str(log)})
        path = a.outdir/f"queue_gpu{a.gpu}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"gpu": a.gpu, "planned": a.targets, "finished": states}, indent=2))
        tmp.replace(path)
        print(json.dumps(states[-1]), flush=True)
    if any(s["return_code"] for s in states):
        sys.exit(1)

if __name__ == "__main__":
    main()
