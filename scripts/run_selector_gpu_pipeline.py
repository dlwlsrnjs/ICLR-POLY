#!/usr/bin/env python3
"""Two-GPU collection and training pipeline, with durable stage status and logs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--start-stage", choices=["collect128", "train128", "collect997", "train997"], default="collect128")
    a = ap.parse_args()
    a.run_root.mkdir(parents=True, exist_ok=True)
    a.results.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    data, collection = a.run_root/"data", a.run_root/"collection"
    stage_names = ["collect128", "train128", "collect997", "train997"]
    statuses = []

    def status(stage, state, jobs):
        payload = {"stage": stage, "status": state, "jobs": jobs,
                   "pid": os.getpid(), "updated_at_unix": time.time(), "previous_stages": statuses}
        p = a.run_root/"pipeline_status.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(p)
        print(json.dumps({"stage": stage, "status": state}), flush=True)

    for stage in stage_names[stage_names.index(a.start_stage):]:
        jobs, processes, handles = [], [], []
        for gpu in (0, 1):
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
            if stage.startswith("collect"):
                targets = [0, 8, 11, 10, 4] if gpu == 0 else [6, 9, 7, 1, 2, 3, 5]
                cmd = [sys.executable, str(scripts/"run_selector_gpu_queue.py"),
                       "--data", str(data), "--outdir", str(collection), "--gpu", str(gpu),
                       "--targets", *map(str, targets)]
                if stage == "collect128":
                    cmd += ["--cohort", str(data/"cohort_128.json")]
            else:
                cmd = [sys.executable, str(scripts/"train_selector_gpu.py"),
                       "--data", str(data), "--collection", str(collection),
                       "--outdir", str(a.results/f"{stage}_gpu{gpu}"),
                       "--folds", *(str(i) for i in range(gpu, 6, 2)),
                       "--seeds", *(("7",) if stage == "train128" else ("7", "17", "27"))]
                if stage == "train128":
                    cmd += ["--cohort", str(data/"cohort_128.json"),
                            "--supervised-steps", "500", "--ppo-updates", "300"]
            log = a.run_root/f"{stage}_gpu{gpu}.log"
            handle = log.open("a")
            process = subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT)
            handles.append(handle)
            processes.append(process)
            jobs.append({"gpu": gpu, "pid": process.pid, "command": cmd, "log": str(log)})
        status(stage, "running", jobs)
        codes = [p.wait() for p in processes]
        for h in handles:
            h.close()
        for j, code in zip(jobs, codes):
            j["return_code"] = code
        if any(codes):
            status(stage, "failed", jobs)
            sys.exit(1)
        statuses.append({"stage":stage, "status":"completed", "jobs":jobs})
        status(stage, "completed", jobs)
    status("all", "completed", [])


if __name__ == "__main__":
    main()
