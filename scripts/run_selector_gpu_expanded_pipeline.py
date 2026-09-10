#!/usr/bin/env python3
"""Run 12-target verification, then a 30-target / family-held-out CUDA experiment."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def atomic(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2)+"\n")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--wait-previous-pid", type=int)
    ap.add_argument("--start-stage", default="collect128_12",
                    choices=["collect128_12", "train128_12", "expand", "collect128_30", "train128_30", "collect997_30", "train997_30"])
    a = ap.parse_args()
    import psutil
    a.results.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    data, collection = a.run_root/"data", a.run_root/"collection"
    status_path = a.run_root/"expanded_pipeline_status.json"
    pipeline_lock = (a.run_root/".expanded_pipeline.lock").open("a")
    try:
        fcntl.flock(pipeline_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another expanded pipeline is already running for this run root")
    previous_status = json.loads(status_path.read_text()) if status_path.exists() else {}
    history = previous_status.get("history", []).copy()
    if previous_status:
        atomic(a.run_root/f"expanded_pipeline_status_before_resume_{time.time_ns()}.json", previous_status)

    def status(stage, state, jobs=None):
        payload = {"stage":stage, "status":state, "pid":os.getpid(), "updated_at_unix":time.time(),
                   "jobs":jobs or [], "history":history}
        atomic(status_path, payload)
        print(json.dumps({"stage":stage,"status":state}),flush=True)

    if a.wait_previous_pid:
        try:
            previous = psutil.Process(a.wait_previous_pid)
            birth = previous.create_time()
        except psutil.NoSuchProcess:
            previous = None
        status("wait_previous_collection", "waiting")
        while previous is not None:
            try:
                if not previous.is_running() or previous.create_time() != birth or previous.status() == psutil.STATUS_ZOMBIE:
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(10)

    stages = ["collect128_12", "train128_12", "expand", "collect128_30", "train128_30", "collect997_30", "train997_30"]
    for stage in stages[stages.index(a.start_stage):]:
        if stage == "expand":
            status(stage, "waiting_for_downloads")
            while True:
                path = a.run_root/"downloads.json"
                if path.exists():
                    downloads = json.loads(path.read_text())
                    if downloads["finished"] == downloads["planned"]:
                        break
                time.sleep(10)
            if downloads["completed"] != downloads["planned"]:
                status(stage, "failed_downloads")
                raise SystemExit(1)
            manifest_path = data/"manifest.json"
            manifest = json.loads(manifest_path.read_text())
            if len(manifest["targets"]) == 12:
                (data/"manifest_original12.json").write_text(manifest_path.read_text())
                for target in downloads["targets"]:
                    manifest["targets"].append({k:target[k] for k in ("model_id","family","revision","path")})
                manifest["target_checkpoints"] = len(manifest["targets"])
                manifest["planned_target_calls"] = manifest["source_items"]*manifest["configurations"]*len(manifest["targets"])
                manifest["conservative_family_groups"] = len({t["family"] for t in manifest["targets"]})
                manifest["extension_manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                atomic(manifest_path, manifest)
            if len(manifest["targets"]) != 30:
                raise ValueError("Expected 30 pinned targets after expansion")
            history.append({"stage":stage,"status":"completed"})
            status(stage, "completed")
            continue

        manifest = json.loads((data/"manifest.json").read_text())
        target_count = 12 if stage.endswith("_12") else 30
        families = sorted({t["family"] for t in manifest["targets"][:target_count]})
        preliminary = "128" in stage
        # Start with large models and greedily balance file sizes across the two GPUs.
        costs = []
        for i, target in enumerate(manifest["targets"][:target_count]):
            weights = sum(p.stat().st_size for p in Path(target["path"]).glob("*.safetensors") if p.is_file())
            costs.append((weights, i))
        partitions, totals = [[], []], [0, 0]
        for weight, i in sorted(costs, reverse=True):
            gpu = 0 if totals[0] <= totals[1] else 1
            partitions[gpu].append(i)
            totals[gpu] += max(weight, 1_000_000_000)
        jobs, children, handles = [], [], []
        for gpu in (0, 1):
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
            if stage.startswith("collect"):
                cmd = [sys.executable, str(scripts/"run_selector_gpu_queue.py"),
                       "--data", str(data), "--outdir", str(collection), "--gpu",str(gpu),
                       "--targets", *map(str,partitions[gpu])]
                if preliminary:
                    cmd += ["--cohort",str(data/"cohort_128.json")]
            else:
                output = a.results/f"{stage}_gpu{gpu}"
                if output.exists():
                    progress = output/"progress.json"
                    if progress.exists() and json.loads(progress.read_text())["status"] == "completed":
                        jobs.append({"gpu":gpu,"status":"already_completed","outdir":str(output),"return_code":0})
                        continue
                    output = a.results/f"{stage}_gpu{gpu}_retry_{int(time.time())}"
                cmd = [sys.executable,str(scripts/"train_selector_gpu.py"),
                       "--data",str(data),"--collection",str(collection),"--outdir",str(output),
                       "--require-targets",str(target_count),"--folds",*map(str,range(gpu,len(families),2)),
                       "--seeds",*(("7",) if preliminary else ("7","17","27"))]
                if preliminary:
                    cmd += ["--cohort",str(data/"cohort_128.json"),"--supervised-steps","500","--ppo-updates","300"]
            log = a.run_root/f"{stage}_gpu{gpu}.log"
            handle = log.open("a")
            child = subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT)
            handles.append(handle)
            job = {"gpu":gpu,"pid":child.pid,"command":cmd,"log":str(log)}
            jobs.append(job)
            children.append((child,job))
        status(stage, "running", jobs)
        for child, job in children:
            job["return_code"] = child.wait()
        for handle in handles:
            handle.close()
        if any(j["return_code"] for j in jobs):
            status(stage, "failed", jobs)
            raise SystemExit(1)
        history.append({"stage":stage,"status":"completed","jobs":jobs})
        status(stage, "completed", jobs)
    status("all", "completed")


if __name__ == "__main__":
    main()
