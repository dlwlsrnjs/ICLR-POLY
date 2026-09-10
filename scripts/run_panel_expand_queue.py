#!/usr/bin/env python3
"""Expand the panel_v2 joint+probe collection to ~30 targets across 2 GPUs.

Reuses run_panel_v2.sh (benign reconstruction probe + harmful joint gen + recon judge +
Qwen3Guard judge) unchanged, so every new target lands on the SAME 12-condition grid and
250 harmful items as the existing 10, and carries the cheap probe anchors
(english_direct raw ASR = alignment, translated_Finnish raw ASR = multilingual gap) plus
the benign reconstruction capability curve. Per-model loader flags are derived from the
architecture. Continue-on-failure; a failed target is logged and skipped.
"""
from __future__ import annotations
import json, subprocess, threading, queue, time, re
from pathlib import Path

ROOT = Path("/home/ubuntu/342/jinkwon/poly/PolyJigsaw")
DL = ROOT / "private_artifacts/selector_gpu_20260903/downloads.json"
STATUS = ROOT / "private_artifacts/panel_v2/expand_status.json"


def tag_for(model_id):
    return re.sub(r"[^a-z0-9]+", "_", model_id.split("/")[-1].lower()).strip("_")


def flags_for(t):
    f = []
    if t.get("model_type") == "qwen3":
        f += ["--no-thinking"]
    if t.get("max_position_embeddings", 8192) <= 4096:
        f += ["--max-model-len", "4096"]
    if t.get("model_type") in ("deepseek_v2", "stablelm", "internlm2"):
        f += ["--trust-remote-code"]
    if "internlm" in t["model_id"].lower():
        f += ["--tokenizer-mode", "slow"]
    return f


def build_jobs():
    dl = json.loads(DL.read_text())
    jobs = []
    for t in dl["targets"]:
        if t.get("download_status") != "completed":
            continue
        jobs.append(dict(tag=tag_for(t["model_id"]), model=t["path"], family=t["family"],
                         extra=" ".join(flags_for(t)), model_id=t["model_id"]))
    return jobs


def worker(gpu, q, results, lock):
    while True:
        try:
            job = q.get_nowait()
        except queue.Empty:
            return
        tag, model, extra = job["tag"], job["model"], job["extra"]
        done_marker = ROOT / f"private_artifacts/panel_v2/{tag}/harmful_summary.json"
        if done_marker.exists():
            with lock:
                results[tag] = {"status": "already_done", "gpu": gpu, **job}
            q.task_done(); continue
        log = ROOT / f"private_artifacts/panel_v2/expand_{tag}.log"
        cmd = ["bash", "scripts/run_panel_v2.sh", tag, model, str(gpu), extra]
        start = time.time()
        with open(log, "w") as lf:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT).returncode
        ok = done_marker.exists()
        with lock:
            results[tag] = {"status": "done" if ok else f"failed_rc{rc}", "gpu": gpu,
                            "seconds": round(time.time() - start, 1), **job}
            STATUS.write_text(json.dumps(results, indent=2) + "\n")
        q.task_done()


def idle_gpus(threshold_mib=3000):
    """Only use GPUs that are idle now; other users may hold a GPU on this shared box."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    gpus = []
    for line in out.strip().splitlines():
        idx, used = (x.strip() for x in line.split(","))
        if int(used) < threshold_mib:
            gpus.append(int(idx))
    return gpus


def main():
    import sys
    jobs = build_jobs()
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    results, lock = {}, threading.Lock()
    gpus = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else idle_gpus()
    if not gpus:
        raise SystemExit("no idle GPUs; another user may be running. Refusing to contend.")
    STATUS.write_text(json.dumps({"planned": [j["tag"] for j in jobs], "using_gpus": gpus}, indent=2) + "\n")
    print(json.dumps({"using_gpus": gpus, "planned": len(jobs)}))
    threads = [threading.Thread(target=worker, args=(g, q, results, lock)) for g in gpus]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    STATUS.write_text(json.dumps(results, indent=2) + "\n")
    done = sum(1 for v in results.values() if v["status"] in ("done", "already_done"))
    print(json.dumps({"total": len(jobs), "succeeded": done,
                      "failed": [k for k, v in results.items() if v["status"].startswith("failed")]}, indent=2))


if __name__ == "__main__":
    main()
