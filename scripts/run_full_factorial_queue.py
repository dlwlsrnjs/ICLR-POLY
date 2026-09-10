#!/usr/bin/env python3
"""Full experiment: run the 32-arm fragment x language x arrangement factorial on ALL
panel_v2 targets, so every target has the full structured config space (not just the
10-arm interleave subset). Reuses run_fragment_factorial.sh. Per-model loader flags,
memory fraction, and env are derived from the architecture. Continue-on-failure; runs
across whichever GPUs are idle now (this is a shared machine — never contend with a busy
GPU held by another user).
"""
from __future__ import annotations
import json, subprocess, threading, queue, time, re, os
from pathlib import Path

ROOT = Path("/home/ubuntu/342/jinkwon/poly/PolyJigsaw")
DL = ROOT / "private_artifacts/selector_gpu_20260903/downloads.json"
FRAG = ROOT / "private_artifacts/frag_factorial_20260903"
STATUS = FRAG / "full_queue_status.json"

# original HF-cached targets (tag -> model id); flags/util added below
ORIGINAL = {
    "qwen25_1_5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen25_32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "zephyr7b": "HuggingFaceH4/zephyr-7b-beta",
}
# already have the full factorial
DONE_FRAG = {"falcon3_7b", "phi35", "qwen25_14b", "qwen25_3b", "qwen25_7b"}


def flags_and_env(tag, meta):
    """meta: dict with model_type, max_position_embeddings if from downloads.json, else {}."""
    extra, env = [], {}
    mt = meta.get("model_type", "")
    if mt == "qwen3" or "qwen3" in tag:
        extra += ["--no-thinking"]
    if meta.get("max_position_embeddings", 8192) <= 4096:
        extra += ["--max-model-len", "4096"]
    if "internlm" in tag:
        extra += ["--trust-remote-code", "--tokenizer-mode", "slow"]
    # low, contention-tolerant memory fractions (shared machine; transient OOM avoidance)
    if "32b" in tag:
        util = 0.80
    elif "14b" in tag:
        util = 0.55
    elif any(k in tag for k in ("7b", "8b", "9b", "mistral", "zephyr", "yi_1_5_6b")):
        util = 0.45
    else:
        util = 0.30
    if "deepseek" in tag:
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        util = 0.45
    return " ".join(extra), env, util


def build_jobs():
    dl = {re.sub(r"[^a-z0-9]+", "_", t["model_id"].split("/")[-1].lower()).strip("_"): t
          for t in json.loads(DL.read_text())["targets"] if t.get("download_status") == "completed"}
    panel = sorted(p.name for p in (ROOT / "private_artifacts/panel_v2").iterdir()
                   if p.is_dir() and (p / "harmful_summary.json").exists())
    jobs = []
    for tag in panel:
        if tag in DONE_FRAG:
            continue
        if tag in dl:
            meta = dl[tag]; ref = meta["path"]
        elif tag in ORIGINAL:
            ref = ORIGINAL[tag]; meta = {"model_id": ref}
        else:
            continue  # unknown mapping (e.g. olmo2_7b never completed panel); skip
        extra, env, util = flags_and_env(tag, meta)
        jobs.append(dict(tag=tag, model=ref, extra=extra, env=env, util=util))
    return jobs


def idle_gpus(threshold=8000):
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return [int(l.split(",")[0]) for l in out.strip().splitlines() if int(l.split(",")[1]) > threshold]


def worker(gpu, q, results, lock):
    while True:
        try:
            job = q.get_nowait()
        except queue.Empty:
            return
        tag = job["tag"]
        done = FRAG / tag / "harmful_summary.json"
        if done.exists():
            with lock:
                results[tag] = {"status": "already_done", "gpu": gpu, **{k: job[k] for k in ("model", "extra", "util")}}
            q.task_done(); continue
        env = dict(os.environ, HF_HOME="/home/ubuntu/342/jinkwon/hf_cache", HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", GPU_MEM_UTIL=str(job["util"]), **job["env"])
        log = FRAG / f"full_{tag}.log"
        cmd = ["bash", "scripts/run_fragment_factorial.sh", tag, job["model"], str(gpu), job["extra"]]
        start = time.time()
        # retry transient "free memory less than desired" races on a shared GPU
        for attempt in range(4):
            with open(log, "w") as lf:
                rc = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
            if done.exists():
                break
            txt = log.read_text(errors="ignore")
            if "less than desired GPU memory utilization" in txt or "Free memory on device" in txt:
                time.sleep(90)  # wait for the other worker / users to free memory, then retry
                continue
            break
        ok = done.exists()
        with lock:
            results[tag] = {"status": "done" if ok else f"failed_rc{rc}", "gpu": gpu,
                            "seconds": round(time.time() - start, 1),
                            **{k: job[k] for k in ("model", "extra", "util")}}
            STATUS.write_text(json.dumps(results, indent=2) + "\n")
        q.task_done()


def main():
    import sys
    jobs = build_jobs()
    gpus = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else idle_gpus()
    if not gpus:
        raise SystemExit("no idle GPUs; refusing to contend on this shared machine")
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    results, lock = {}, threading.Lock()
    STATUS.write_text(json.dumps({"planned": [j["tag"] for j in jobs], "using_gpus": gpus}, indent=2) + "\n")
    print(json.dumps({"using_gpus": gpus, "planned": len(jobs), "tags": [j["tag"] for j in jobs]}, indent=2))
    threads = [threading.Thread(target=worker, args=(g, q, results, lock)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    STATUS.write_text(json.dumps(results, indent=2) + "\n")
    done = sum(1 for v in results.values() if v["status"] in ("done", "already_done"))
    print(json.dumps({"total": len(jobs), "succeeded": done,
                      "failed": [k for k, v in results.items() if v["status"].startswith("failed")]}, indent=2))


if __name__ == "__main__":
    main()
