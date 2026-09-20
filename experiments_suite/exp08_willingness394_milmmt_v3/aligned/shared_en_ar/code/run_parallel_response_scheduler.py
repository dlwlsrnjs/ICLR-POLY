#!/usr/bin/env python3
"""Keep all available L40S lanes focused on unfinished panel responses.

This restartable coordinator takes over after the three medium-model tmux lanes.
It overlaps Phi-3-medium on GPU 3 with Qwen-32B on GPUs 1+2, then uses GPUs
1+2 for the final two tensor-parallel models. Completed, reproducible model
caches are removed only when space is needed for the next pinned checkpoint.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time


MODELS = {
    "phi3_medium_14b": {
        "repo_id": "microsoft/Phi-3-medium-4k-instruct",
        "revision": "48d87cd5e0523b430d77e93becd3655cd6897230",
        "gpu": "3",
        "tp": 1,
        "min_free_mib": 30000,
        "trust_remote_code": True,
    },
    "qwen25_32b": {
        "repo_id": "Qwen/Qwen2.5-32B-Instruct",
        "revision": "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
        "gpu": "1,2",
        "tp": 2,
        "min_free_mib": 40000,
        "trust_remote_code": False,
    },
    "mistral24b": {
        "repo_id": "mistralai/Mistral-Small-24B-Instruct-2501",
        "revision": "9527884be6e5616bdd54de542f9ae13384489724",
        "gpu": "1,2",
        "tp": 2,
        "min_free_mib": 40000,
        "trust_remote_code": False,
    },
    "gemma2_27b": {
        "repo_id": "google/gemma-2-27b-it",
        "revision": "aaf20e6b9f4c0fcf043f6fb2a2068419086d77b0",
        "gpu": "1,2",
        "tp": 2,
        "min_free_mib": 40000,
        "trust_remote_code": False,
    },
}


PHASE_ONE_SESSIONS = {
    "falcon3_10b": "panel_falcon10_g3",
    "glm4_9b": "panel_glm9_g1",
    "qwen25_14b": "panel_qwen14_g2",
}


COMPLETED_CACHE_CANDIDATES = {
    "qwen25_3b": "models--Qwen--Qwen2.5-3B-Instruct",
    "llama32_3b_it": "models--meta-llama--Llama-3.2-3B-Instruct",
    "gemma2_2b_it": "models--google--gemma-2-2b-it",
    "falcon3_3b": "models--tiiuae--Falcon3-3B-Instruct",
    "mistral7b": "models--mistralai--Mistral-7B-Instruct-v0.3",
    "falcon3_7b": "models--tiiuae--Falcon3-7B-Instruct",
    "llama31_8b_it": "models--meta-llama--Llama-3.1-8B-Instruct",
    "gemma2_9b_it": "models--google--gemma-2-9b-it",
    "falcon3_10b": "models--tiiuae--Falcon3-10B-Instruct",
    "glm4_9b": "models--THUDM--glm-4-9b-chat-hf",
    "qwen25_14b": "models--Qwen--Qwen2.5-14B-Instruct",
    "phi3_medium_14b": "models--microsoft--Phi-3-medium-4k-instruct",
    "qwen25_32b": "models--Qwen--Qwen2.5-32B-Instruct",
    "mistral24b": "models--mistralai--Mistral-Small-24B-Instruct-2501",
    "gemma2_27b": "models--google--gemma-2-27b-it",
}


def rows(path):
    if not path.exists():
        return 0
    with path.open() as handle:
        return sum(bool(line.strip()) for line in handle)


def complete(run, tag):
    root = run / "panel" / tag
    jobs = rows(root / "jobs.jsonl")
    return jobs > 0 and rows(root / "responses.jsonl") == jobs


def session_exists(name):
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def wait_session(name):
    while session_exists(name):
        time.sleep(15)


def cache_name(repo_id):
    return "models--" + repo_id.replace("/", "--")


def safe_remove_cache(hub, cache):
    target = hub / cache
    if target.parent != hub or not target.name.startswith("models--"):
        raise ValueError(f"Unsafe cache target: {target}")
    if target.exists():
        try:
            shutil.rmtree(target)
            return True
        except FileNotFoundError:
            # A concurrent lane may have removed the same completed cache.
            return False
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--hf-cli", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--token-path", type=Path, required=True)
    args = parser.parse_args()

    run = args.run.resolve()
    repo = args.repo.resolve()
    python = Path(os.path.abspath(args.python))
    hf_cli = Path(os.path.abspath(args.hf_cli))
    hf_home = args.hf_home.resolve()
    token_path = args.token_path.resolve()
    hub = hf_home / "hub"
    if not all([run.is_dir(), repo.is_dir(), python.is_file(), hf_cli.is_file(),
                hub.is_dir(), token_path.is_file()]):
        parser.error("Invalid run, repository, executable, cache, or token")

    package = Path(__file__).resolve().parents[1]
    queue = Path(__file__).with_name("run_panel_queue.py")
    status_path = run / "pipeline_status_parallel_responses.json"
    state_lock = threading.Lock()
    state = {
        "stage": "starting",
        "updated_utc": None,
        "completed_models": [],
        "failed_models": [],
        "removed_completed_model_caches": [],
        "lanes": {},
    }

    def status(stage=None, **extra):
        with state_lock:
            if stage is not None:
                state["stage"] = stage
            state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            state.update(extra)
            temporary = status_path.with_suffix(status_path.suffix + ".tmp")
            temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
            temporary.replace(status_path)
            print(json.dumps(state, sort_keys=True), flush=True)

    download_env = dict(
        os.environ,
        HF_HOME=str(hf_home),
        HF_TOKEN_PATH=str(token_path),
        HF_HUB_DISABLE_XET="0",
    )
    exclude = ["*.bin", "original/*", "*.gguf", "*.h5", "*.msgpack", "*.ot",
               "consolidated*.safetensors"]

    def clean_completed(exclude_tags=()):
        removed_now = []
        for tag, cache in COMPLETED_CACHE_CANDIDATES.items():
            if tag in exclude_tags or not complete(run, tag):
                continue
            if safe_remove_cache(hub, cache):
                removed_now.append(tag)
        if removed_now:
            with state_lock:
                state["removed_completed_model_caches"].extend(removed_now)
            status()

    def run_model(tag):
        if complete(run, tag):
            with state_lock:
                if tag not in state["completed_models"]:
                    state["completed_models"].append(tag)
            status()
            return True
        model = MODELS[tag]
        status("downloading", active_model=tag, lane_gpus=model["gpu"])
        command = [
            str(hf_cli), "download", model["repo_id"], "--revision", model["revision"],
            "--exclude", *exclude,
        ]
        download = subprocess.run(command, env=download_env)
        snapshot = hub / cache_name(model["repo_id"]) / "snapshots" / model["revision"]
        if download.returncode or not snapshot.is_dir():
            with state_lock:
                state["failed_models"].append({
                    "model": tag, "stage": "download", "returncode": download.returncode,
                })
            status("download_failed", active_model=tag)
            return False
        command = [
            str(python), "-u", str(queue), "--run", str(run), "--repo", str(repo),
            "--python", str(python), "--gpu", model["gpu"],
            "--status-file", f"pipeline_status_followup_{tag}.json",
            "--memory-utilization", ".85", "--batch-size", "8",
            "--max-num-batched-tokens", "2048",
            "--tensor-parallel-size", str(model["tp"]),
            "--min-free-mib", str(model["min_free_mib"]), "--continue-on-error",
            "--model", f"{tag}={snapshot}",
        ]
        if model["trust_remote_code"]:
            command.extend(["--trust-remote-code-tag", tag])
        status("collecting", active_model=tag, lane_gpus=model["gpu"])
        result = subprocess.run(command, env=download_env)
        if result.returncode == 0 and complete(run, tag):
            with state_lock:
                if tag not in state["completed_models"]:
                    state["completed_models"].append(tag)
            status("model_complete", active_model=tag,
                   response_rows=rows(run / "panel" / tag / "responses.jsonl"))
            return True
        with state_lock:
            state["failed_models"].append({
                "model": tag, "stage": "collection", "returncode": result.returncode,
                "responses": rows(run / "panel" / tag / "responses.jsonl"),
            })
        status("collection_failed", active_model=tag)
        return False

    def phi_lane():
        status(lanes={**state["lanes"], "gpu3": "waiting_for_falcon3_10b"})
        wait_session(PHASE_ONE_SESSIONS["falcon3_10b"])
        if not complete(run, "falcon3_10b"):
            status("phase_one_failure", incomplete_model="falcon3_10b")
            return
        clean_completed(exclude_tags={"glm4_9b", "qwen25_14b", "qwen25_32b"})
        status(lanes={**state["lanes"], "gpu3": "phi3_medium_14b"})
        run_model("phi3_medium_14b")
        status(lanes={**state["lanes"], "gpu3": "complete"})

    def tp_lane():
        status(lanes={**state["lanes"], "gpu1_2": "waiting_for_glm4_9b_qwen25_14b"})
        wait_session(PHASE_ONE_SESSIONS["glm4_9b"])
        wait_session(PHASE_ONE_SESSIONS["qwen25_14b"])
        missing = [tag for tag in ("glm4_9b", "qwen25_14b") if not complete(run, tag)]
        if missing:
            status("phase_one_failure", incomplete_models=missing)
            return
        clean_completed(exclude_tags={"phi3_medium_14b", "qwen25_32b"})
        status(lanes={**state["lanes"], "gpu1_2": "qwen25_32b"})
        run_model("qwen25_32b")
        status(lanes={**state["lanes"], "gpu1_2": "qwen25_32b_complete"})

    status("waiting_for_phase_one", lanes={
        "gpu3": "waiting_for_falcon3_10b",
        "gpu1_2": "waiting_for_glm4_9b_qwen25_14b",
    })
    workers = [threading.Thread(target=phi_lane), threading.Thread(target=tp_lane)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    prerequisites = ["phi3_medium_14b", "qwen25_32b"]
    if not all(complete(run, tag) for tag in prerequisites):
        status("scheduler_stopped_incomplete", incomplete_models=[
            tag for tag in prerequisites if not complete(run, tag)
        ])
        return 2

    clean_completed()
    for tag in ("mistral24b", "gemma2_27b"):
        status("final_tp_queue", active_model=tag, lane_gpus="1,2")
        if not run_model(tag):
            status("scheduler_stopped_incomplete", incomplete_model=tag)
            return 2
        clean_completed()

    status("response_collection_complete", completed_panel_models=sum(
        complete(run, path.name) for path in (run / "panel").iterdir() if path.is_dir()
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
