#!/usr/bin/env python3
"""Run an explicitly assigned slice of the frozen 160-arm MJ/LG queue.

This adapter keeps the collection and audit implementation in
``full_grid_l40s_small_20260917.py`` while replacing machine-local paths, GPU
indices, owner names, and bucket prefixes.  Every worker therefore writes to a
separate local root and immutable HF bucket namespace.

GPU indices are the indices printed by ``nvidia-smi`` on the worker host.  Two
GPUs use target+reconstruction followed by guard on the judge GPU.  Three GPUs
run target, reconstruction, and guard concurrently.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any

import full_grid_l40s_small_20260917 as queue


OWNER_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,31}$")


def csv_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def gpu_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--gpus must contain integer nvidia-smi indices") from exc
    if len(result) not in (2, 3) or len(set(result)) != len(result) or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("--gpus requires two or three distinct non-negative indices")
    return result


def executable(value: str, label: str) -> Path:
    resolved = shutil.which(value) if "/" not in value else value
    if not resolved:
        raise SystemExit(f"{label} executable not found: {value}")
    path = Path(resolved).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise SystemExit(f"{label} is not executable: {path}")
    return path


def replace_option(command: list[str], option: str, value: int) -> None:
    try:
        index = command.index(option)
    except ValueError as exc:
        raise RuntimeError(f"collector command omitted {option}") from exc
    command[index + 1] = str(value)


def configure(args: argparse.Namespace) -> tuple[str, tuple[int, ...]]:
    if not OWNER_RE.fullmatch(args.owner):
        raise SystemExit("--owner must match [a-z0-9][a-z0-9_-]{1,31}")

    gpus = args.gpus
    execution = (queue.EXECUTION_THREE_GPU if len(gpus) == 3
                 else queue.EXECUTION_TWO_GPU)
    run_root = (Path(args.run_root).expanduser().resolve() if args.run_root else
                queue.REPO / args.run_id / args.owner)
    hf_home = Path(args.hf_home).expanduser().resolve()
    token_path = Path(args.hf_token_path).expanduser().resolve()
    bucket_root = (args.bucket_root.rstrip("/") if args.bucket_root else
                   f"hf://buckets/jin-kwon/poly/PolyJigsaw/{args.run_id}/{args.owner}")
    if not bucket_root.startswith("hf://buckets/"):
        raise SystemExit("--bucket-root must be an hf://buckets/... path")

    queue.RUN_ID = args.run_id
    queue.OWNER = args.owner
    queue.RUN_ROOT = run_root
    queue.UNIT_ROOT = run_root / "units"
    queue.STAGING_ROOT = run_root / "staging"
    queue.RECEIPT_ROOT = run_root / "receipts"
    queue.LOG_ROOT = run_root / "logs"
    queue.STATUS_PATH = run_root / "status.jsonl"
    queue.LOCK_PATH = run_root / "queue.lock"
    queue.PYTHON = executable(args.python, "Python")
    queue.HF_BIN = executable(args.hf_bin, "HF CLI")
    queue.HF_HOME = hf_home
    queue.HF_TOKEN_PATH = token_path
    queue.BUCKET_ROOT = bucket_root
    queue.HANDOFF_COMMIT = queue.git_head()
    queue.VISIBLE_GPUS = ",".join(str(item) for item in gpus)
    queue.PHYSICAL_GPUS = {
        "target": gpus[0],
        "reconstruction_judge": gpus[1],
        "guard_judge": gpus[2] if len(gpus) == 3 else gpus[1],
    }
    queue.EXECUTION_GPUS = {execution: gpus}
    queue.EXTERNAL_ASSIGNMENTS = {}

    original_collector_command = queue.collector_command
    original_collection_environment = queue.collection_environment
    original_make_protocol = queue.make_protocol

    def wait_for_assigned_mode(minimum_free_mib: int = args.minimum_free_mib) -> str:
        del minimum_free_mib
        wait_for_assigned_gpus(execution)
        return execution

    def wait_for_assigned_gpus(requested: str,
                               minimum_free_mib: int = args.minimum_free_mib) -> None:
        del minimum_free_mib
        if requested != execution:
            raise RuntimeError(
                f"durable unit requires {requested}, but this worker is configured for {execution}")
        while True:
            free = queue.gpu_free_mib()
            blocked = {index: free.get(index, 0) for index in gpus
                       if free.get(index, 0) < args.minimum_free_mib}
            if not blocked:
                queue.status("assigned_gpus_available", execution=execution,
                             free_mib={index: free[index] for index in gpus})
                return
            queue.status("assigned_gpus_wait", execution=execution,
                         minimum_free_mib=args.minimum_free_mib, blocked=blocked)
            time.sleep(60)

    def portable_collector_command(model: queue.ModelSpec, dataset: queue.DatasetSpec,
                                   root: Path, requested: str) -> list[str]:
        if requested != execution:
            raise RuntimeError(f"worker configured for {execution}, got {requested}")
        command = original_collector_command(model, dataset, root, requested)
        replace_option(command, "--target-gpu", gpus[0])
        if requested == queue.EXECUTION_THREE_GPU:
            replace_option(command, "--recon-gpu", gpus[1])
            replace_option(command, "--guard-gpu", gpus[2])
        else:
            replace_option(command, "--judge-gpu", gpus[1])
        return command

    def portable_environment(model: queue.ModelSpec, protocol: dict[str, Any],
                             requested: str) -> dict[str, str]:
        if requested != execution:
            raise RuntimeError(f"worker configured for {execution}, got {requested}")
        env = original_collection_environment(model, protocol, requested)
        env["CUDA_VISIBLE_DEVICES"] = queue.VISIBLE_GPUS
        return env

    def portable_protocol(model: queue.ModelSpec, dataset: queue.DatasetSpec,
                          requested: str = execution) -> dict[str, Any]:
        if requested != execution:
            raise RuntimeError(f"worker configured for {execution}, got {requested}")
        protocol = original_make_protocol(model, dataset, requested)
        runtime = protocol["runtime"]
        runtime["physical_gpus"] = dict(queue.PHYSICAL_GPUS)
        runtime["cuda_visible_devices"] = queue.VISIBLE_GPUS
        if requested == queue.EXECUTION_THREE_GPU:
            runtime["logical_mapping"] = {
                "target_worker_cuda:0": gpus[0],
                "reconstruction_worker_cuda:0": gpus[1],
                "guard_worker_cuda:0": gpus[2],
            }
        else:
            runtime["logical_mapping"] = {
                "target_worker_cuda:0": gpus[0],
                "reconstruction_worker_cuda:0": gpus[1],
                "guard_worker_cuda:0_after_reconstruction_exit": gpus[1],
            }
        core = {key: value for key, value in protocol.items() if key != "protocol_hash"}
        return {**core, "protocol_hash": queue.canonical_hash(core)}

    queue.wait_for_execution_mode = wait_for_assigned_mode
    queue.wait_for_mode_gpus = wait_for_assigned_gpus
    queue.collector_command = portable_collector_command
    queue.collection_environment = portable_environment
    queue.make_protocol = portable_protocol

    if not args.cleanup_model_cache:
        def retain_cache(model: queue.ModelSpec) -> None:
            queue.status("target_cache_retained", model=model.model,
                         reason="distributed_worker_safe_default")
        queue.cleanup_target_cache = retain_cache

    return execution, gpus


def parser() -> argparse.ArgumentParser:
    default_hf = os.environ.get("POLY_HF_BIN") or shutil.which("hf") or "hf"
    default_home = os.environ.get("HF_HOME") or str(queue.REPO / "bigmodel_l40s" / "hf_home")
    default_token = os.environ.get("HF_TOKEN_PATH") or "~/.cache/huggingface/token"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("plan", "preflight", "run"), nargs="?", default="run")
    ap.add_argument("--owner", required=True,
                    help="unique worker/shard owner used in local and bucket paths")
    ap.add_argument("--models", required=True,
                    help="comma-separated model tags assigned to this worker")
    ap.add_argument("--datasets", default="mj,lg",
                    help="comma-separated subset of mj,lg")
    ap.add_argument("--gpus", required=True, type=gpu_list,
                    help="two or three distinct physical nvidia-smi indices")
    ap.add_argument("--run-id", default="full_grid_20260917_v1")
    ap.add_argument("--run-root", default="",
                    help="local root; defaults to <repo>/<run-id>/<owner>")
    ap.add_argument("--bucket-root", default="",
                    help="immutable output prefix; defaults to the owner namespace")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--hf-bin", default=default_hf)
    ap.add_argument("--hf-home", default=default_home)
    ap.add_argument("--hf-token-path", default=default_token)
    ap.add_argument("--minimum-free-mib", type=int, default=40000)
    ap.add_argument("--cleanup-model-cache", action="store_true",
                    help="remove a target cache only after both assigned datasets are receipted")
    return ap


def main() -> int:
    args = parser().parse_args()
    if args.minimum_free_mib < 1024:
        raise SystemExit("--minimum-free-mib must be at least 1024")
    selected_models = csv_set(args.models)
    selected_datasets = csv_set(args.datasets)
    known = {model.tag for model in queue.MODELS}
    aliases = {model.model: model.tag for model in queue.MODELS}
    unknown = selected_models.difference(known | set(aliases))
    if unknown:
        raise SystemExit(f"unknown --models entries: {sorted(unknown)}")
    if not selected_datasets or not selected_datasets.issubset({"mj", "lg"}):
        raise SystemExit("--datasets accepts a non-empty subset of mj,lg")

    execution, gpus = configure(args)
    chosen = [model for model in queue.MODELS
              if model.tag in selected_models or model.model in selected_models]
    if args.command == "plan":
        print(json.dumps({
            "schema": "poly_distributed_worker_plan_v1",
            "run_id": queue.RUN_ID,
            "owner": queue.OWNER,
            "run_root": str(queue.RUN_ROOT),
            "bucket_root": queue.BUCKET_ROOT,
            "execution": execution,
            "gpus": list(gpus),
            "models": [dataclasses.asdict(model) for model in chosen],
            "datasets": sorted(selected_datasets),
            "settings_per_unit": len(queue.SETTINGS),
            "max_new_tokens": 1024,
            "false_reject": False,
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "preflight":
        queue.preflight()
        return 0
    return queue.run_queue(selected_models, selected_datasets)


if __name__ == "__main__":
    raise SystemExit(main())
