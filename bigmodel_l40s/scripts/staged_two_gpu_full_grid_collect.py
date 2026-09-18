#!/usr/bin/env python3
"""Two-GPU staged collector for the exact 160-arm full grid.

The target worker runs on one physical GPU while the reconstruction worker
consumes its durable generated JSONL files on the second GPU.  Once target and
reconstruction finish, the reconstruction process exits and the same judge GPU
is reused for Qwen3Guard.  All durable stage files and final row schemas are the
same as ``pipelined_full_grid_collect.py``; only the execution schedule differs.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any

import pipelined_full_grid_collect as base


def run_pair(processes: list[mp.Process]) -> None:
    for process in processes:
        process.start()
    while any(process.is_alive() for process in processes):
        failed = [(process.name, process.exitcode) for process in processes
                  if process.exitcode not in (None, 0)]
        if failed:
            base.terminate(processes)
            raise RuntimeError(f"staged worker failure: {failed}")
        time.sleep(0.5)
    for process in processes:
        process.join(timeout=30)
    failed = [(process.name, process.exitcode) for process in processes
              if process.exitcode != 0]
    if failed:
        raise RuntimeError(f"staged worker failure after join: {failed}")


def write_manifest(args: argparse.Namespace,
                   aggregates: list[dict[str, Any]]) -> None:
    base.write_manifest(args, aggregates)
    path = Path(args.root) / "MANIFEST.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["pipeline"] = "two_gpu_staged_pipeline_v1"
    base.atomic_json(path, payload)


def run_pipeline(args: argparse.Namespace) -> int:
    if not args.protocol_hash:
        raise SystemExit("POLY_PROTOCOL_HASH/--protocol-hash is required")
    if args.target_gpu == args.judge_gpu:
        raise SystemExit("target and staged judge must use distinct physical GPUs")

    args.recon_gpu = args.judge_gpu
    args.guard_gpu = args.judge_gpu
    # No guard worker finalizes arms during the first stage, so permit the
    # target producer to materialize the entire durable generated set.
    args.max_inflight = 160

    root = Path(args.root).resolve()
    for directory in (root, root / "attack", root / "attack" / "_raw"):
        directory.mkdir(parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
    cc = base.load_closed_compare()
    order = json.loads(Path(args.order).read_text(encoding="utf-8"))["order"]
    arm_names = [entry[0] for entry in base.requested_arms(cc, order, args.tlang)]
    work = root / "_pipeline" / args.protocol_hash[:16]
    for name in ("generated", "reconstruction", "guard", "finalized"):
        (work / name).mkdir(parents=True, exist_ok=True)
        os.chmod(work / name, 0o700)

    context = mp.get_context("spawn")
    target = context.Process(
        target=base.target_worker, name="target-gpu",
        args=(args, work, arm_names),
    )
    reconstruction = context.Process(
        target=base.reconstruction_worker, name="reconstruction-gpu",
        args=(args, work, arm_names),
    )
    base.emit(
        "staged_pipeline_started",
        target_gpu=args.target_gpu,
        judge_gpu=args.judge_gpu,
        arms=len(arm_names),
        schedule="target+reconstruction_then_guard",
    )
    run_pair([target, reconstruction])
    base.emit("staged_pipeline_reconstruction_stage_complete", arms=len(arm_names))

    guard = context.Process(
        target=base.guard_worker, name="guard-gpu",
        args=(args, work, arm_names),
    )
    run_pair([guard])
    base.emit("staged_pipeline_guard_stage_complete", arms=len(arm_names))

    aggregates: dict[str, dict[str, Any]] = {}
    for arm in arm_names:
        aggregate = base.merge_arm(args, work, arm)
        if aggregate is None:
            raise RuntimeError(f"staged pipeline left arm incomplete: {arm}")
        aggregates[arm] = aggregate
    ordered = [aggregates[arm] for arm in arm_names]
    write_manifest(args, ordered)
    base.emit("pipeline_complete", arms=len(ordered),
              rows=len(ordered) * args.n_items,
              pipeline="two_gpu_staged_pipeline_v1")
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--order", required=True)
    ap.add_argument("--harm", required=True)
    ap.add_argument("--tlang", required=True)
    ap.add_argument("--n-items", type=int, required=True)
    ap.add_argument("--util", type=float, required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-inflight", type=int, default=160)
    ap.add_argument("--target-gpu", type=int, default=2)
    ap.add_argument("--judge-gpu", type=int, default=3)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--protocol-hash", default=os.environ.get("POLY_PROTOCOL_HASH", ""))
    return ap


def main() -> int:
    args = parser().parse_args()
    if args.n_items < 1 or args.max_tokens < 1 or args.max_model_len < args.max_tokens:
        raise SystemExit("invalid item/token limits")
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
