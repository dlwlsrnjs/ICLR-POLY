#!/usr/bin/env python3
"""Continuously WildGuard-judge completed panel models on one spare GPU."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def rows(path):
    if not path.exists():
        return 0
    with path.open() as handle:
        return sum(bool(line.strip()) for line in handle)


def collection_complete(target):
    jobs = rows(target / "jobs.jsonl")
    return jobs > 0 and rows(target / "responses.jsonl") == jobs


def wildguard_complete(target):
    responses = rows(target / "responses.jsonl")
    labels = rows(target / "judge/outputs/wildguard.jsonl")
    return responses > 0 and labels == responses


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--memory-utilization", type=float, default=.75)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    run = args.run.resolve()
    repo = args.repo.resolve()
    python = Path(os.path.abspath(args.python))
    model = args.model_path.resolve()
    if (not run.is_dir() or not repo.is_dir() or not python.is_file() or
            not model.is_dir() or not args.gpu.isdigit() or
            not 0 < args.memory_utilization <= 1 or args.poll_seconds < 5):
        parser.error("Invalid run, repository, executable, model, GPU, or runtime setting")
    if model.name != "cbba4823f3e8020e5a74a5e29bf85072def6f2ff":
        parser.error("WildGuard snapshot differs from the frozen protocol")

    package = Path(__file__).resolve().parents[1]
    aligned = Path(__file__).resolve().parents[2]
    judge = aligned / "judge.py"
    expected = sorted(path.stem for path in (package / "configs").glob("*.json"))
    status_path = run / "pipeline_status_wildguard_gpu3.json"
    log_dir = run / "logs/judges_prepass"
    log_dir.mkdir(parents=True, exist_ok=True)
    failures = {}

    def status(stage, **extra):
        completed = [tag for tag in expected if wildguard_complete(run / "panel" / tag)]
        available = [tag for tag in expected if collection_complete(run / "panel" / tag)]
        value = {
            "stage": stage,
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu": int(args.gpu),
            "completed_wildguard_models": completed,
            "completed_wildguard_count": len(completed),
            "collection_complete_models": available,
            "collection_complete_count": len(available),
            "failed_attempts": failures,
            **extra,
        }
        atomic_json(status_path, value)
        print(json.dumps(value, sort_keys=True), flush=True)

    status("starting")
    while True:
        if all(wildguard_complete(run / "panel" / tag) for tag in expected):
            status("wildguard_queue_complete")
            return 0

        candidate = next((
            tag for tag in expected
            if collection_complete(run / "panel" / tag)
            and not wildguard_complete(run / "panel" / tag)
        ), None)
        if candidate is None:
            status("waiting_for_completed_collection")
            time.sleep(args.poll_seconds)
            continue

        target = run / "panel" / candidate
        status("judging", model=candidate, responses=rows(target / "responses.jsonl"))
        command = [
            str(python), "-u", str(judge), "wildguard",
            "--run", str(target), "--repo", str(repo),
            "--model-path", str(model), "--gpu", args.gpu,
            "--memory-utilization", str(args.memory_utilization),
        ]
        with (log_dir / f"{candidate}.wildguard.log").open("a") as output:
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
        if result.returncode == 0 and wildguard_complete(target):
            failures.pop(candidate, None)
            status("model_complete", model=candidate,
                   labels=rows(target / "judge/outputs/wildguard.jsonl"))
            continue

        failures[candidate] = failures.get(candidate, 0) + 1
        status("judge_failed", model=candidate, returncode=result.returncode,
               labels=rows(target / "judge/outputs/wildguard.jsonl"))
        if failures[candidate] >= 3:
            status("stopped_after_repeated_failure", model=candidate)
            return 2
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
