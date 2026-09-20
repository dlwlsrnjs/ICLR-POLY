#!/usr/bin/env python3
"""Finish WildGuard, reconstruction, and summaries on one spare GPU.

The queue continuously discovers completed target collections. It prioritizes
missing WildGuard labels, then semantic reconstruction, then CPU aggregation.
Every underlying stage uses its existing durable output protocol.
"""

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


def reconstruction_complete(target):
    path = target / "judge/reconstruction/summary.json"
    if not path.exists():
        return False
    try:
        summary = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return summary.get("n_rows") == rows(target / "responses.jsonl")


def prior_complete(target):
    path = target / "willingness_prior.json"
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(value.get("status"))


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--reconstruction-model", type=Path, required=True)
    parser.add_argument("--wildguard-model", type=Path, required=True)
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    run = args.run.resolve()
    repo = args.repo.resolve()
    python = Path(os.path.abspath(args.python))
    reconstruction_model = args.reconstruction_model.resolve()
    wildguard_model = args.wildguard_model.resolve()
    if (not run.is_dir() or not repo.is_dir() or not python.is_file() or
            not reconstruction_model.is_dir() or not wildguard_model.is_dir() or
            not args.gpu.isdigit() or args.poll_seconds < 5):
        parser.error("Invalid run, repository, executable, judge model, GPU, or poll interval")
    if reconstruction_model.name != "a09a35458c702b33eeacc393d103063234e8bc28":
        parser.error("Reconstruction snapshot differs from the frozen protocol")
    if wildguard_model.name != "cbba4823f3e8020e5a74a5e29bf85072def6f2ff":
        parser.error("WildGuard snapshot differs from the frozen protocol")

    package = Path(__file__).resolve().parents[1]
    aligned = Path(__file__).resolve().parents[2]
    judge = aligned / "judge.py"
    summarize = aligned / "summarize.py"
    expected = sorted(path.stem for path in (package / "configs").glob("*.json"))
    status_path = run / "pipeline_status_serial_judges_gpu3.json"
    log_dir = run / "logs/judges_gpu3"
    log_dir.mkdir(parents=True, exist_ok=True)
    failures = {}

    def model_sets():
        collected = [tag for tag in expected if collection_complete(run / "panel" / tag)]
        wildguard = [tag for tag in expected if wildguard_complete(run / "panel" / tag)]
        reconstruction = [tag for tag in expected if reconstruction_complete(run / "panel" / tag)]
        priors = [tag for tag in expected if prior_complete(run / "panel" / tag)]
        return collected, wildguard, reconstruction, priors

    def status(stage, **extra):
        collected, wildguard, reconstruction, priors = model_sets()
        value = {
            "stage": stage,
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu": int(args.gpu),
            "collection_complete_count": len(collected),
            "wildguard_complete_count": len(wildguard),
            "reconstruction_complete_count": len(reconstruction),
            "prior_complete_count": len(priors),
            "collection_complete_models": collected,
            "wildguard_complete_models": wildguard,
            "reconstruction_complete_models": reconstruction,
            "prior_complete_models": priors,
            "failed_attempts": failures,
            **extra,
        }
        atomic_json(status_path, value)
        print(json.dumps(value, sort_keys=True), flush=True)

    def run_stage(tag, stage, command):
        key = f"{tag}:{stage}"
        status(stage, model=tag)
        with (log_dir / f"{tag}.{stage}.log").open("a") as output:
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
        return key, result.returncode

    status("starting")
    while True:
        targets = {tag: run / "panel" / tag for tag in expected}
        if all(prior_complete(targets[tag]) for tag in expected):
            status("judge_and_prior_queue_complete")
            return 0

        tag = next((tag for tag in expected
                    if collection_complete(targets[tag])
                    and not wildguard_complete(targets[tag])), None)
        if tag is not None:
            key, returncode = run_stage(tag, "wildguard", [
                str(python), "-u", str(judge), "wildguard",
                "--run", str(targets[tag]), "--repo", str(repo),
                "--model-path", str(wildguard_model), "--gpu", args.gpu,
                "--memory-utilization", ".75",
            ])
            success = returncode == 0 and wildguard_complete(targets[tag])
        else:
            tag = next((tag for tag in expected
                        if collection_complete(targets[tag])
                        and not reconstruction_complete(targets[tag])), None)
            if tag is not None:
                key, returncode = run_stage(tag, "reconstruction", [
                    str(python), "-u", str(judge), "reconstruction",
                    "--run", str(targets[tag]), "--repo", str(repo),
                    "--model-path", str(reconstruction_model), "--gpu", args.gpu,
                    "--batch-size", "16",
                ])
                success = returncode == 0 and reconstruction_complete(targets[tag])
            else:
                tag = next((tag for tag in expected
                            if collection_complete(targets[tag])
                            and wildguard_complete(targets[tag])
                            and reconstruction_complete(targets[tag])
                            and not prior_complete(targets[tag])), None)
                if tag is not None:
                    key, returncode = run_stage(tag, "summarize", [
                        str(python), "-u", str(summarize),
                        "--run", str(targets[tag]),
                    ])
                    success = returncode == 0 and prior_complete(targets[tag])
                else:
                    status("waiting_for_completed_collection")
                    time.sleep(args.poll_seconds)
                    continue

        if success:
            failures.pop(key, None)
            status("stage_complete", model=tag, completed_stage=key.split(":", 1)[1])
            continue
        failures[key] = failures.get(key, 0) + 1
        status("stage_failed", model=tag, failed_stage=key.split(":", 1)[1],
               returncode=returncode)
        if failures[key] >= 3:
            status("stopped_after_repeated_failure", model=tag,
                   failed_stage=key.split(":", 1)[1])
            return 2
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
