#!/usr/bin/env python3
"""Run the current-panel selector fold/seed matrix across available GPUs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from collections import defaultdict
from pathlib import Path

import numpy as np


FAMILIES = ("falcon", "gemma", "glm", "llama", "mistral", "phi", "qwen")


def summarize(outdir: Path, expected_jobs=21):
    records = []
    for path in sorted(outdir.glob("selector_*.json")):
        if path.name == "selector_familyloo_summary.json":
            continue
        try:
            records.extend(json.loads(path.read_text(encoding="utf-8"))["records"])
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["method"], row["budget"], row["held_family"])].append(row["test_asr"])
    rows = []
    for method, budget in sorted({(r["method"], r["budget"]) for r in records}):
        family_scores = {
            family: float(np.mean(grouped[(method, budget, family)]))
            for family in FAMILIES
            if grouped[(method, budget, family)]
        }
        rows.append(
            {
                "method": method,
                "budget": budget,
                "family_macro_test_asr": float(np.mean(list(family_scores.values()))),
                "family_scores": family_scores,
                "n_records": sum(
                    len(grouped[(method, budget, family)]) for family in family_scores
                ),
            }
        )
    payload = {
        "protocol": {
            "status": "complete" if len(list(outdir.glob("selector_*_seed*.json"))) == expected_jobs else "partial",
            "expected_jobs": expected_jobs,
            "completed_jobs": len(list(outdir.glob("selector_*_seed*.json"))),
            "aggregation": "model mean within held family, then macro mean over seven families",
        },
        "summary": rows,
        "records": records,
    }
    (outdir / "selector_familyloo_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--benign-root", type=Path, required=True)
    parser.add_argument("--false-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--datasets", nargs="+", choices=("joint", "mj", "lg"), default=["joint"])
    parser.add_argument("--split-seed", type=int, default=47)
    parser.add_argument("--supervised-steps", type=int, default=400)
    parser.add_argument("--ppo-updates", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=96)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    jobs = [(dataset, family, seed) for dataset in args.datasets for family in FAMILIES for seed in args.seeds]
    locks = threading.Lock()
    failures = []

    def worker(gpu, assigned):
        for dataset, family, seed in assigned:
            output = args.outdir / f"selector_{dataset}_{family}_seed{seed}.json"
            if output.exists():
                print(f"skip complete family={family} seed={seed}", flush=True)
                continue
            log = args.outdir / f"selector_{dataset}_{family}_seed{seed}.log"
            command = [
                sys.executable,
                str(Path(__file__).with_name("train_current_asr_selector_gpu.py")),
                "--raw-root", str(args.raw_root),
                "--benign-root", str(args.benign_root),
                "--false-root", str(args.false_root),
                "--held-family", family,
                "--dataset", dataset,
                "--seed", str(seed),
                "--split-seed", str(args.split_seed),
                "--device", f"cuda:{gpu}",
                "--supervised-steps", str(args.supervised_steps),
                "--ppo-updates", str(args.ppo_updates),
                "--batch-size", str(args.batch_size),
                "--out", str(output),
            ]
            print(f"start gpu={gpu} dataset={dataset} family={family} seed={seed}", flush=True)
            with log.open("w", encoding="utf-8") as handle:
                result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
            with locks:
                if result.returncode:
                    failures.append({"gpu": gpu, "dataset": dataset, "family": family, "seed": seed, "code": result.returncode})
                    print(f"FAILED gpu={gpu} dataset={dataset} family={family} seed={seed}", flush=True)
                else:
                    print(f"done gpu={gpu} dataset={dataset} family={family} seed={seed}", flush=True)
                summarize(args.outdir, len(jobs))

    partitions = [jobs[index::len(args.gpus)] for index in range(len(args.gpus))]
    threads = [
        threading.Thread(target=worker, args=(gpu, partition), daemon=False)
        for gpu, partition in zip(args.gpus, partitions)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    payload = summarize(args.outdir, len(jobs))
    status = {
        "failures": failures,
        "completed_jobs": payload["protocol"]["completed_jobs"],
        "expected_jobs": len(jobs),
    }
    (args.outdir / "queue_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
