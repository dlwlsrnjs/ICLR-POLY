#!/usr/bin/env python3
"""Sweep the interleaving load and collect harm-free reconstruction metrics.

Runs ``run_qwen_interleaving_probe`` across a grid of language counts and display
conditions, then writes one combined summary. This is the reconstruction-difficulty
curve: it shows how far the interleaving load can be pushed before the target model
can no longer recover the meaning. It is the safe precondition for calibrating any
later reconstruction-gated safety experiment, and it runs on benign FLORES by default.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("lingua", "flores"), default="flores")
    parser.add_argument("--flores-dataset", default="/home/ubuntu/342/jinkwon/datasets/flores200/dev.parquet")
    parser.add_argument("--lingua-dataset", default="../datasets/lingua_safetybench_text/lingua_aligned_text_all.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-items", type=int, default=20)
    parser.add_argument("--granularity", choices=("coarse", "fine"), default="coarse")
    parser.add_argument("--language-counts", type=int, nargs="+", default=[2, 4, 6, 8, 10])
    parser.add_argument("--conditions", nargs="+", default=["ordered", "shuffled"])
    parser.add_argument("--workdir", type=Path, default=Path("private_artifacts/interleaving_sweep"))
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    cells = []
    for num_languages in args.language_counts:
        for condition in args.conditions:
            tag = f"n{num_languages}_{condition}_{args.granularity}"
            cell_summary = args.workdir / f"summary_{tag}.json"
            cmd = [
                sys.executable, str(here / "run_qwen_interleaving_probe.py"),
                "--source", args.source,
                "--flores-dataset", args.flores_dataset,
                "--lingua-dataset", args.lingua_dataset,
                "--model", args.model, "--device", args.device,
                "--num-items", str(args.num_items),
                "--num-languages", str(num_languages),
                "--condition", condition,
                "--granularity", args.granularity,
                "--outdir", str(args.workdir / f"artifact_{tag}"),
                "--summary", str(cell_summary),
            ]
            print(f"[sweep] {tag}", flush=True)
            subprocess.run(cmd, check=True)
            data = json.loads(cell_summary.read_text())
            cells.append({
                "num_languages": num_languages,
                "condition": condition,
                "granularity": args.granularity,
                "n_items": data["n_items"],
                **data["aggregate"],
            })

    combined = {
        "source": args.source,
        "model": args.model,
        "granularity": args.granularity,
        "task": "reconstruction_and_classification_only_no_execution",
        "grid": cells,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(combined, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
