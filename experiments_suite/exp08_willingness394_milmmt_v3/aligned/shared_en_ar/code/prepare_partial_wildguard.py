#!/usr/bin/env python3
"""Prepare an incremental WildGuard snapshot while response collection appends.

This intentionally creates only WildGuard inputs and copied judge code.  It does
not create final provenance/runtime files; the frozen final judge stage rebuilds
those after collection completes and appends labels for the remaining keys.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()

    run = args.run.resolve()
    aligned = Path(__file__).resolve().parents[2]
    package = aligned.parent
    sys.path.insert(0, str(aligned))
    from run_io import answer_section, load_run

    last_error = None
    for _ in range(10):
        try:
            _, _, responses = load_run(run, complete=False)
            break
        except json.JSONDecodeError as error:
            last_error = error
            time.sleep(0.2)
    else:
        raise RuntimeError("Could not read an append-stable response snapshot") from last_error

    if not responses:
        raise ValueError("No collected responses to judge")

    judge = run / "judge"
    if (judge / "provenance.json").exists() or (judge / "wildguard.runtime.json").exists():
        raise ValueError("Final provenance/runtime already exists; do not run a partial prepass")
    for folder in ("inputs", "outputs", "code"):
        (judge / folder).mkdir(parents=True, exist_ok=True)

    source = package / "runs/wildguard_labeling_v1/code"
    for name in ("batch_label.py", "label.py"):
        src = source / name
        dst = judge / "code" / name
        if dst.exists() and dst.read_bytes() != src.read_bytes():
            raise ValueError(f"Existing judge code differs: {dst}")
        shutil.copy2(src, dst)

    inputs = [
        {
            "key": row["key"],
            "original_request": row["original_request"],
            "response": answer_section(row["response"]),
            "finish_reason": row["finish_reason"],
        }
        for row in responses
    ]
    if len({row["key"] for row in inputs}) != len(inputs):
        raise ValueError("Duplicate response key in partial snapshot")
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in inputs)
    destination = judge / "inputs/jobs.jsonl"
    temporary = destination.with_suffix(".jsonl.tmp")
    temporary.write_text(payload)
    temporary.replace(destination)
    print(json.dumps({
        "status": "partial_wildguard_snapshot_ready",
        "n_rows": len(inputs),
        "input_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "run": str(run),
    }, indent=2))


if __name__ == "__main__":
    main()
