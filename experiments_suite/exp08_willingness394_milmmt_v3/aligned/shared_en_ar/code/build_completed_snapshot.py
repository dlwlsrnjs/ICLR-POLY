#!/usr/bin/env python3
"""Build an immutable upload snapshot containing only completed panel stages."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone


def rows(path):
    if not path.exists():
        return 0
    with path.open() as handle:
        return sum(bool(line.strip()) for line in handle)


def load_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source, destination):
    if source.is_dir():
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.lock"),
        )


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    run = args.run.resolve()
    source_package = args.source_package.resolve()
    out = args.out.resolve()
    if not run.is_dir() or not source_package.is_dir():
        parser.error("Invalid run or source package")
    if out.exists():
        parser.error("Snapshot output already exists")
    out.mkdir(parents=True)

    panel = run / "panel"
    tags = sorted(path.name for path in panel.iterdir() if path.is_dir())
    collections = []
    wildguard = []
    reconstruction = []
    priors = []
    progress = {}
    for tag in tags:
        target = panel / tag
        jobs = rows(target / "jobs.jsonl")
        responses = rows(target / "responses.jsonl")
        labels = rows(target / "judge/outputs/wildguard.jsonl")
        reconstruction_summary = load_json(target / "judge/reconstruction/summary.json")
        reconstruction_done = bool(
            reconstruction_summary
            and reconstruction_summary.get("n_rows") == responses
            and responses == jobs
            and jobs > 0
        )
        prior = load_json(target / "willingness_prior.json")
        progress[tag] = {
            "jobs": jobs,
            "responses": responses,
            "wildguard_labels": labels,
            "reconstruction_complete": reconstruction_done,
            "prior_status": prior.get("status") if prior else None,
        }
        if jobs > 0 and responses == jobs:
            collections.append(tag)
        if jobs > 0 and responses == jobs and labels == responses:
            wildguard.append(tag)
        if reconstruction_done:
            reconstruction.append(tag)
        if prior and prior.get("status"):
            priors.append(tag)

    # Freeze run-level reproducibility inputs and status receipts.
    copy_tree(run / "inputs", out / "run/inputs")
    copy_tree(run / "qa", out / "run/qa")
    for source in sorted(run.glob("pipeline_status*.json")):
        copy_file(source, out / "run" / source.name)

    # Freeze the exact source/config/data package used to construct the run.
    copy_tree(source_package, out / "experiment_source")

    # Include only collection-complete models. A downstream stage is copied
    # only if its completion invariant held at snapshot selection time.
    for tag in collections:
        source = panel / tag
        destination = out / "run/panel" / tag
        destination.mkdir(parents=True)
        for name in [
            "blocked.json", "collection_metadata.json", "jobs.jsonl",
            "manifest.json", "responses.jsonl",
        ]:
            path = source / name
            if path.is_file():
                copy_file(path, destination / name)
        copy_tree(source / "analysis", destination / "analysis")

        judge_source = source / "judge"
        judge_destination = destination / "judge"
        if tag in wildguard or tag in reconstruction:
            for name in ["provenance.json"]:
                path = judge_source / name
                if path.is_file():
                    copy_file(path, judge_destination / name)
            copy_tree(judge_source / "code", judge_destination / "code")
            copy_tree(judge_source / "inputs", judge_destination / "inputs")
        if tag in wildguard:
            for name in ["wildguard.runtime.json"]:
                path = judge_source / name
                if path.is_file():
                    copy_file(path, judge_destination / name)
            for name in ["wildguard.jsonl", "wildguard.metadata.json"]:
                path = judge_source / "outputs" / name
                if path.is_file():
                    copy_file(path, judge_destination / "outputs" / name)
        if tag in reconstruction:
            path = judge_source / "reconstruction.runtime.json"
            if path.is_file():
                copy_file(path, judge_destination / path.name)
            copy_tree(judge_source / "reconstruction", judge_destination / "reconstruction")
        if tag in priors:
            copy_file(source / "willingness_prior.json", destination / "willingness_prior.json")

        # Stable logs for completed stages only.
        for source_log, relative in [
            (run / "logs" / f"{tag}.collect.log", Path("run/logs") / f"{tag}.collect.log"),
            (run / "logs/judges_prepass" / f"{tag}.wildguard.log",
             Path("run/logs/judges_prepass") / f"{tag}.wildguard.log"),
        ]:
            if source_log.is_file():
                copy_file(source_log, out / relative)
        if tag in reconstruction:
            source_log = run / "logs/judges_gpu3" / f"{tag}.reconstruction.log"
            if source_log.is_file():
                copy_file(source_log, out / "run/logs/judges_gpu3" / source_log.name)

    cutoff = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    readme = f"""# Willingness panel checkpoint

Immutable private-bucket snapshot captured at `{cutoff}`.

## Included completed work

- Collection-complete models: {len(collections)}/{len(tags)}
- WildGuard-complete models: {len(wildguard)}/{len(tags)}
- Reconstruction-complete models: {len(reconstruction)}/{len(tags)}
- Final prior summaries: {len(priors)}/{len(tags)}

Partial or actively changing model outputs were deliberately excluded.

## What is fixed and what varies

This is **not a fixed-willingness experiment**. The comprehension cell is fixed
and the willingness frame varies. The fixed comprehension cell is
`g3_ordered_n2`: English plus Arabic, three fragments per language (six total),
with within-language fragment order preserved. The same item, translations,
puzzle payload, and comprehension setting are used across all five frames.

The varied willingness frames are `plain`, `persona`, `fiction`, `pap`, and
`persona+fiction`. There were 331 candidate over-refusal items; Arabic semantic
QA retained 284, yielding 1,420 jobs per model. Generation used deterministic
decoding and a 1,024-token output limit as recorded in each config and runtime
metadata file.
"""
    (out / "README.md").write_text(readme)

    files = []
    for path in sorted(value for value in out.rglob("*") if value.is_file()):
        relative = str(path.relative_to(out))
        files.append({"path": relative, "bytes": path.stat().st_size,
                      "sha256": digest(path)})
    manifest = {
        "schema": "willingness331_completed_checkpoint_v1",
        "cutoff_utc": cutoff,
        "source_run": str(run),
        "selection_policy": "collection-complete models and completion-validated downstream stages only",
        "collection_complete_models": collections,
        "wildguard_complete_models": wildguard,
        "reconstruction_complete_models": reconstruction,
        "prior_complete_models": priors,
        "excluded_or_partial_progress": {
            tag: value for tag, value in progress.items() if tag not in collections
        },
        "all_model_progress_at_cutoff": progress,
        "files": files,
        "total_files": len(files),
        "total_bytes_excluding_manifest": sum(value["bytes"] for value in files),
    }
    (out / "SNAPSHOT_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({
        "snapshot": str(out),
        "cutoff_utc": cutoff,
        "collection_complete": len(collections),
        "wildguard_complete": len(wildguard),
        "reconstruction_complete": len(reconstruction),
        "prior_complete": len(priors),
        "files": len(files) + 1,
        "bytes": sum(value["bytes"] for value in files) +
                 (out / "SNAPSHOT_MANIFEST.json").stat().st_size,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
