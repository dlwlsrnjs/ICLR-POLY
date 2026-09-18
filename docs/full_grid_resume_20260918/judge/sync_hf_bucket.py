#!/usr/bin/env python3
"""Incrementally upload completed PolyJigsaw files to a private HF Bucket.

Only complete, validated files are uploaded. Raw and judged JSONL files remain
the source of truth on local storage; the Bucket is a concurrent durable copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import batch_bucket_files


MODELS = (
    "qwen25_14b",
    "qwen25_32b",
    "gemma2_27b",
    "mistral24b",
    "phi3_medium_14b",
)
DATASET_ROWS = {"lg": 250, "mj": 315}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bucket-id", default="jin-kwon/poly")
    parser.add_argument(
        "--prefix",
        default=(
            "PolyJigsaw/full_grid_20260917_v1/primary_large/"
            "incremental/full_grid_5models_bf16"
        ),
    )
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--batch-files", type=int, default=32)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return rows


def validate_jsonl(path: Path, kind: str, dataset: str) -> str:
    rows = read_jsonl(path)
    expected = DATASET_ROWS[dataset]
    if len(rows) != expected:
        raise ValueError(f"{path}: {len(rows)} rows, expected {expected}")
    ids = [str(row.get("item_id")) for row in rows]
    if len(set(ids)) != expected or "None" in ids:
        raise ValueError(f"{path}: item_id set is incomplete or duplicated")
    common = {"item_id", "reconstruction", "answer", "raw_output"}
    if any(not common.issubset(row) for row in rows):
        raise ValueError(f"{path}: missing required response fields")
    if kind == "raw" and any({"R", "U", "J"}.intersection(row) for row in rows):
        raise ValueError(f"{path}: raw file contains judge fields")
    if kind == "judged" and any(
        not {"R", "U", "J", "reconstruction_judge_raw", "guard_judge_raw"}.issubset(row)
        for row in rows
    ):
        raise ValueError(f"{path}: judged file is incomplete")
    return sha256_file(path)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_state(path: Path) -> dict:
    if not path.is_file():
        return {"version": 1, "files": {}}
    with path.open() as handle:
        state = json.load(handle)
    if not isinstance(state.get("files"), dict):
        raise ValueError(f"invalid upload state: {path}")
    return state


def dataset_from_path(path: Path) -> str:
    for part in path.parts:
        if part in DATASET_ROWS:
            return part
    raise ValueError(f"cannot determine dataset: {path}")


def validated_candidates(run_root: Path) -> tuple[list[tuple[Path, str]], list[str]]:
    candidates: list[tuple[Path, str]] = []
    errors: list[str] = []
    raw_hashes: dict[str, str] = {}

    for path in sorted((run_root / "raw").glob("*/*.jsonl")):
        try:
            digest = validate_jsonl(path, "raw", dataset_from_path(path))
            candidates.append((path, digest))
            raw_hashes[str(path.resolve())] = digest
        except (OSError, ValueError) as error:
            errors.append(str(error))

    for path in sorted((run_root / "judged").glob("*/*.jsonl")):
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        try:
            digest = validate_jsonl(path, "judged", dataset_from_path(path))
            if not meta_path.is_file():
                raise ValueError(f"{path}: missing metadata")
            metadata = json.loads(meta_path.read_text())
            source = str(Path(metadata["source"]).resolve())
            source_digest = raw_hashes.get(source)
            if source_digest is None or metadata.get("source_sha256") != source_digest:
                raise ValueError(f"{path}: source SHA-256 mismatch")
            if metadata.get("rows") != DATASET_ROWS[dataset_from_path(path)]:
                raise ValueError(f"{path}: metadata row count mismatch")
            if not metadata.get("recon_revision") or not metadata.get("guard_revision"):
                raise ValueError(f"{path}: missing judge revisions")
            candidates.extend(((path, digest), (meta_path, sha256_file(meta_path))))
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(str(error))

    return candidates, errors


def remote_path(prefix: str, run_root: Path, path: Path) -> str:
    return f"{prefix.rstrip('/')}/{path.relative_to(run_root).as_posix()}"


def manifest_value(args: argparse.Namespace, state: dict, errors: list[str]) -> dict:
    return {
        "schema": "polyjigsaw-hf-incremental-v1",
        "bucket_id": args.bucket_id,
        "prefix": args.prefix,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "files": state["files"],
        "validation_errors": errors,
    }


def pipeline_complete(state: dict) -> bool:
    files = state["files"]
    for model in MODELS:
        for dataset in DATASET_ROWS:
            raw_prefix = f"raw/{dataset}/{model}_{dataset}__"
            judged_prefix = f"judged/{dataset}/{model}_{dataset}__"
            if sum(key.startswith(raw_prefix) and key.endswith(".jsonl") for key in files) != 160:
                return False
            if sum(key.startswith(judged_prefix) and key.endswith(".jsonl") for key in files) != 160:
                return False
            if sum(key.startswith(judged_prefix) and key.endswith(".meta.json") for key in files) != 160:
                return False
    return True


def run_once(args: argparse.Namespace, state_path: Path, manifest_path: Path) -> bool:
    state = load_state(state_path)
    candidates, errors = validated_candidates(args.run_root)
    pending: list[tuple[Path, str, str, str]] = []
    for path, digest in candidates:
        relative = path.relative_to(args.run_root).as_posix()
        remote = remote_path(args.prefix, args.run_root, path)
        if state["files"].get(relative, {}).get("sha256") != digest:
            pending.append((path, digest, relative, remote))

    print(
        f"scan valid={len(candidates)} pending={len(pending)} errors={len(errors)}",
        flush=True,
    )
    for error in errors[:20]:
        print(f"validation_error: {error}", flush=True)
    if args.dry_run:
        return False

    for start in range(0, len(pending), args.batch_files):
        batch = pending[start : start + args.batch_files]
        batch_bucket_files(
            args.bucket_id,
            add=[(path, remote) for path, _, _, remote in batch],
        )
        uploaded_at = datetime.now(timezone.utc).isoformat()
        for path, digest, relative, remote in batch:
            state["files"][relative] = {
                "sha256": digest,
                "size": path.stat().st_size,
                "remote_path": remote,
                "uploaded_at": uploaded_at,
            }
        atomic_json(state_path, state)
        print(f"uploaded {min(start + len(batch), len(pending))}/{len(pending)}", flush=True)

    manifest = manifest_value(args, state, errors)
    atomic_json(manifest_path, manifest)
    batch_bucket_files(
        args.bucket_id,
        add=[(manifest_path, f"{args.prefix.rstrip('/')}/manifest/incremental_manifest.json")],
    )

    if pipeline_complete(state) and not errors:
        receipt_path = manifest_path.parent / "FINAL_RECEIPT.json"
        receipt = {
            "schema": "polyjigsaw-hf-final-receipt-v1",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": sha256_file(manifest_path),
            "uploaded_entries": len(state["files"]),
            "expected_raw_jsonl": 1600,
            "expected_judged_jsonl": 1600,
            "expected_judge_metadata": 1600,
        }
        atomic_json(receipt_path, receipt)
        batch_bucket_files(
            args.bucket_id,
            add=[(receipt_path, f"{args.prefix.rstrip('/')}/receipts/FINAL_RECEIPT.json")],
        )
        print("FINAL_RECEIPT uploaded", flush=True)
        return True
    return False


def main() -> int:
    args = parse_args()
    if args.interval < 1 or args.batch_files < 1:
        raise SystemExit("--interval and --batch-files must be positive")
    sync_dir = args.run_root / "hf_sync"
    state_path = sync_dir / "upload_state.json"
    manifest_path = sync_dir / "incremental_manifest.json"
    while True:
        try:
            complete = run_once(args, state_path, manifest_path)
        except Exception as error:
            print(f"sync_error: {type(error).__name__}: {error}", flush=True)
            complete = False
        if args.once or args.dry_run or complete:
            return 0 if not args.dry_run or not complete else 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
