#!/usr/bin/env python3
"""Inventory cached target checkpoints; never equate cache presence with a run.

Only a reviewed catalogue of general text targets is included. Family grouping
is deliberately conservative for splitting: Qwen generations stay together,
and Zephyr stays with its Mistral lineage. No model is loaded or downloaded.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path


TARGETS = [
    ("Qwen/Qwen2.5-1.5B-Instruct", "qwen"),
    ("Qwen/Qwen2.5-3B-Instruct", "qwen"),
    ("Qwen/Qwen2.5-7B-Instruct", "qwen"),
    ("Qwen/Qwen2.5-14B-Instruct", "qwen"),
    ("Qwen/Qwen2.5-32B-Instruct", "qwen"),
    ("Qwen/Qwen3-8B", "qwen"),
    ("microsoft/Phi-3.5-mini-instruct", "phi"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
    ("HuggingFaceH4/zephyr-7b-beta", "mistral"),
    ("allenai/OLMo-2-1124-7B-Instruct", "olmo"),
    ("internlm/internlm2_5-7b-chat", "internlm"),
    ("tiiuae/Falcon3-7B-Instruct", "falcon"),
]


def inspect_snapshot(path):
    config = json.loads((path / "config.json").read_text())
    index = path / "model.safetensors.index.json"
    if index.is_file():
        expected = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    elif (path / "model.safetensors").is_file():
        expected = ["model.safetensors"]
    else:
        expected = []
    missing = [name for name in expected if not (path / name).is_file()]
    tokenizer_files = [name for name in ("tokenizer.json", "tokenizer.model")
                       if (path / name).is_file()]
    tokenizer_config = path / "tokenizer_config.json"
    return {
        "revision": path.name,
        "snapshot_path": str(path),
        "architecture": config.get("architectures", []),
        "expected_weight_files": len(expected),
        "missing_weight_files": missing,
        "weight_bytes": sum((path / name).stat().st_size for name in expected
                            if (path / name).is_file()),
        "tokenizer_files": tokenizer_files,
        "tokenizer_config_present": tokenizer_config.is_file(),
        "local_files_present": bool(expected) and not missing
                               and bool(tokenizer_files) and tokenizer_config.is_file(),
        "inference_verified_by_this_inventory": False,
    }


def inventory(cache):
    records = []
    for model_id, group in TARGETS:
        snapshots = []
        for base in (cache, cache / "hub"):
            folder = base / ("models--" + model_id.replace("/", "--")) / "snapshots"
            for snapshot in sorted(folder.glob("*")):
                if (snapshot / "config.json").is_file():
                    snapshots.append(inspect_snapshot(snapshot))
        records.append({
            "model_id": model_id,
            "family_split_group": group,
            "snapshots": snapshots,
            "local_files_present": any(s["local_files_present"] for s in snapshots),
        })
    cached = [r for r in records if r["local_files_present"]]
    groups = dict(sorted(Counter(r["family_split_group"] for r in cached).items()))
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_root": str(cache),
        "scope": "Reviewed general text targets; not all cached models or historical API targets",
        "cache_check_limit": "File presence only; no checksum, runtime or chat-template validation",
        "local_target_checkpoints": len(cached),
        "conservative_family_split_groups": len(groups),
        "checkpoint_count_by_group": groups,
        "new_target_calls_in_this_inventory": 0,
        "excluded_roles": ["moderation judge", "translation model", "multimodal specialist"],
        "targets": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = inventory(args.cache)
    # A new snapshot is explicit; do not silently overwrite an older inventory.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({k: report[k] for k in (
        "local_target_checkpoints", "conservative_family_split_groups",
        "checkpoint_count_by_group", "new_target_calls_in_this_inventory")}, indent=2))


if __name__ == "__main__":
    main()
