#!/usr/bin/env python3
"""Attach original Lingua-SafetyBench image files to a static private pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


def secure_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dataset-root", required=True)
    args = ap.parse_args()

    source = Path(args.input)
    root = Path(args.dataset_root).resolve()
    rows = [json.loads(line) for line in source.open(encoding="utf-8")]
    scenario_cache: dict[str, list[dict]] = {}
    image_hashes = {}
    for row in rows:
        scenario = row["scenario"]
        if scenario not in scenario_cache:
            path = root / "2-Text-Dominant Risk" / "English" / f"{scenario}.json"
            scenario_cache[scenario] = json.loads(path.read_text(encoding="utf-8"))
        match = re.search(r"__(\d{4})$", row["item_id"])
        if not match:
            raise ValueError(f"Cannot recover source index: {row['item_id']}")
        source_index = int(match.group(1))
        original = scenario_cache[scenario][source_index]
        if original["question"].strip() != row["original"].strip():
            raise ValueError(f"Question mismatch: {row['item_id']}")
        relative = Path(original["image_path"])
        image_file = (root / "bundle_data" / relative).resolve()
        if not image_file.is_file() or root not in image_file.parents:
            raise FileNotFoundError(image_file)
        digest = image_hashes.setdefault(str(image_file), file_sha256(image_file))
        row["source_index"] = source_index
        row["image_relative_path"] = str(relative)
        row["image_file"] = str(image_file)
        row["image_sha256"] = digest

    output = Path(args.output)
    secure_write(output, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    manifest = {
        "artifact": output.name,
        "source_sha256": file_sha256(source),
        "output_sha256": file_sha256(output),
        "n_items": len(rows),
        "n_unique_images": len(image_hashes),
        "risk_type": "text_dominant",
        "images": "original benign/neutral, semantically relevant Lingua-SafetyBench images",
        "contains_controlled_harmful_prompts": True,
    }
    secure_write(output.with_suffix(".manifest.json"), json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
