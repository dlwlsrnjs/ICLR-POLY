#!/usr/bin/env python3
"""Extract an aligned Lingua-SafetyBench research panel.

Images are referenced in place, not copied. Unsafe questions are written to owner-readable files.
Alignment is validated across all ten languages before any output is produced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

LANGUAGES = [
    "English", "Arabic", "Chinese", "Finnish", "French",
    "German", "Japanese", "Norwegian", "Russian", "Spanish",
]
SCENARIOS = [
    "Economic Harm", "Fraud", "Hate Speech", "Illegal Activity",
    "Malware Generation", "Physical Harm", "Privacy Violence", "Sex",
]
RISK_PATHS = {
    "text_dominant": Path("2-Text-Dominant Risk"),
    "image_visual": Path("1-Image-Dominant Risk/Visual"),
    "image_typography": Path("1-Image-Dominant Risk/Typography"),
    "image_mixed": Path("1-Image-Dominant Risk/Mixed"),
}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def read_group(root: Path, risk: str, scenario: str):
    by_lang = {}
    for language in LANGUAGES:
        path = root / RISK_PATHS[risk] / language / f"{scenario}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if any(set(item) != {"question", "image_path"} for item in data):
            raise ValueError(f"Unexpected schema: {path}")
        if any(not str(item["question"]).strip() for item in data):
            raise ValueError(f"Empty question: {path}")
        by_lang[language] = data
    counts = {language: len(data) for language, data in by_lang.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Language count mismatch for {risk}/{scenario}: {counts}")
    return by_lang


def aligned_items(root: Path):
    for risk in RISK_PATHS:
        for scenario in SCENARIOS:
            by_lang = read_group(root, risk, scenario)
            count = len(by_lang["English"])
            for index in range(count):
                yield {
                    "item_id": f"{risk}__{slug(scenario)}__{index:04d}",
                    "risk_type": risk,
                    "scenario": scenario,
                    "source_index": index,
                    "text_sufficiency": "high" if risk == "text_dominant" else "requires_visual_derivation",
                    "questions": {
                        language: by_lang[language][index]["question"].strip()
                        for language in LANGUAGES
                    },
                    "image_paths": {
                        language: by_lang[language][index]["image_path"]
                        for language in LANGUAGES
                    },
                }


def balanced_primary_sample(items, n, seed):
    eligible = defaultdict(list)
    for item in items:
        if item["risk_type"] != "text_dominant":
            continue
        # Fifteen tokens gives room for 4–6 non-trivial semantic fragments.
        if len(item["questions"]["English"].split()) >= 15:
            eligible[item["scenario"]].append(item)
    rng = random.Random(seed)
    base, remainder = divmod(n, len(SCENARIOS))
    selected = []
    for pos, scenario in enumerate(SCENARIOS):
        quota = base + int(pos < remainder)
        candidates = eligible[scenario][:]
        rng.shuffle(candidates)
        if len(candidates) < quota:
            raise ValueError(f"Not enough eligible rows for {scenario}")
        chosen = candidates[:quota]
        # Scenario-stratified 60/20/20 assignment, completed globally below.
        selected.extend(chosen)
    rng.shuffle(selected)
    targets = ["dev"] * 60 + ["validation"] * 20 + ["test"] * 20
    if n != 100:
        targets = ["dev"] * round(n * 0.6) + ["validation"] * round(n * 0.2)
        targets += ["test"] * (n - len(targets))
    # Round-robin scenarios across splits improves category balance.
    ordered = sorted(selected, key=lambda x: (SCENARIOS.index(x["scenario"]), x["item_id"]))
    split_slots = {s: targets.count(s) for s in set(targets)}
    assigned = []
    cycle = ["dev", "validation", "dev", "test", "dev"]
    cursor = 0
    for item in ordered:
        while split_slots.get(cycle[cursor % len(cycle)], 0) == 0:
            cursor += 1
        split = cycle[cursor % len(cycle)]
        split_slots[split] -= 1
        cursor += 1
        item = dict(item)
        item["split"] = split
        assigned.append(item)
    rng.shuffle(assigned)
    return assigned


def normalized_english(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def full_primary_set(items, seed):
    """Return all suitable Text-Dominant items, deduplicated before splitting."""
    rng = random.Random(seed)
    candidates = [
        item for item in items
        if item["risk_type"] == "text_dominant"
        and len(item["questions"]["English"].split()) >= 15
    ]
    rng.shuffle(candidates)
    deduped = {}
    for item in candidates:
        deduped.setdefault(normalized_english(item["questions"]["English"]), item)
    by_scenario = defaultdict(list)
    for item in deduped.values():
        by_scenario[item["scenario"]].append(item)
    output = []
    for scenario in SCENARIOS:
        group = by_scenario[scenario]
        rng.shuffle(group)
        n = len(group)
        n_dev = round(n * 0.60)
        n_validation = round(n * 0.20)
        for pos, item in enumerate(group):
            if pos < n_dev:
                split = "dev"
            elif pos < n_dev + n_validation:
                split = "validation"
            else:
                split = "test"
            row = dict(item)
            row["split"] = split
            output.append(row)
    rng.shuffle(output)
    return output


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.chmod(path, 0o600)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default="datasets/Lingua-SafetyBench-bucket/extracted/dataset",
    )
    parser.add_argument("--outdir", default="datasets/lingua_safetybench_text")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    root = Path(args.dataset_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    all_items = list(aligned_items(root))
    expected_records = sum(len(item["questions"]) for item in all_items)
    if expected_records != 100440:
        raise ValueError(f"Expected 100440 language records, got {expected_records}")

    full_path = outdir / "lingua_aligned_text_all.jsonl"
    full_primary_path = outdir / "lingua_polyjigsaw_textdominant_full.jsonl"
    sample_path = outdir / "lingua_polyjigsaw_100.jsonl"
    write_jsonl(full_path, all_items)
    full_primary = full_primary_set(all_items, args.seed)
    write_jsonl(full_primary_path, full_primary)
    sample = balanced_primary_sample(all_items, args.sample_size, args.seed)
    write_jsonl(sample_path, sample)

    manifest = {
        "source_root": str(root),
        "languages": LANGUAGES,
        "scenarios": SCENARIOS,
        "risk_types": list(RISK_PATHS),
        "aligned_base_items": len(all_items),
        "language_records": expected_records,
        "primary_sample": {
            "path": sample_path.name,
            "n": len(sample),
            "risk_type": "text_dominant",
            "scenario_counts": dict(sorted(Counter(x["scenario"] for x in sample).items())),
            "split_counts": dict(sorted(Counter(x["split"] for x in sample).items())),
            "seed": args.seed,
        },
        "full_primary": {
            "path": full_primary_path.name,
            "n": len(full_primary),
            "risk_type": "text_dominant",
            "min_english_words": 15,
            "deduplicated_on": "normalized English question",
            "scenario_counts": dict(sorted(Counter(x["scenario"] for x in full_primary).items())),
            "split_counts": dict(sorted(Counter(x["split"] for x in full_primary).items())),
            "seed": args.seed,
        },
        "files": {
            full_path.name: {"sha256": sha256(full_path), "mode": "0600"},
            full_primary_path.name: {"sha256": sha256(full_primary_path), "mode": "0600"},
            sample_path.name: {"sha256": sha256(sample_path), "mode": "0600"},
        },
        "content_note": "Text files contain unsafe prompts and no model answers.",
    }
    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
