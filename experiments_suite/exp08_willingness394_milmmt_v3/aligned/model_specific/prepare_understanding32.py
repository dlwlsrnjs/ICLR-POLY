#!/usr/bin/env python3
"""Prepare the final 331-item bank as the exact MJ/LG 32-cell plain grid.

This module deliberately does not reimplement the PolyJigsaw renderer.  It uses
``GridContract``, which hash-pins and calls the renderer used by the existing
MJ/LG runs.  The only adaptation here is from the final release JSONL schema to
the renderer's aligned ``questions`` mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grid_contract import GridContract, digest  # noqa: E402


LANGUAGE_ORDER = (
    "English",
    "Arabic",
    "Chinese",
    "Norwegian",
    "Finnish",
    "Bengali",
    "Thai",
    "Korean",
)

# Preserve the iteration order used by scripts/closed_compare.py::build_arms.
FRAGMENT_COUNTS = (3, 5, 8, 12)
ARRANGEMENTS = ("ordered", "shuffled")
LANGUAGE_COUNTS = (2, 4, 6, 8)
CELLS = tuple(
    f"g{g}_{arrangement}_n{n}"
    for g in FRAGMENT_COUNTS
    for arrangement in ARRANGEMENTS
    for n in LANGUAGE_COUNTS
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_release_files(release: Path) -> dict[str, str]:
    sums_path = release / "SHA256SUMS.json"
    sums = json.loads(sums_path.read_text(encoding="utf-8"))
    required = (
        "data/english_originals.jsonl",
        "data/translations_qa_accepted.jsonl",
    )
    for relative in required:
        path = release / relative
        if relative not in sums:
            raise ValueError(f"Release checksum is missing {relative}")
        actual = sha256(path)
        if actual != sums[relative]:
            raise ValueError(f"Release checksum mismatch: {relative}")
    return {relative: sums[relative] for relative in required}


def load_release(release: Path) -> tuple[list[dict], dict[tuple[str, str], str], dict]:
    checksums = verify_release_files(release)
    items = read_jsonl(release / "data/english_originals.jsonl")
    translations = read_jsonl(release / "data/translations_qa_accepted.jsonl")

    item_ids = [row.get("id") for row in items]
    if len(items) != 331 or len(set(item_ids)) != 331 or None in item_ids:
        raise ValueError("Final release must contain 331 unique English items")
    originals = {row["id"]: row.get("prompt") for row in items}
    if any(not isinstance(text, str) or not text.strip() for text in originals.values()):
        raise ValueError("Final release contains an empty English prompt")

    bank: dict[tuple[str, str], str] = {}
    for row in translations:
        item_id = row.get("item_id")
        language = row.get("language")
        text = row.get("translated")
        key = (item_id, language)
        if item_id not in originals or language not in LANGUAGE_ORDER[1:]:
            raise ValueError(f"Translation outside the frozen bank/profile: {key}")
        if key in bank:
            raise ValueError(f"Duplicate translation: {key}")
        if row.get("qa_pass") is not True or not isinstance(text, str) or not text.strip():
            raise ValueError(f"Translation is not QA accepted: {key}")
        if row.get("english_original") != originals[item_id]:
            raise ValueError(f"Translation source differs from frozen English: {key}")
        bank[key] = text

    expected = {
        (item_id, language)
        for item_id in item_ids
        for language in LANGUAGE_ORDER[1:]
    }
    missing = expected - set(bank)
    extra = set(bank) - expected
    if missing or extra or len(bank) != 2317:
        raise ValueError(
            f"Final release translation matrix is incomplete: missing={len(missing)}, extra={len(extra)}"
        )
    provenance = {
        "release": str(release.resolve()),
        "sha256sums_sha256": sha256(release / "SHA256SUMS.json"),
        "data_checksums": checksums,
        "english_items": len(items),
        "translation_pairs": len(bank),
    }
    return items, bank, provenance


def load_splits(plan_path: Path, item_ids: set[str]) -> tuple[dict[str, str], dict]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    splits = plan.get("splits")
    if not isinstance(splits, dict) or set(splits) != item_ids:
        raise ValueError("The fixed split does not cover exactly the final 331 items")
    counts = {phase: sum(value == phase for value in splits.values()) for phase in ("selection", "validation")}
    if counts != {"selection": 100, "validation": 231}:
        raise ValueError(f"Unexpected fixed split counts: {counts}")
    if set(splits.values()) != {"selection", "validation"}:
        raise ValueError("The fixed split contains an unknown phase")
    provenance = {
        "path": str(plan_path.resolve()),
        "sha256": sha256(plan_path),
        "split_sha256": digest(splits),
        "counts": counts,
    }
    return splits, provenance


def language_count_for_cell(cell: str) -> int:
    n = int(cell.rsplit("n", 1)[1])
    if n not in LANGUAGE_COUNTS:
        raise ValueError(f"Invalid language count in cell: {cell}")
    return n


def language_permutation(item_id: str, seed: int) -> list[str]:
    """Return one stable per-item permutation of the seven non-English languages."""
    material = f"understanding32-language-v1|{seed}|{item_id}".encode()
    stable_seed = int.from_bytes(hashlib.sha256(material).digest(), "big")
    languages = list(LANGUAGE_ORDER[1:])
    random.Random(stable_seed).shuffle(languages)
    return languages


def languages_for_item(item_id: str, cell: str, seed: int) -> list[str]:
    n = language_count_for_cell(cell)
    return ["English", *language_permutation(item_id, seed)[: n - 1]]


def render_puzzles(
    items: list[dict],
    bank: dict[tuple[str, str], str],
    splits: dict[str, str],
    contract: GridContract,
    language_seed: int,
) -> list[dict]:
    jobs: list[dict] = []
    for item in items:
        item_id = item["id"]
        split = splits[item_id]
        questions = {"English": item["prompt"]}
        questions.update(
            {language: bank[(item_id, language)] for language in LANGUAGE_ORDER[1:]}
        )
        for cell in CELLS:
            g_text, arrangement, _ = cell.split("_")
            languages = languages_for_item(item_id, cell, language_seed)
            # Existing MJ/LG runs pass only the cell's language prefix to the
            # renderer.  Keep source_translations metadata identical as well as
            # the rendered prompt/payload.
            row = {
                "item_id": item_id,
                "scenario": "Benign Control",
                "questions": {language: questions[language] for language in languages},
            }
            rendered = contract.render(
                row,
                languages,
                int(g_text.removeprefix("g")),
                arrangement,
                "plain",
            )
            if rendered["gold_english"] != item["prompt"]:
                raise AssertionError("Renderer changed the frozen English request")
            if list(rendered["source_translations"]) != languages:
                raise AssertionError("Renderer language order differs from the frozen profile")
            jobs.append(
                {
                    "id": item_id,
                    "dataset": "overrefusal331_understanding_axis_v2",
                    "cell": cell,
                    "frame": "plain",
                    "arm": f"{cell}__plain",
                    "split": split,
                    "languages": languages,
                    "original_request": item["prompt"],
                    **rendered,
                }
            )
    return jobs


def model_config(base: dict, phase: str, release_provenance: dict, split_provenance: dict, item_selection: dict, language_seed: int) -> dict:
    config = dict(base)
    for key in ("cell", "languages", "language_profile", "benign_selection_sha256"):
        config.pop(key, None)
    config.update(
        schema="understanding_axis32_v2",
        dataset="overrefusal331_understanding_axis_v2",
        expected_items=331,
        cell="multiple",
        frames=["plain"],
        cells=list(CELLS),
        phase=phase,
        available_languages=list(LANGUAGE_ORDER),
        language_selection={
            "schema": "nested_per_item_language_permutation_v1",
            "seed": language_seed,
            "english_always_included": True,
            "non_english_pool": list(LANGUAGE_ORDER[1:]),
            "nesting": "For each item, n2 is a prefix of n4, which is a prefix of n6, which is a prefix of n8.",
            "shared_across_models_g_and_arrangements": True,
        },
        release_provenance=release_provenance,
        split_provenance=split_provenance,
        item_selection=item_selection,
        renderer_contract="exact_mj_lg_grid_contract",
        strong_reconstruction_prompt=False,
        system_message=None,
    )
    return config


def materialize_model(dest: Path, base: dict, puzzles: list[dict], phase: str, release_provenance: dict, split_provenance: dict, item_selection: dict, language_seed: int) -> dict:
    config = model_config(base, phase, release_provenance, split_provenance, item_selection, language_seed)
    jobs = []
    for puzzle in puzzles:
        job = dict(puzzle)
        job["key"] = digest({"config": config, "job": job})
        jobs.append(job)
    expected_items = item_selection["selected_items"]
    expected_jobs = expected_items * len(CELLS)
    if len(jobs) != expected_jobs or len({row["key"] for row in jobs}) != expected_jobs:
        raise AssertionError("Prepared job count or key uniqueness is wrong")
    if {row["frame"] for row in jobs} != {"plain"} or {row["cell"] for row in jobs} != set(CELLS):
        raise AssertionError("Prepared jobs do not contain the exact plain 32-cell grid")
    manifest = {
        "schema": "understanding_axis32_v2_run",
        "config": config,
        "config_sha256": digest(config),
        "jobs_sha256": digest(jobs),
        "jobs": len(jobs),
        "candidate_items": expected_items,
        "eligible_items": expected_items,
        "blocked_items": 0,
        "cells": len(CELLS),
        "frames": ["plain"],
        "same_items_in_all_cells": True,
    }
    manifest_path = dest / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError(f"Frozen inputs changed; choose a new output directory: {dest}")
    write_jsonl(dest / "jobs.jsonl", jobs)
    write_json(dest / "blocked.json", {})
    write_json(manifest_path, manifest)
    return {
        "model_tag": base["model_tag"],
        "target_model": base["target_model"],
        "target_revision": base["target_revision"],
        "run": str(dest.resolve()),
        "jobs": len(jobs),
        "manifest_sha256": sha256(manifest_path),
    }


def prepare_all(
    release: Path,
    configs: Path,
    split_plan: Path,
    repo: Path,
    out: Path,
    language_seed: int = 20260920,
) -> dict:
    phase = "all"
    preparer_sha256 = sha256(Path(__file__))
    items, bank, release_provenance = load_release(release)
    splits, split_provenance = load_splits(split_plan, {row["id"] for row in items})
    config_paths = sorted(configs.glob("*.json"))
    if len(config_paths) != 17:
        raise ValueError(f"Expected exactly 17 pinned model configs, found {len(config_paths)}")
    bases = [json.loads(path.read_text(encoding="utf-8")) for path in config_paths]
    if len({base.get("model_tag") for base in bases}) != 17:
        raise ValueError("Model configs contain missing or duplicate model tags")
    renderer_hashes = {digest(base.get("renderer_sha256")) for base in bases}
    if len(renderer_hashes) != 1:
        raise ValueError("The 17 model configs do not pin one common renderer")
    contract = GridContract(repo, bases[0]["renderer_sha256"])
    item_selection = {
        "phase_pool": "all",
        "candidate_items": 331,
        "selected_items": 331,
        "method": "all_candidates",
        "sample_seed": None,
        "selected_ids_sha256": digest(sorted(row["id"] for row in items)),
    }
    puzzles = render_puzzles(items, bank, splits, contract, language_seed)

    shared = out / "shared" / phase
    write_jsonl(shared / "puzzles.jsonl", puzzles)
    shared_manifest = {
        "schema": "understanding_axis32_v2_shared_puzzles",
        "phase": phase,
        "items": len({row["id"] for row in puzzles}),
        "cells": list(CELLS),
        "jobs": len(puzzles),
        "frames": ["plain"],
        "available_languages": list(LANGUAGE_ORDER),
        "language_selection": {
            "schema": "nested_per_item_language_permutation_v1",
            "seed": language_seed,
            "english_always_included": True,
            "shared_across_models_g_and_arrangements": True,
        },
        "renderer_sha256": bases[0]["renderer_sha256"],
        "preparer_sha256": preparer_sha256,
        "puzzles_sha256": digest(puzzles),
        "release_provenance": release_provenance,
        "split_provenance": split_provenance,
        "item_selection": item_selection,
    }
    write_json(shared / "manifest.json", shared_manifest)

    models = [
        materialize_model(
            out / "models" / base["model_tag"] / phase,
            base,
            puzzles,
            phase,
            release_provenance,
            split_provenance,
            item_selection,
            language_seed,
        )
        for base in bases
    ]
    plan = {
        "schema": "understanding_axis32_v2_preparation",
        "phase": phase,
        "models": models,
        "model_count": len(models),
        "cells": list(CELLS),
        "jobs_per_model": len(puzzles),
        "total_target_jobs": sum(model["jobs"] for model in models),
        "shared_puzzles": str((shared / "puzzles.jsonl").resolve()),
        "shared_puzzles_sha256": digest(puzzles),
        "preparer_sha256": preparer_sha256,
        "release_provenance": release_provenance,
        "split_provenance": split_provenance,
        "item_selection": item_selection,
        "language_selection": shared_manifest["language_selection"],
    }
    write_json(out / f"PREPARATION_{phase.upper()}.json", plan)
    return plan


def main() -> None:
    here = Path(__file__).resolve()
    aligned = here.parents[1]
    project = here.parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--configs",
        type=Path,
        default=aligned / "shared_en_ar" / "configs",
        help="Directory containing the 17 pinned model configs",
    )
    parser.add_argument(
        "--split-plan",
        type=Path,
        default=here.with_name("TRANSITION331_PLAN.json"),
        help="Existing frozen 100/231 split",
    )
    parser.add_argument("--repo", type=Path, default=project)
    parser.add_argument(
        "--language-seed",
        type=int,
        default=20260920,
        help="Seed for the nested per-item random permutation of non-English languages",
    )
    args = parser.parse_args()
    plan = prepare_all(
        args.release,
        args.configs,
        args.split_plan,
        args.repo,
        args.out,
        args.language_seed,
    )
    print(
        json.dumps(
            {
                "phase": plan["phase"],
                "models": plan["model_count"],
                "cells": len(plan["cells"]),
                "jobs_per_model": plan["jobs_per_model"],
                "total_target_jobs": plan["total_target_jobs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
