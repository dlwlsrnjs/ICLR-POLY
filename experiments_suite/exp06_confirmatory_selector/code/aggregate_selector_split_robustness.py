#!/usr/bin/env python3
"""Aggregate family-balanced selector results equally across item splits."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def records_from(directory):
    rows = []
    for path in sorted(directory.glob("selector_*_seed*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["records"])
    incumbent = directory / "incumbent_evaluation.json"
    if incumbent.exists():
        rows.extend(json.loads(incumbent.read_text(encoding="utf-8"))["records"])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    all_records = []
    for directory in args.dirs:
        all_records.extend(records_from(directory))

    # Collapse random repetitions first, then model/dataset/training seeds within
    # each held family.  Item splits and families retain equal final weight.
    leaf = defaultdict(list)
    for row in all_records:
        leaf[
            (
                row["split_seed"],
                row["method"],
                row["budget"],
                row["held_family"],
                row["dataset"],
                row["model"],
                row["seed"],
            )
        ].append(row["test_asr"])
    family_cells = defaultdict(list)
    for key, values in leaf.items():
        split, method, budget, family, *_ = key
        family_cells[(split, method, budget, family)].append(float(np.mean(values)))

    split_summaries = []
    split_values = defaultdict(dict)
    for (split, method, budget, family), values in family_cells.items():
        split_values[(split, method, budget)][family] = float(np.mean(values))
    for (split, method, budget), family_scores in sorted(split_values.items()):
        split_summaries.append(
            {
                "split_seed": split,
                "method": method,
                "budget": budget,
                "family_macro_test_asr": float(np.mean(list(family_scores.values()))),
                "family_scores": family_scores,
            }
        )

    overall = []
    methods = sorted({(row["method"], row["budget"]) for row in all_records})
    for method, budget in methods:
        rows = [
            row for row in split_summaries
            if row["method"] == method and row["budget"] == budget
        ]
        overall.append(
            {
                "method": method,
                "budget": budget,
                "equal_split_family_macro_test_asr": float(
                    np.mean([row["family_macro_test_asr"] for row in rows])
                ),
                "split_scores": {
                    str(row["split_seed"]): row["family_macro_test_asr"] for row in rows
                },
            }
        )
    payload = {
        "protocol": {
            "item_split_seeds": sorted({row["split_seed"] for row in all_records}),
            "aggregation": "random repetitions -> model/dataset/training seeds -> family macro -> equal item-split mean",
            "directories": [str(path) for path in args.dirs],
        },
        "overall": overall,
        "by_split": split_summaries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for row in overall:
        print(
            row["method"], row["budget"],
            f'{row["equal_split_family_macro_test_asr"]:.6f}', row["split_scores"]
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
