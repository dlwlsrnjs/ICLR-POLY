#!/usr/bin/env python3
"""Evaluate frozen policies on the other dataset without re-tuning them."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve()
SPEC = importlib.util.spec_from_file_location("search", HERE.with_name("search_separate_axis_pipeline.py"))
S = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(S)


def main() -> None:
    experiment = HERE.parents[1]; suite = HERE.parents[2]; project = HERE.parents[4]
    clusters = experiment / "results/separate_axis_clusters"
    search = json.loads((experiment / "results/separate_axis_pipeline_search.json").read_text())
    policies = {dataset: row["chosen_policy"] for dataset, row in search["datasets"].items()}
    output = {"schema": "poly_cross_dataset_frozen_policy_check/v1", "results": {}}
    for dataset in ("mj", "lg"):
        cases, shared = S.prepare(project, experiment, suite, clusters, dataset)
        test = [case for case in cases if case["split"] == "test"]
        output["results"][dataset] = {}
        for source, policy in policies.items():
            output["results"][dataset][f"policy_selected_on_{source}"] = {
                "policy": policy,
                "summary": S.evaluate(test, shared, policy["method"], policy, 12),
            }
    path = experiment / "results/cross_dataset_policy_check.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
