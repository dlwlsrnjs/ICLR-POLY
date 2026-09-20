#!/usr/bin/env python3
"""Count usable success/failure/missing evidence for the two-axis prior."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve()
SPEC = importlib.util.spec_from_file_location("axis", HERE.with_name("run_separate_axis_ablation.py"))
A = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(A)


def main() -> None:
    project = HERE.parents[4]; experiment = HERE.parents[1]
    rows = {}
    totals = {
        "understanding": {"rows": 0, "valid_success": 0, "valid_failure": 0, "invalid": 0},
        "willingness": {"rows": 0, "eligible_r1": 0, "valid_success": 0, "valid_failure": 0,
                         "reconstruction_failure": 0, "invalid_or_missing_judge": 0},
    }
    for model in A.MODELS if hasattr(A, "MODELS") else (
        "falcon3_10b", "falcon3_3b", "falcon3_7b", "gemma2_27b", "gemma2_2b_it",
        "gemma2_9b_it", "glm4_9b", "llama31_8b_it", "llama32_3b_it", "mistral24b",
        "mistral7b", "phi35_mini", "phi3_medium_14b", "qwen25_14b", "qwen25_32b",
        "qwen25_3b", "qwen25_7b"):
        data = A.load_harmless(project, model)
        model_counts = {}
        for axis in ("understanding", "willingness"):
            count = {key: 0 for key in totals[axis]}
            for outcome in data[axis].values():
                count["rows"] += 1
                if axis == "understanding":
                    if not outcome["R_valid"]:
                        count["invalid"] += 1
                    elif outcome["R"]:
                        count["valid_success"] += 1
                    else:
                        count["valid_failure"] += 1
                elif not outcome["R_valid"] or (outcome["R"] and not outcome["full_valid"]):
                    count["invalid_or_missing_judge"] += 1
                elif not outcome["R"]:
                    count["reconstruction_failure"] += 1
                else:
                    count["eligible_r1"] += 1
                    count["valid_success" if outcome["full"] else "valid_failure"] += 1
            model_counts[axis] = count
            for key, value in count.items(): totals[axis][key] += value
        rows[model] = model_counts
    payload = {
        "schema": "poly_full_evidence_prior_audit/v1",
        "rules": {
            "understanding": "valid R success and failure retained; invalid missing",
            "willingness": "only valid R=1 cohort; valid fulfillment success and failure retained; R=0 missing for this axis"
        },
        "totals": totals,
        "models": rows,
    }
    output = experiment / "results/full_evidence_prior_audit.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(totals))


if __name__ == "__main__":
    main()
