#!/usr/bin/env python3
"""Materialize the frozen separate-axis policies and a concise result report."""
from __future__ import annotations

import json
from pathlib import Path


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def delta(value: float) -> str:
    return f"{100 * value:+.2f} pp"


def main() -> None:
    experiment = Path(__file__).resolve().parents[1]
    results = experiment / "results"
    search_path = results / "separate_axis_pipeline_search.json"
    search = json.loads(search_path.read_text())

    rows = []
    for dataset, payload in search["datasets"].items():
        selected = {
            "schema": "poly_frozen_separate_axis_policy/v1",
            "dataset": dataset,
            "selected_without_test_feedback": True,
            "selection_validation_test_items": payload["item_counts"],
            "candidate_count": payload["candidate_count"],
            "policy": payload["chosen_policy"],
            "heldout_test": payload["heldout_test"],
            "heldout_baselines": payload["heldout_baselines"],
            "axis_contract": {
                "understanding": "independent 32-setting GP/posterior",
                "willingness": "independent 5-frame GP/posterior",
                "joint_prior": False,
                "joint_posterior": False,
                "joint_embedding_or_centroid": False,
                "renderer_only_consumes_two_selected_values": True,
            },
            "source": search_path.name,
        }
        (results / f"selected_pipeline_{dataset}.json").write_text(
            json.dumps(selected, ensure_ascii=False, indent=2) + "\n"
        )

        ours = payload["heldout_test"]
        b1 = payload["heldout_baselines"]["B1_priorless_joint_gp_ucb_harmful"]
        b2 = payload["heldout_baselines"]["B2_priorless_separate_axis_harmful"]
        rows.append(
            {
                "dataset": dataset.upper(),
                "method": payload["chosen_policy"]["method"],
                "ours": ours,
                "b1": b1,
                "b2": b2,
            }
        )

    global_payload = search.get("global_policy")
    if global_payload:
        frozen_global = {
            "schema": "poly_frozen_global_separate_axis_policy/v1",
            "selected_without_test_feedback": True,
            "selection_objective": "equal MJ/LG macro",
            "policy": global_payload["chosen_policy"],
            "heldout_test_by_dataset": global_payload["heldout_test_by_dataset"],
            "axis_contract": {
                "understanding": "independent 32-setting posterior/GP",
                "willingness": "independent 5-frame posterior/GP",
                "joint_prior_or_posterior": False,
            },
            "source": search_path.name,
        }
        (results / "selected_pipeline_global.json").write_text(
            json.dumps(frozen_global, ensure_ascii=False, indent=2) + "\n"
        )

    lines = [
        "# Frozen separate-axis pipeline result",
        "",
        "This is an offline replay over the already generated MJ/LG 160-setting grids. "
        "Candidate policies were ranked on selection, shortlisted on validation, frozen, "
        "and evaluated once on item-disjoint held-out test splits.",
        "",
        "## Frozen policies",
        "",
        "| Dataset | Frozen policy | Test verified ASR | Oracle recovery | Budget AUC | Mean harmful pulls | Mean harmless probes |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        out = row["ours"]
        lines.append(
            f"| {row['dataset']} | `{row['method']}` | {pct(out['verified_asr'])} | "
            f"{pct(out['oracle_recovery'])} | {out['harmful_budget_auc']:.4f} | "
            f"{out['mean_harmful_pulls']:.2f} | {out['mean_harmless_requests']:.2f} |"
        )

    lines += [
        "",
        "## Dataset-specific-policy held-out comparisons",
        "",
        "B1 is the conventional prior-free joint 160-arm harmful-only GP-UCB baseline. "
        "B2 is the stronger prior-free harmful-only baseline that keeps the two axes separate.",
        "",
        "| Dataset | Comparator | Δ verified ASR | Δ budget AUC | Δ mean harmful pulls |",
        "|---|---|---:|---:|---:|",
    ]
    if global_payload:
        policy = global_payload["chosen_policy"]
        insertion = [
            "## One-policy deployment choice",
            "",
            f"Equal-dataset selection/validation chose `{policy['method']}` without test feedback. "
            "Its per-dataset held-out results are stored in `selected_pipeline_global.json`.",
            "",
            "| Dataset | Verified ASR | Oracle recovery | Budget AUC | Mean harmful pulls | Harmless probes |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for dataset, summary in global_payload["heldout_test_by_dataset"].items():
            insertion.append(
                f"| {dataset.upper()} | {pct(summary['verified_asr'])} | {pct(summary['oracle_recovery'])} | "
                f"{summary['harmful_budget_auc']:.4f} | {summary['mean_harmful_pulls']:.2f} | "
                f"{summary['mean_harmless_requests']:.2f} |"
            )
        insertion += [
            "",
            "| Dataset | Global policy vs comparator | Δ verified ASR | Δ budget AUC | Δ harmful pulls |",
            "|---|---|---:|---:|---:|",
        ]
        for dataset, summary in global_payload["heldout_test_by_dataset"].items():
            for key, label in (("B1_priorless_joint_gp_ucb_harmful", "B1 joint"),
                               ("B2_priorless_separate_axis_harmful", "B2 separate")):
                base = search["datasets"][dataset]["heldout_baselines"][key]
                insertion.append(
                    f"| {dataset.upper()} | {label} | {delta(summary['verified_asr'] - base['verified_asr'])} | "
                    f"{summary['harmful_budget_auc'] - base['harmful_budget_auc']:+.4f} | "
                    f"{summary['mean_harmful_pulls'] - base['mean_harmful_pulls']:+.2f} |"
                )
        insertion.append("")
        index = lines.index("## Dataset-specific-policy held-out comparisons")
        lines[index:index] = insertion
    for row in rows:
        ours = row["ours"]
        for label in ("b1", "b2"):
            base = row[label]
            lines.append(
                f"| {row['dataset']} | {label.upper()} | {delta(ours['verified_asr'] - base['verified_asr'])} | "
                f"{ours['harmful_budget_auc'] - base['harmful_budget_auc']:+.4f} | "
                f"{ours['mean_harmful_pulls'] - base['mean_harmful_pulls']:+.2f} |"
            )

    lines += [
        "",
        "## Interpretation",
        "",
        "- The selected objective is harmful-query efficiency, not test-set maximum final ASR. "
        "That is why B2 can finish slightly higher at budget 12 while the frozen method has better "
        "early-budget AUC and fewer harmful pulls.",
        *[
            f"- {row['dataset']} selected `{row['method']}`."
            for row in rows
        ],
        "- `P9_full_evidence_soft_knn`, when selected, uses all valid successes and failures, no hard "
        "cluster and no explicit OOD gate; `P7/P8` are calibrated OOD/soft-neighbour ablations.",
        "- Understanding and willingness embeddings, clusters, posteriors and acquisition functions "
        "remain separate. The renderer merely accepts one setting selected by each axis.",
        "",
        "## Limits",
        "",
        "- Exact WildGuard refusal state is available for 70.59% of MJ rows and 76.47% of LG rows. "
        "Rows without it retain the verified-success observation but cannot contribute an exact "
        "partial-versus-full-refusal transition.",
        "- The held-out split is small (MJ 16 items, LG 11 items, crossed with 17 models). Report "
        "confidence intervals and repeat on a fresh live set before a paper-level superiority claim.",
        "- Harmless probes are free only in the harmful-risk budget. They are still reported as real "
        "requests and must be included in token and wall-clock cost tables.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "/home/ljk98/POLY/.venv-gen/bin/python code/build_separate_axis_clusters.py",
        "/home/ljk98/POLY/.venv-gen/bin/python code/search_separate_axis_pipeline.py",
        "/home/ljk98/POLY/.venv-gen/bin/python code/run_separate_axis_ablation.py --seeds 20 --budget 12",
        "/home/ljk98/POLY/.venv-gen/bin/python code/audit_separate_axis_pipeline.py",
        "```",
        "",
    ]
    (results / "OPTIMAL_PIPELINE_REPORT.md").write_text("\n".join(lines))
    print(results / "OPTIMAL_PIPELINE_REPORT.md")


if __name__ == "__main__":
    main()
