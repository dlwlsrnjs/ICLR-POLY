#!/usr/bin/env python3
"""Family-clustered paired inference for confirmatory selector experiments."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_records(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def selector_records(directory: Path):
    records = []
    for path in sorted(directory.glob("selector_*_seed*.json")):
        records.extend(load_records(path))
    incumbent = directory / "incumbent_evaluation.json"
    if incumbent.exists():
        records.extend(load_records(incumbent))
    return records


def cell_means(records, method, budget):
    """Average repetitions, then models/datasets/seeds, within each family."""
    leaf = defaultdict(list)
    for row in records:
        if row["method"] != method or int(row["budget"] if "budget" in row else row["equivalent_full_arm_budget"]) != budget:
            continue
        key = (
            row.get("family", row.get("held_family")),
            row.get("model"),
            row.get("dataset", "joint"),
            row.get("split_seed", -1),
            row.get("seed", -1),
        )
        leaf[key].append(float(row["test_asr"]))
    family_split = defaultdict(list)
    for key, values in leaf.items():
        family_split[(key[0], key[3])].append(float(np.mean(values)))
    family = defaultdict(list)
    for (family_name, _split_seed), values in family_split.items():
        family[family_name].append(float(np.mean(values)))
    return {name: float(np.mean(values)) for name, values in family.items()}


def paired_test(records, treatment, control, budget, bootstrap, seed):
    treated = cell_means(records, treatment, budget)
    controlled = cell_means(records, control, budget)
    families = sorted(set(treated) & set(controlled))
    differences = np.asarray([treated[f] - controlled[f] for f in families])
    rng = np.random.default_rng(seed)
    samples = differences[
        rng.integers(0, len(differences), size=(bootstrap, len(differences)))
    ].mean(1)
    # Exact family-level sign-flip randomization test (2^7 for current panel).
    null = np.asarray(
        [
            np.mean(differences * np.asarray(signs))
            for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
        ]
    )
    observed = float(np.mean(differences))
    p_value = float((np.sum(np.abs(null) >= abs(observed) - 1e-15) + 1) / (len(null) + 1))
    return {
        "treatment": treatment,
        "control": control,
        "budget": budget,
        "family_macro_treatment": float(np.mean([treated[f] for f in families])),
        "family_macro_control": float(np.mean([controlled[f] for f in families])),
        "paired_difference": observed,
        "family_cluster_bootstrap_95ci": [
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)),
        ],
        "exact_family_signflip_p_two_sided": p_value,
        "families": families,
        "family_differences": dict(zip(families, differences.tolist())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--itemheldout", type=Path)
    parser.add_argument("--multifidelity", type=Path)
    parser.add_argument("--selector-dir", type=Path, nargs="+")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    args = parser.parse_args()

    analyses = []
    if args.itemheldout and args.itemheldout.exists():
        records = load_records(args.itemheldout)
        for budget in (2, 4, 8, 16):
            for treatment in (
                "advanced_nested_meanstd",
                "advanced_nested_cvar25",
                "advanced_nested_minimax",
                "advanced_nested_meanstd_posterior",
                "advanced_nested_cvar25_posterior",
                "advanced_nested_minimax_posterior",
            ):
                control = (
                    "structured_gp_crossfamily_posterior"
                    if treatment.endswith("_posterior")
                    else "structured_gp_crossfamily_incumbent"
                )
                analyses.append(
                    {
                        "experiment": "strict_itemheldout",
                        **paired_test(
                            records,
                            treatment,
                            control,
                            budget,
                            args.bootstrap,
                            1000 + budget,
                        ),
                    }
                )
    if args.multifidelity and args.multifidelity.exists():
        records = load_records(args.multifidelity)
        for budget in (2, 4, 8, 16):
            for treatment in (
                "multifidelity_graph_kernel_ucb",
                "prior_seeded_successive_halving_top8",
                "prior_seeded_successive_halving_top16",
                "prior_seeded_successive_halving_top32",
            ):
                analyses.append(
                    {
                        "experiment": "equal_item_cost_multifidelity",
                        **paired_test(
                            records,
                            treatment,
                            "full_fidelity_gp",
                            budget,
                            args.bootstrap,
                            2000 + budget,
                        ),
                    }
                )
    if args.selector_dir and all(path.exists() for path in args.selector_dir):
        records = []
        for directory in args.selector_dir:
            records.extend(selector_records(directory))
        for budget in (0, 2, 4, 8, 16):
            for treatment, control in (
                ("ppo_context", "supervised_context"),
                ("ppo_context", "ppo_no_context"),
                ("ppo_context", "ppo_no_feedback"),
                ("ppo_context", "random_probe_best_observed"),
                ("ppo_query_best_observed", "random_probe_best_observed"),
                (
                    "ppo_query_best_observed",
                    "ppo_no_feedback_query_best_observed",
                ),
            ):
                analyses.append(
                    {
                        "experiment": "gpu_familyloo_selector",
                        **paired_test(
                            records,
                            treatment,
                            control,
                            budget,
                            args.bootstrap,
                            3000 + budget,
                        ),
                    }
                )

    payload = {
        "protocol": {
            "unit_of_inference": "model family (7 clusters)",
            "point_estimate": "mean of paired family-level differences",
            "confidence_interval": f"{args.bootstrap} family-cluster bootstrap resamples",
            "hypothesis_test": "exact two-sided family-level sign-flip randomization test",
            "multiplicity": "raw p-values; contrasts are prespecified and no multiplicity correction is applied",
        },
        "comparisons": analyses,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Confirmatory family-clustered statistics",
        "",
        "The model family is the unit of inference (7 clusters). CIs are family-cluster bootstraps; p-values are exact two-sided sign-flip tests and are unadjusted.",
        "",
        "| Experiment | Treatment vs control | B | Δ family-macro ASR | 95% CI | p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in analyses:
        low, high = row["family_cluster_bootstrap_95ci"]
        lines.append(
            f'| {row["experiment"]} | {row["treatment"]} vs {row["control"]} | '
            f'{row["budget"]} | {row["paired_difference"]:+.4f} | '
            f'[{low:+.4f}, {high:+.4f}] | {row["exact_family_signflip_p_two_sided"]:.4f} |'
        )
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out_json} and {args.out_md} ({len(analyses)} comparisons)")


if __name__ == "__main__":
    main()
