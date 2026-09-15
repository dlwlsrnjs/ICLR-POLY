#!/usr/bin/env python3
"""Strict item-held-out confirmation of advanced additive BO.

For every target family and target model, arm queries observe only a calibration
item split.  The selected calibration incumbent is scored on disjoint test
items.  Source priors, graph kernels, and nested configuration selection use
other families only; target-family outcomes are unavailable until calibration.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base = load_module("poly_family_loo_item", "pilot_family_loo_asr_selector.py")
advanced = load_module("poly_advanced_item", "pilot_advanced_additive_bo.py")


def compact_configs():
    """Predeclared portfolio small enough for repeated strict split analysis."""
    keep = []
    for config in advanced.candidate_configs():
        if config.acquisition == "tree":
            keep.append(config)
        elif config.acquisition == "ei" and config.exploration == 0.01:
            if (config.dynamic_strength, config.local_after) in ((0.0, 8), (0.75, 999)):
                keep.append(config)
        elif config.acquisition == "ucb" and config.exploration == 1.2:
            if (config.dynamic_strength, config.local_after) in ((0.0, 8), (0.75, 999)):
                keep.append(config)
    return keep


def family_vectors(panel, arms, dataset, ids, excluded):
    grouped = defaultdict(list)
    for tag, datasets in panel.items():
        family = base.model_family(tag)
        if family in excluded:
            continue
        grouped[family].append(base.arm_means(datasets[dataset], ids, arms))
    if not grouped:
        raise RuntimeError(f"no source families after excluding {sorted(excluded)}")
    return np.stack([np.mean(grouped[family], axis=0) for family in sorted(grouped)])


def inner_family_scores(panel, arms, axes, train_ids, outer, dataset, config, budget):
    families = sorted({base.model_family(tag) for tag in panel} - {outer})
    scores = []
    for inner in families:
        sources = family_vectors(panel, arms, dataset, train_ids, {outer, inner})
        curves = []
        for tag in sorted(tag for tag in panel if base.model_family(tag) == inner):
            reward = base.arm_means(panel[tag][dataset], train_ids, arms)
            curve, *_ = advanced.search_curve(
                reward,
                sources,
                axes,
                config,
                budget,
                np.random.default_rng(
                    advanced.stable_seed("strict", outer, inner, dataset, config.name)
                ),
            )
            curves.append(0.75 * curve[-1] + 0.25 * float(np.mean(curve)))
        if curves:
            scores.append(float(np.mean(curves)))
    return np.asarray(scores, dtype=float)


def select_configs(panel, arms, axes, train_ids, outer, dataset, configs, budget):
    scored = []
    for config in configs:
        values = inner_family_scores(
            panel, arms, axes, train_ids, outer, dataset, config, budget
        )
        scored.append((config, values))
    objectives = {
        "advanced_nested_meanstd": lambda x: float(np.mean(x) - 0.5 * np.std(x)),
        "advanced_nested_cvar25": lambda x: float(
            np.mean(np.sort(x)[: max(1, int(np.ceil(0.25 * len(x))))])
        ),
        "advanced_nested_minimax": lambda x: float(np.min(x)),
    }
    selected = {}
    audit = {}
    for method, objective in objectives.items():
        ranking = sorted(
            ((objective(values), config.name, config, values) for config, values in scored),
            key=lambda row: (-row[0], row[1]),
        )
        score, _, config, values = ranking[0]
        selected[method] = config
        audit[method] = {
            "config": asdict(config),
            "objective": score,
            "inner_family_values": values.tolist(),
            "runner_up": [
                {"config": row[1], "objective": row[0]} for row in ranking[1:4]
            ],
        }
    return selected, audit


def incumbents(queried, calibration, budgets):
    answer = {}
    for budget in budgets:
        prefix = np.asarray(queried[:budget], dtype=int)
        answer[budget] = int(prefix[np.argmax(calibration[prefix])])
    return answer


def advanced_posterior_recommendations(
    queried, calibration, sources, axes, config, budgets
):
    kernel, _ = advanced.build_kernel(
        sources, axes, config.beta, config.interaction, config.task_mix
    )
    signal = max(0.08, float(np.std(sources)))
    answer = {}
    for budget in budgets:
        q = queried[:budget]
        observed = [float(calibration[index]) for index in q]
        prior, _ = advanced.transfer_prior(
            sources, q, observed, config.temperature, config.dynamic_strength
        )
        mean, _ = advanced.posterior(
            prior, kernel, q, observed, signal=signal, noise=config.noise
        )
        answer[budget] = int(np.argmax(mean))
    return answer


def summarize(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["method"], row["budget"])].append(row)
    output = []
    for (method, budget), rows in sorted(grouped.items()):
        family_scores = {}
        for family in sorted({row["family"] for row in rows}):
            family_scores[family] = float(
                np.mean([row["test_asr"] for row in rows if row["family"] == family])
            )
        output.append(
            {
                "method": method,
                "budget": budget,
                "family_macro_test_asr": float(np.mean(list(family_scores.values()))),
                "family_scores": family_scores,
                "n_records": len(rows),
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--random-repetitions", type=int, default=32)
    args = parser.parse_args()

    panel, arms = base.load_complete_panel(args.root)
    axes = advanced.arm_axes(base, arms)
    basic_kernel = base.structured_kernel(arms)
    configs = compact_configs()
    families = sorted({base.model_family(tag) for tag in panel})
    max_budget = max(args.budgets)
    records, selection_audit = [], {}

    for split_seed in args.split_seeds:
        for dataset, expected_rows in base.EXPECTED_ROWS.items():
            exemplar = next(iter(panel.values()))[dataset][arms[0]]
            train_ids, calibration_ids, test_ids = base.split_items(
                list(exemplar), split_seed + (0 if dataset == "mj" else 100_000)
            )
            assert len(train_ids) + len(calibration_ids) + len(test_ids) == expected_rows
            for outer in families:
                selected, audit = select_configs(
                    panel, arms, axes, train_ids, outer, dataset, configs, max_budget
                )
                selection_audit[f"{split_seed}:{dataset}:{outer}"] = audit
                sources = family_vectors(panel, arms, dataset, train_ids, {outer})
                prior = sources.mean(axis=0)
                for tag in sorted(tag for tag in panel if base.model_family(tag) == outer):
                    calibration = base.arm_means(
                        panel[tag][dataset], calibration_ids, arms
                    )
                    test = base.arm_means(panel[tag][dataset], test_ids, arms)
                    common = {
                        "split_seed": split_seed,
                        "dataset": dataset,
                        "family": outer,
                        "model": tag,
                    }
                    oracle = int(np.argmax(test))
                    records.append(
                        {
                            **common,
                            "method": "oracle_test_upper_bound",
                            "budget": -1,
                            "arm": arms[oracle],
                            "test_asr": float(test[oracle]),
                        }
                    )
                    fixed = int(np.argmax(prior))
                    for budget in args.budgets:
                        records.append(
                            {
                                **common,
                                "method": "best_fixed_other_families",
                                "budget": budget,
                                "arm": arms[fixed],
                                "test_asr": float(test[fixed]),
                            }
                        )

                    old_index, old_queried = base.gp_select(
                        calibration,
                        prior,
                        basic_kernel,
                        max_budget,
                        np.random.default_rng(
                            advanced.stable_seed("old", split_seed, dataset, tag)
                        ),
                    )
                    del old_index
                    for budget, index in incumbents(
                        old_queried, calibration, args.budgets
                    ).items():
                        records.append(
                            {
                                **common,
                                "method": "structured_gp_crossfamily_incumbent",
                                "budget": budget,
                                "arm": arms[index],
                                "test_asr": float(test[index]),
                            }
                        )
                    for budget in args.budgets:
                        q = old_queried[:budget]
                        observed = [float(calibration[index]) for index in q]
                        mean, _ = base.gp_posterior(
                            prior, basic_kernel, q, observed, signal=0.22, noise=0.025
                        )
                        index = int(np.argmax(mean))
                        records.append(
                            {
                                **common,
                                "method": "structured_gp_crossfamily_posterior",
                                "budget": budget,
                                "arm": arms[index],
                                "test_asr": float(test[index]),
                            }
                        )

                    for method, config in selected.items():
                        _, queried, relevance, _ = advanced.search_curve(
                            calibration,
                            sources,
                            axes,
                            config,
                            max_budget,
                            np.random.default_rng(
                                advanced.stable_seed(
                                    "advanced", split_seed, dataset, tag, method
                                )
                            ),
                        )
                        for budget, index in incumbents(
                            queried, calibration, args.budgets
                        ).items():
                            records.append(
                                {
                                    **common,
                                    "method": method,
                                    "budget": budget,
                                    "arm": arms[index],
                                    "test_asr": float(test[index]),
                                    "selected_config": config.name,
                                    "axis_relevance": relevance,
                                }
                            )
                        for budget, index in advanced_posterior_recommendations(
                            queried, calibration, sources, axes, config, args.budgets
                        ).items():
                            records.append(
                                {
                                    **common,
                                    "method": f"{method}_posterior",
                                    "budget": budget,
                                    "arm": arms[index],
                                    "test_asr": float(test[index]),
                                    "selected_config": config.name,
                                    "axis_relevance": relevance,
                                }
                            )

                    for repetition in range(args.random_repetitions):
                        rng = np.random.default_rng(
                            advanced.stable_seed(
                                "random", split_seed, dataset, tag, repetition
                            )
                        )
                        order = rng.choice(len(arms), size=max_budget, replace=False)
                        for budget, index in incumbents(
                            order.tolist(), calibration, args.budgets
                        ).items():
                            records.append(
                                {
                                    **common,
                                    "method": "random_probe_best_observed",
                                    "budget": budget,
                                    "repetition": repetition,
                                    "arm": arms[index],
                                    "test_asr": float(test[index]),
                                }
                            )

        print(f"completed split seed {split_seed}", flush=True)
        partial = {
            "protocol": {"status": "partial", "completed_through_seed": split_seed},
            "summary": summarize(records),
            "records": records,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.with_suffix(args.out.suffix + ".partial").write_text(
            json.dumps(partial, indent=2), encoding="utf-8"
        )

    payload = {
        "protocol": {
            "metric": "Qwen3Guard unsafe ASR (U)",
            "outer_split": "leave-one-model-family-out",
            "item_split": "50% source training / 25% target calibration / 25% target test",
            "selection": "best calibration incumbent, scored on disjoint test items",
            "target_test_used_for_selection": False,
            "target_family_used_for_prior_kernel_or_tuning": False,
            "source_information_restricted_to_train_item_ids": True,
            "models": sorted(panel),
            "families": families,
            "datasets": sorted(base.EXPECTED_ROWS),
            "arms": len(arms),
            "split_seeds": args.split_seeds,
            "budgets": args.budgets,
            "candidate_configs": len(configs),
            "random_repetitions": args.random_repetitions,
        },
        "selection_audit": selection_audit,
        "summary": summarize(records),
        "records": records,
    }
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(args.out)
    args.out.with_suffix(args.out.suffix + ".partial").unlink(missing_ok=True)
    print("method,budget,family_macro_test_asr")
    for row in payload["summary"]:
        print(
            f'{row["method"]},{row["budget"]},{row["family_macro_test_asr"]:.6f}'
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
