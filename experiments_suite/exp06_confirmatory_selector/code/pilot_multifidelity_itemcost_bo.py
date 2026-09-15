#!/usr/bin/env python3
"""Cost-normalized multi-fidelity arm search on held-out target items.

One unit of cost is one target-model response judged for one benchmark item.
Full-fidelity methods spend all calibration items per arm; multi-fidelity methods
may allocate small batches and revisit promising arms.  Recommendations are
evaluated only on disjoint test items.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base = load("poly_mf_base", "pilot_family_loo_asr_selector.py")
advanced = load("poly_mf_advanced", "pilot_advanced_additive_bo.py")


def source_vectors(panel, arms, dataset, train_ids, held_family):
    grouped = defaultdict(list)
    for tag, datasets in panel.items():
        family = base.model_family(tag)
        if family == held_family:
            continue
        grouped[family].append(base.arm_means(datasets[dataset], train_ids, arms))
    return np.stack([np.mean(grouped[f], axis=0) for f in sorted(grouped)])


def posterior(prior, kernel, queried, observed, noises, signal):
    covariance = signal**2 * kernel
    if not queried:
        return prior.copy(), np.diag(covariance).copy()
    q = np.asarray(queried, dtype=int)
    system = covariance[np.ix_(q, q)] + np.diag(np.asarray(noises))
    cross = covariance[:, q]
    inverse = np.linalg.pinv(system)
    mean = prior + cross @ inverse @ (np.asarray(observed) - prior[q])
    variance = np.diag(covariance) - np.einsum("ij,jk,ik->i", cross, inverse, cross)
    return mean, np.clip(variance, 1e-10, None)


def multifidelity_gp(rows, item_ids, prior, kernel, checkpoints, rng, batch=2):
    n_arms = len(prior)
    permutations = [rng.permutation(item_ids).tolist() for _ in range(n_arms)]
    positions = np.zeros(n_arms, dtype=int)
    sums = np.zeros(n_arms, dtype=float)
    counts = np.zeros(n_arms, dtype=float)
    signal = max(0.08, float(np.std(prior)))
    recommendations = {}
    maximum = max(checkpoints)
    spent = 0
    while spent < maximum:
        active = counts > 0
        residual = np.zeros(n_arms, dtype=float)
        residual[active] = sums[active] / counts[active] - prior[active]
        effective = np.abs(kernel) @ counts
        correction = kernel @ (counts * residual) / (effective + 4.0)
        mean = prior + correction
        uncertainty = signal / np.sqrt(1.0 + counts + 0.05 * effective)
        acquisition = mean + 1.2 * uncertainty
        exhausted = positions >= len(item_ids)
        acquisition[exhausted] = -np.inf
        index = int(np.argmax(acquisition + rng.normal(0, 1e-10, n_arms)))
        take = min(batch, len(item_ids) - positions[index], maximum - spent)
        ids = permutations[index][positions[index] : positions[index] + take]
        value = float(np.mean([rows[index][item_id] for item_id in ids]))
        positions[index] += take
        spent += take
        sums[index] += value * take
        counts[index] += take
        if spent in checkpoints:
            active = counts > 0
            residual = np.zeros(n_arms, dtype=float)
            residual[active] = sums[active] / counts[active] - prior[active]
            # Conservative direct shrinkage avoids selecting an unobserved arm
            # solely because of a noisy cross-arm extrapolation.
            rec_mean = (4.0 * prior + sums) / (4.0 + counts)
            recommendations[spent] = int(np.argmax(rec_mean))
    return recommendations


def full_fidelity_gp(calibration, prior, kernel, budgets, rng):
    _, queried = base.gp_select(
        calibration, prior, kernel, max(budgets), rng, beta=1.25, signal=0.22, noise=0.025
    )
    recommendations = {}
    for budget in budgets:
        prefix = np.asarray(queried[:budget], dtype=int)
        recommendations[budget] = int(prefix[np.argmax(calibration[prefix])])
    return recommendations


def successive_halving(rows, item_ids, prior, checkpoints, rng, batch=2, topk=32):
    """Prior-seeded racing with cumulative empirical means and equal item cost."""
    maximum = max(checkpoints)
    candidates = np.argsort(-prior, kind="stable")[: min(topk, len(prior))].tolist()
    permutations = {arm: rng.permutation(item_ids).tolist() for arm in candidates}
    sums = defaultdict(float)
    counts = defaultdict(int)
    spent = 0
    recommendations = {}
    cursor = 0
    round_target = batch
    while spent < maximum:
        if not candidates:
            break
        arm = candidates[cursor % len(candidates)]
        remaining = len(item_ids) - counts[arm]
        if remaining <= 0:
            cursor += 1
            if all(counts[a] >= len(item_ids) for a in candidates):
                break
            continue
        take = min(batch, remaining, maximum - spent)
        ids = permutations[arm][counts[arm] : counts[arm] + take]
        sums[arm] += sum(rows[arm][item_id] for item_id in ids)
        counts[arm] += take
        spent += take
        cursor += 1
        if spent in checkpoints:
            recommendations[spent] = max(
                candidates,
                key=lambda a: (sums[a] / counts[a] if counts[a] else prior[a], prior[a]),
            )
        if cursor % len(candidates) == 0 and min(counts[a] for a in candidates) >= round_target:
            candidates.sort(
                key=lambda a: (sums[a] / max(counts[a], 1), prior[a]), reverse=True
            )
            if len(candidates) > 4:
                candidates = candidates[: max(4, len(candidates) // 2)]
            cursor = 0
            round_target = min(len(item_ids), round_target * 2)
    # Once all surviving arms reach full calibration fidelity, additional calls
    # cannot change the racing incumbent.  Carry it forward to later budgets.
    if candidates:
        final = max(
            candidates,
            key=lambda a: (sums[a] / counts[a] if counts[a] else prior[a], prior[a]),
        )
        for checkpoint in checkpoints:
            recommendations.setdefault(checkpoint, final)
    return recommendations


def summarize(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["method"], row["equivalent_full_arm_budget"])].append(row)
    summary = []
    for (method, budget), rows in sorted(grouped.items()):
        family_scores = {
            family: float(np.mean([r["test_asr"] for r in rows if r["family"] == family]))
            for family in sorted({r["family"] for r in rows})
        }
        summary.append(
            {
                "method": method,
                "equivalent_full_arm_budget": budget,
                "family_macro_test_asr": float(np.mean(list(family_scores.values()))),
                "family_scores": family_scores,
                "n_records": len(rows),
            }
        )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--repetitions", type=int, default=32)
    args = parser.parse_args()

    panel, arms = base.load_complete_panel(args.root)
    axes = advanced.arm_axes(base, arms)
    families = sorted({base.model_family(tag) for tag in panel})
    records = []
    for split_seed in args.split_seeds:
        for dataset in sorted(base.EXPECTED_ROWS):
            exemplar = next(iter(panel.values()))[dataset][arms[0]]
            train_ids, calibration_ids, test_ids = base.split_items(
                list(exemplar), split_seed + (100_000 if dataset == "lg" else 0)
            )
            item_costs = [budget * len(calibration_ids) for budget in args.budgets]
            for family in families:
                sources = source_vectors(panel, arms, dataset, train_ids, family)
                prior = sources.mean(axis=0)
                kernel, relevance = advanced.build_kernel(
                    sources, axes, beta=0.7, interaction=0.25, task_mix=0.35
                )
                for tag in sorted(t for t in panel if base.model_family(t) == family):
                    calibration = base.arm_means(
                        panel[tag][dataset], calibration_ids, arms
                    )
                    test = base.arm_means(panel[tag][dataset], test_ids, arms)
                    arm_rows = [panel[tag][dataset][arm] for arm in arms]
                    common = {
                        "split_seed": split_seed,
                        "dataset": dataset,
                        "family": family,
                        "model": tag,
                    }
                    for budget in args.budgets:
                        fixed = int(np.argmax(prior))
                        records.append(
                            {
                                **common,
                                "method": "best_fixed_other_families",
                                "equivalent_full_arm_budget": budget,
                                "item_call_cost": budget * len(calibration_ids),
                                "arm": arms[fixed],
                                "test_asr": float(test[fixed]),
                            }
                        )
                    full = full_fidelity_gp(
                        calibration,
                        prior,
                        kernel,
                        args.budgets,
                        np.random.default_rng(advanced.stable_seed("full", split_seed, tag, dataset)),
                    )
                    for budget, index in full.items():
                        records.append(
                            {
                                **common,
                                "method": "full_fidelity_gp",
                                "equivalent_full_arm_budget": budget,
                                "item_call_cost": budget * len(calibration_ids),
                                "arm": arms[index],
                                "test_asr": float(test[index]),
                            }
                        )
                    for repetition in range(args.repetitions):
                        seed = advanced.stable_seed(
                            "mf", split_seed, dataset, tag, repetition
                        )
                        mf = multifidelity_gp(
                            arm_rows,
                            calibration_ids,
                            prior,
                            kernel,
                            item_costs,
                            np.random.default_rng(seed),
                        )
                        halvings = {
                            topk: successive_halving(
                                arm_rows,
                                calibration_ids,
                                prior,
                                item_costs,
                                np.random.default_rng(seed + 17 + topk),
                                topk=topk,
                            )
                            for topk in (8, 16, 32)
                        }
                        random_order = np.random.default_rng(seed + 31).choice(
                            len(arms), size=max(args.budgets), replace=False
                        )
                        for budget, cost in zip(args.budgets, item_costs):
                            random_prefix = random_order[:budget]
                            random_index = int(
                                random_prefix[np.argmax(calibration[random_prefix])]
                            )
                            candidates = [
                                ("multifidelity_graph_kernel_ucb", mf[cost]),
                                ("random_full_fidelity", random_index),
                            ] + [
                                (f"prior_seeded_successive_halving_top{topk}", result[cost])
                                for topk, result in halvings.items()
                            ]
                            for method, index in candidates:
                                records.append(
                                    {
                                        **common,
                                        "method": method,
                                        "equivalent_full_arm_budget": budget,
                                        "item_call_cost": cost,
                                        "repetition": repetition,
                                        "arm": arms[index],
                                        "test_asr": float(test[index]),
                                        "axis_relevance": relevance.tolist(),
                                    }
                                )
        print(f"completed split seed {split_seed}", flush=True)

    payload = {
        "protocol": {
            "metric": "Qwen3Guard unsafe ASR (U)",
            "cost_unit": "one target-model generation judged on one calibration item",
            "comparison": "equal total item-call cost",
            "multifidelity_surrogate": "graph-kernel UCB with per-arm cumulative sufficient statistics",
            "outer_split": "leave-one-model-family-out",
            "item_split": "50% source train / 25% target calibration / 25% target test",
            "target_test_used_for_selection": False,
            "models": sorted(panel),
            "families": families,
            "arms": len(arms),
            "split_seeds": args.split_seeds,
            "budgets": args.budgets,
            "repetitions": args.repetitions,
        },
        "summary": summarize(records),
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(args.out)
    print("method,budget,family_macro_test_asr")
    for row in payload["summary"]:
        print(
            f'{row["method"]},{row["equivalent_full_arm_budget"]},'
            f'{row["family_macro_test_asr"]:.6f}'
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
