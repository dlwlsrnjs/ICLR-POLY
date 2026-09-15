#!/usr/bin/env python3
"""Nested-LOFO advanced additive/transfer GP-BO on the finite POLY arm grid.

The target family is never used to construct the prior, kernel, or choose a
hyperparameter configuration.  The surrogate combines (i) axis-wise graph
diffusion kernels, (ii) a second-order ANOVA additive kernel, and optionally
(iii) a regularized source-task covariance kernel.  Online source weights are
updated only from arms already queried on the target (an RGPE-style transfer
mechanism).  The final score is the best observed full-benchmark ASR, matching
the conventional replay protocol used by the existing selector comparison.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.special import ndtr


def load_conventional():
    path = Path(__file__).with_name("pilot_conventional_asr_selector.py")
    spec = importlib.util.spec_from_file_location("conventional_asr", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    kernel = 0.5 * (kernel + kernel.T)
    diagonal = np.sqrt(np.clip(np.diag(kernel), 1e-12, None))
    return kernel / np.outer(diagonal, diagonal)


def diffusion(level_count: int, beta: float, edges) -> np.ndarray:
    adjacency = np.zeros((level_count, level_count), dtype=float)
    for left, right in edges:
        adjacency[left, right] = adjacency[right, left] = 1.0
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    return normalize_kernel(expm(-beta * laplacian))


def arm_axes(base, arms):
    comp_levels = (3, 5, 8, 12)
    order_levels = ("ordered", "shuffled")
    n_levels = (2, 4, 6, 8)
    suffix_levels = ("plain", "persona", "fiction", "pap", "persona+fiction")
    rows = []
    for arm in arms:
        match = base.ARM_RE.match(arm)
        if match is None:
            raise ValueError(f"not a factorial arm: {arm}")
        rows.append((
            comp_levels.index(int(match.group("comp"))),
            order_levels.index(match.group("order")),
            n_levels.index(int(match.group("n"))),
            suffix_levels.index(match.group("suffix")),
        ))
    return np.asarray(rows, dtype=int)


def axis_kernels(axes: np.ndarray, beta: float) -> list[np.ndarray]:
    level_kernels = [
        diffusion(4, beta, [(0, 1), (1, 2), (2, 3)]),
        diffusion(2, beta, [(0, 1)]),
        diffusion(4, beta, [(0, 1), (1, 2), (2, 3)]),
        # The compound persona+fiction frame is adjacent to both constituents.
        diffusion(5, beta, [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4)]),
    ]
    return [kernel[index[:, None], index[None, :]] for kernel, index in zip(level_kernels, axes.T)]


def relevance_weights(source_values: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Source-only functional-ANOVA relevance estimate, regularized from zero."""
    scores = np.zeros(axes.shape[1], dtype=float)
    for values in source_values:
        total = float(np.var(values)) + 1e-9
        for axis in range(axes.shape[1]):
            fitted = np.zeros_like(values)
            for level in np.unique(axes[:, axis]):
                mask = axes[:, axis] == level
                fitted[mask] = float(np.mean(values[mask]))
            scores[axis] += float(np.var(fitted)) / total
    scores /= max(len(source_values), 1)
    scores += 0.05
    return scores / scores.sum()


def source_family_vectors(base, panel, target_family, dataset, arms):
    grouped = defaultdict(list)
    for tag, datasets in panel.items():
        family = base.model_family(tag)
        if family == target_family:
            continue
        grouped[family].append(np.asarray([datasets[dataset][arm] for arm in arms], dtype=float))
    return np.stack([np.mean(grouped[family], axis=0) for family in sorted(grouped)])


def build_kernel(source_values, axes, beta, interaction, task_mix):
    components = axis_kernels(axes, beta)
    weights = relevance_weights(source_values, axes)
    main = sum(weight * kernel for weight, kernel in zip(weights, components))
    pairs, pair_weights = [], []
    for left, right in itertools.combinations(range(len(components)), 2):
        pairs.append(components[left] * components[right])
        pair_weights.append(float(np.sqrt(weights[left] * weights[right])))
    pair_weights = np.asarray(pair_weights)
    pair_weights /= pair_weights.sum()
    anova = sum(weight * kernel for weight, kernel in zip(pair_weights, pairs))
    structural = normalize_kernel((1.0 - interaction) * main + interaction * anova)
    if task_mix <= 0 or len(source_values) < 2:
        return structural, weights
    centered = source_values - source_values.mean(axis=0, keepdims=True)
    task = centered.T @ centered / max(len(source_values) - 1, 1)
    task += 0.05 * float(np.mean(np.diag(task))) * np.eye(len(task))
    task = normalize_kernel(task)
    return normalize_kernel((1.0 - task_mix) * structural + task_mix * task), weights


def family_balanced_prior(source_values: np.ndarray) -> np.ndarray:
    return source_values.mean(axis=0)


def transfer_prior(source_values, queried, observed, temperature, dynamic_strength):
    global_prior = family_balanced_prior(source_values)
    if not queried or dynamic_strength <= 0:
        return global_prior, np.full(len(source_values), 1.0 / len(source_values))
    q = np.asarray(queried, dtype=int)
    y = np.asarray(observed, dtype=float)
    predictions = source_values[:, q]
    # Offset-invariant matching after two observations; the one-point case still
    # supplies useful level information but is strongly shrunk to the global mean.
    if len(q) >= 2:
        predictions = predictions - predictions.mean(axis=1, keepdims=True)
        y = y - y.mean()
    loss = np.mean((predictions - y[None, :]) ** 2, axis=1)
    logits = -(loss - loss.min()) / max(temperature, 1e-8)
    weights = np.exp(np.clip(logits, -50, 0))
    weights /= weights.sum()
    adaptive = weights @ source_values
    ramp = dynamic_strength * min(1.0, len(q) / 4.0)
    return (1.0 - ramp) * global_prior + ramp * adaptive, weights


def posterior(prior, kernel, queried, observed, signal, noise):
    covariance = signal**2 * kernel
    if not queried:
        return prior.copy(), covariance.copy()
    q = np.asarray(queried, dtype=int)
    kqq = covariance[np.ix_(q, q)] + noise * np.eye(len(q))
    cross = covariance[:, q]
    try:
        alpha = np.linalg.solve(kqq, np.asarray(observed) - prior[q])
        solved = np.linalg.solve(kqq, cross.T)
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(kqq)
        alpha = inverse @ (np.asarray(observed) - prior[q])
        solved = inverse @ cross.T
    mean = prior + cross @ alpha
    covariance_post = covariance - cross @ solved
    return mean, 0.5 * (covariance_post + covariance_post.T)


@dataclass(frozen=True)
class Config:
    name: str
    beta: float
    interaction: float
    task_mix: float
    acquisition: str
    exploration: float
    noise: float
    dynamic_strength: float
    temperature: float = 0.01
    local_after: int = 999
    local_topk: int = 3


def acquisition_values(kind, mean, covariance, observed, exploration, rng):
    sigma = np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
    if kind == "ucb":
        return mean + exploration * sigma
    if kind == "ei":
        incumbent = max(observed) if observed else float(np.max(mean))
        improvement = mean - incumbent - exploration
        z = improvement / sigma
        return improvement * ndtr(z) + sigma * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    if kind == "thompson":
        jitter = 1e-9 * np.eye(len(mean))
        return rng.multivariate_normal(mean, covariance + jitter, method="eigh")
    raise ValueError(kind)


def search_curve(reward, source_values, axes, config, budget, rng):
    kernel, relevance = build_kernel(
        source_values, axes, config.beta, config.interaction, config.task_mix
    )
    queried, observed, curve, weight_trace = [], [], [], []
    signal = max(0.08, float(np.std(source_values)))
    for step in range(budget):
        prior, source_weights = transfer_prior(
            source_values, queried, observed, config.temperature, config.dynamic_strength
        )
        mean, covariance = posterior(
            prior, kernel, queried, observed, signal=signal, noise=config.noise
        )
        if step == 0:
            score = mean
        elif config.acquisition == "tree":
            width = max(1, int(round(config.exploration)))
            top_positions = np.argsort(-np.asarray(observed), kind="stable")[:width]
            score = np.full(len(reward), -np.inf)
            seen_suffixes = set(axes[np.asarray(queried), 3].tolist())
            for rank, position in enumerate(top_positions):
                parent = queried[int(position)]
                neighbors = np.sum(axes != axes[parent], axis=1) == 1
                for candidate in np.flatnonzero(neighbors):
                    if candidate in queried:
                        continue
                    priority = observed[int(position)] - 0.01 * rank
                    priority += 0.03 * int(int(axes[candidate, 3]) not in seen_suffixes)
                    score[candidate] = max(score[candidate], priority)
        else:
            score = acquisition_values(
                config.acquisition, mean, covariance, observed, config.exploration, rng
            )
        if queried:
            score[np.asarray(queried)] = -np.inf
        # Trust-region refinement on the Cartesian-product graph.  The global
        # additive GP supplies the warm phase; late queries refine one-axis
        # mutations around the best observed incumbents.
        if step >= config.local_after and queried:
            top_positions = np.argsort(-np.asarray(observed), kind="stable")[: config.local_topk]
            allowed = np.zeros(len(score), dtype=bool)
            for position in top_positions:
                parent = queried[int(position)]
                allowed |= np.sum(axes != axes[parent], axis=1) == 1
            allowed[np.asarray(queried)] = False
            if np.any(allowed):
                score[~allowed] = -np.inf
        score += rng.normal(0.0, 1e-10, len(score))
        index = int(np.argmax(score))
        queried.append(index)
        observed.append(float(reward[index]))
        curve.append(float(np.max(observed)))
        weight_trace.append(source_weights.tolist())
    return curve, queried, relevance.tolist(), weight_trace


def candidate_configs():
    configs = []
    # A compact, predeclared portfolio.  Inner LOFO chooses among these only.
    for beta, interaction, task_mix in (
        (0.35, 0.00, 0.00),
        (0.70, 0.25, 0.00),
        (1.20, 0.50, 0.00),
        (0.70, 0.25, 0.35),
        (0.70, 0.50, 0.60),
        (1.20, 0.25, 0.60),
    ):
        stem = f"diff_b{beta:g}_a{interaction:g}_t{task_mix:g}"
        for acquisition, exploration in (("ei", 0.0), ("ei", 0.01), ("ucb", 0.6), ("ucb", 1.2)):
            for dynamic in (0.0, 0.75):
                for local_after in (999, 8):
                    local_name = "global" if local_after == 999 else "local8"
                    configs.append(Config(
                        name=f"{stem}_{acquisition}{exploration:g}_rgpe{dynamic:g}_{local_name}",
                        beta=beta, interaction=interaction, task_mix=task_mix,
                        acquisition=acquisition, exploration=exploration,
                        noise=0.0025, dynamic_strength=dynamic, local_after=local_after,
                    ))
    # Discrete trust-region controls are included in the same nested portfolio;
    # they test whether a learned global GP should hand off to beam refinement.
    for width in (1, 3, 5):
        configs.append(Config(
            name=f"crossfamily_tree_w{width}", beta=0.7, interaction=0.25,
            task_mix=0.0, acquisition="tree", exploration=float(width),
            noise=0.0025, dynamic_strength=0.0,
        ))
    return configs


def stable_seed(*parts) -> int:
    return sum((i + 1) * sum(ord(char) for char in str(part)) for i, part in enumerate(parts)) * 1009


def inner_score(
    conventional, base, panel, arms, axes, outer_family, dataset, config, budget,
    robustness_penalty=0.0, selection_mode="meanstd",
):
    families = sorted({base.model_family(tag) for tag in panel if base.model_family(tag) != outer_family})
    scores = []
    for inner_family in families:
        # Remove the outer family explicitly: it is unavailable at every nested stage.
        grouped = defaultdict(list)
        for tag, datasets in panel.items():
            family = base.model_family(tag)
            if family in (outer_family, inner_family):
                continue
            grouped[family].append(np.asarray([datasets[dataset][arm] for arm in arms]))
        if len(grouped) < 2:
            continue
        sources = np.stack([np.mean(grouped[f], axis=0) for f in sorted(grouped)])
        family_values = []
        for tag in sorted(t for t in panel if base.model_family(t) == inner_family):
            reward = np.asarray([panel[tag][dataset][arm] for arm in arms])
            curve, *_ = search_curve(
                reward, sources, axes, config, budget,
                np.random.default_rng(stable_seed(outer_family, inner_family, dataset, config.name)),
            )
            # The primary objective here is simple regret at the final budget;
            # retain a smaller AUC term so ties do not sacrifice early utility.
            family_values.append(0.75 * curve[-1] + 0.25 * float(np.mean(curve)))
        if family_values:
            scores.append(float(np.mean(family_values)))
    if not scores:
        return float("-inf")
    if selection_mode == "minimax":
        return float(np.min(scores))
    if selection_mode == "cvar25":
        tail = max(1, int(np.ceil(0.25 * len(scores))))
        return float(np.mean(np.sort(scores)[:tail]))
    return float(np.mean(scores) - robustness_penalty * np.std(scores))


def summarize(cells, method, budget):
    families = sorted({cell["family"] for cell in cells})
    family_curves = {
        family: np.mean([cell["curve"] for cell in cells if cell["family"] == family], axis=0)
        for family in families
    }
    curve = np.mean(list(family_curves.values()), axis=0)
    oracle = float(np.mean([
        np.mean([cell["oracle"] for cell in cells if cell["family"] == family])
        for family in families
    ]))
    return {
        "method": method,
        "family_macro_curve": curve.tolist(),
        "asr_at_1": float(curve[0]),
        "asr_at_4": float(curve[min(3, budget - 1)]),
        "asr_at_8": float(curve[min(7, budget - 1)]),
        "asr_at_budget": float(curve[-1]),
        "auc": float(np.mean(curve)),
        "oracle_family_macro": oracle,
        "oracle_gap_at_budget": oracle - float(curve[-1]),
        "oracle_gap_closed_from_first": (
            float(curve[-1] - curve[0]) / max(oracle - float(curve[0]), 1e-12)
        ),
        "family_curves": {family: values.tolist() for family, values in family_curves.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--inner-budget", type=int, default=16)
    parser.add_argument("--robustness-penalty", type=float, default=0.0)
    parser.add_argument(
        "--selection-mode", choices=("meanstd", "minimax", "cvar25"), default="meanstd"
    )
    args = parser.parse_args()

    conventional = load_conventional()
    base = conventional.load_base()
    panel, arms = conventional.load_aggregate_panel(base, args.attack_root)
    axes = arm_axes(base, arms)
    families = sorted({base.model_family(tag) for tag in panel})
    configs = candidate_configs()
    selected_by_outer = {}
    cells = []
    for outer_family in families:
        selected_by_outer[outer_family] = {}
        for dataset in ("mj", "lg"):
            ranked = sorted(
                ((inner_score(conventional, base, panel, arms, axes, outer_family, dataset,
                              config, args.inner_budget, args.robustness_penalty,
                              args.selection_mode), config)
                 for config in configs),
                key=lambda item: (-item[0], item[1].name),
            )
            inner_value, selected = ranked[0]
            selected_by_outer[outer_family][dataset] = {
                "config": asdict(selected), "inner_lofo_auc": inner_value,
                "runner_up": [{"name": c.name, "inner_lofo_auc": score} for score, c in ranked[1:4]],
            }
            sources = source_family_vectors(base, panel, outer_family, dataset, arms)
            for tag in sorted(t for t in panel if base.model_family(t) == outer_family):
                reward = np.asarray([panel[tag][dataset][arm] for arm in arms])
                curve, queried, relevance, trace = search_curve(
                    reward, sources, axes, selected, args.budget,
                    np.random.default_rng(stable_seed(outer_family, tag, dataset, selected.name)),
                )
                cells.append({
                    "family": outer_family, "model": tag, "dataset": dataset,
                    "selected_config": selected.name, "inner_lofo_auc": inner_value,
                    "curve": curve, "queried_arms": [arms[i] for i in queried],
                    "axis_relevance": dict(zip(("composition", "order", "fragment_count", "wrapper"), relevance)),
                    "source_weight_trace": trace, "oracle": float(np.max(reward)),
                })

    summary = summarize(cells, "nested_ard_anova_transfer_gp", args.budget)
    payload = {
        "protocol": {
            "metric": "aggregate Qwen3Guard unsafe ASR (U)",
            "selection": "best observed arm within budget",
            "outer_split": "leave-one-model-family-out",
            "inner_split": "leave-one-source-family-out configuration selection",
            "target_family_used_for_prior_kernel_or_tuning": False,
            "models": sorted(panel), "families": families, "datasets": ["mj", "lg"],
            "arms": len(arms), "budget": args.budget, "inner_budget": args.inner_budget,
            "candidate_configs": len(configs),
            "inner_robustness_penalty": args.robustness_penalty,
            "inner_selection_mode": args.selection_mode,
        },
        "method": {
            "kernel": "source-learned ARD graph diffusion + second-order ANOVA + optional task covariance",
            "transfer": "family-balanced RGPE-style online source weighting",
            "acquisition_portfolio": ["expected improvement", "GP-UCB"],
        },
        "summary": summary,
        "selected_by_outer_family": selected_by_outer,
        "cells": cells,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(args.out)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
