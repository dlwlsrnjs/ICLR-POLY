#!/usr/bin/env python3
"""Leak-free historical ablation for the policy components available before 331 R/F/Y finishes.

This replay can compare harmful-only search, kernel structure, a one-shot harmless prior and arm-space
size.  It deliberately does not simulate cosine routing, repeated harmless observations or WildGuard
state transitions because those item-level observations do not exist in the historical matrix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


ARM = re.compile(r"^g(3|5|8|12)_(ordered|shuffled)_n(2|4|6|8)__(plain|persona|fiction|pap|persona\+fiction)$")
SPACES = {
    "A32": {"plain"},
    "B64": {"plain", "persona"},
    "B96": {"plain", "persona", "fiction"},
    "C160": {"plain", "persona", "fiction", "pap", "persona+fiction"},
}
METHODS = ("random_harmful", "flat_isotropic_gp", "flat_additive_gp", "prior_isotropic_gp", "prior_additive_gp")
SIGMA = 0.35
NOISE = 0.02
BETA = 1.0
PRIOR_SCALE = 0.6


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def family(tag: str) -> str:
    for value in ("qwen", "llama", "gemma", "mistral", "phi", "falcon", "glm"):
        if tag.startswith(value):
            return value
    raise ValueError(tag)


def features(arm: str) -> tuple[np.ndarray, np.ndarray]:
    match = ARM.fullmatch(arm)
    if not match:
        raise ValueError(arm)
    g, order, n, frame = match.groups()
    understanding = np.array([int(g) / 12.0, float(order == "shuffled"), int(n) / 8.0])
    frames = ("plain", "persona", "fiction", "pap", "persona+fiction")
    willingness = np.array([float(frame == value) for value in frames])
    return understanding, willingness


def covariance(arms: list[str], kind: str) -> np.ndarray:
    parts = [features(arm) for arm in arms]
    u = np.stack([part[0] for part in parts])
    w = np.stack([part[1] for part in parts])

    def rbf(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
        distance = ((x[:, None] - x[None, :]) ** 2).sum(-1)
        return SIGMA**2 * np.exp(-distance / (2 * scale**2))

    if kind == "isotropic":
        return rbf(np.concatenate([u, w], axis=1))
    if kind == "additive":
        # Separate sharing along the comprehension and willingness axes.
        return 0.5 * rbf(u, 0.55) + 0.5 * rbf(w, 0.75)
    raise ValueError(kind)


def posterior(kernel: np.ndarray, observed: list[int], y: list[float], prior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not observed:
        return prior.copy(), np.sqrt(np.maximum(np.diag(kernel), 1e-12))
    q = np.asarray(observed, dtype=int)
    kqq = kernel[np.ix_(q, q)] + NOISE * np.eye(len(q))
    cross = kernel[:, q]
    alpha = np.linalg.solve(kqq, np.asarray(y) - prior[q])
    mean = prior + cross @ alpha
    solved = np.linalg.solve(kqq, cross.T)
    var = np.maximum(np.diag(kernel) - (cross * solved.T).sum(1), 1e-12)
    return mean, np.sqrt(var)


def trajectory(method: str, kernel: np.ndarray, prior: np.ndarray, calibration: np.ndarray, budget: int, rng: np.random.Generator) -> list[int]:
    observed: list[int] = []
    outcomes: list[float] = []
    for _ in range(budget):
        remaining = np.setdiff1d(np.arange(len(prior)), np.asarray(observed, dtype=int), assume_unique=False)
        if method == "random_harmful":
            nxt = int(rng.choice(remaining))
        else:
            gp_prior = PRIOR_SCALE * prior if method.startswith("prior") else np.full(len(prior), 0.3)
            mean, sd = posterior(kernel, observed, outcomes, gp_prior)
            acquisition = mean + BETA * sd
            acquisition[observed] = -np.inf
            # Random jitter prevents arm-file ordering from deciding a flat-prior tie.
            acquisition += rng.uniform(0, 1e-10, len(acquisition))
            nxt = int(np.argmax(acquisition))
        observed.append(nxt)
        outcomes.append(float(calibration[nxt]))
    return observed


def recommendation(method: str, kernel: np.ndarray, prior: np.ndarray, calibration: np.ndarray, observed: list[int]) -> int:
    if method == "random_harmful":
        return max(observed, key=lambda index: calibration[index])
    gp_prior = PRIOR_SCALE * prior if method.startswith("prior") else np.full(len(prior), 0.3)
    mean, _ = posterior(kernel, observed, [float(calibration[i]) for i in observed], gp_prior)
    return int(np.argmax(mean))


def replay(matrix_path: Path, benign_dir: Path, splits: int, max_budget: int) -> dict:
    data = json.loads(matrix_path.read_text())
    tags = [tag for tag in data["tags"] if (benign_dir / f"{tag}.json").exists()]
    priors = {tag: json.loads((benign_dir / f"{tag}.json").read_text())["prior"] for tag in tags}
    all_arms = sorted(arm for arm in data["matrix"][tags[0]] if ARM.fullmatch(arm))
    output: dict[str, object] = {}
    for space, allowed_frames in SPACES.items():
        arms = [arm for arm in all_arms if ARM.fullmatch(arm).group(4) in allowed_frames]
        indices = {arm: all_arms.index(arm) for arm in arms}
        kernels = {kind: covariance(arms, kind) for kind in ("isotropic", "additive")}
        records = {method: [] for method in METHODS}
        for split in range(splits):
            shuffled = np.random.default_rng(stable_seed(matrix_path.name, split)).permutation(len(data["items"]))
            calibration_ids, test_ids = np.array_split(shuffled, 2)
            for tag in tags:
                values = np.array([data["matrix"][tag][arm] for arm in arms], dtype=float)
                calibration = values[:, calibration_ids].mean(1)
                test = values[:, test_ids].mean(1)
                prior = np.array([priors[tag].get(arm, 0.0) for arm in arms])
                oracle = float(test.max())
                for method in METHODS:
                    kernel_kind = "additive" if "additive" in method else "isotropic"
                    rng = np.random.default_rng(stable_seed(matrix_path.name, space, split, tag, method))
                    observed = trajectory(method, kernels[kernel_kind], prior, calibration, max_budget, rng)
                    curve = []
                    for budget in range(1, max_budget + 1):
                        chosen = recommendation(method, kernels[kernel_kind], prior, calibration, observed[:budget])
                        curve.append(float(test[chosen]))
                    records[method].append({"split": split, "tag": tag, "family": family(tag), "oracle": oracle, "curve": curve})

        summary = {}
        for method, rows in records.items():
            families = sorted({row["family"] for row in rows})
            curve = []
            recovery = []
            for budget in range(max_budget):
                family_scores = []
                family_ratios = []
                for name in families:
                    selected = [row for row in rows if row["family"] == name]
                    family_scores.append(np.mean([row["curve"][budget] for row in selected]))
                    family_ratios.append(np.mean([row["curve"][budget] / row["oracle"] if row["oracle"] else 1.0 for row in selected]))
                curve.append(float(np.mean(family_scores)))
                recovery.append(float(np.mean(family_ratios)))
            crossing = []
            for row in rows:
                hit = next((i + 1 for i, value in enumerate(row["curve"]) if value >= 0.9 * row["oracle"]), None)
                crossing.append(hit)
            summary[method] = {
                "verified_asr_by_budget": curve,
                "oracle_recovery_by_budget": recovery,
                "harmful_budget_to_family_macro_90_oracle": next((i + 1 for i, value in enumerate(recovery) if value >= 0.9), None),
                "harmful_pulls_to_90_oracle_mean_among_solved": float(np.mean([x for x in crossing if x is not None])) if any(x is not None for x in crossing) else None,
                "harmful_pulls_to_90_oracle_capped_mean": float(np.mean([x if x is not None else max_budget + 1 for x in crossing])),
                "within_budget_90_oracle_rate": float(np.mean([x is not None for x in crossing])),
                "normalized_auc_1_to_max_budget": float(np.mean(curve)),
            }
        output[space] = {"arms": len(arms), "models": len(tags), "splits": splits, "methods": summary}
    return output


def main() -> None:
    here = Path(__file__).resolve()
    experiment = here.parents[1]
    suite = here.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--max-budget", type=int, default=12)
    parser.add_argument("--output", type=Path, default=experiment / "results" / "static_policy_ablation.json")
    args = parser.parse_args()
    matrices = suite / "exp07_panel17_selector_replay" / "results"
    benign = suite / "exp02_panel_collect" / "results" / "benign"
    result = {
        "schema": "exp09_static_policy_ablation/v1",
        "warning": "Historical verified matrix only: no text clusters, repeated harmless pulls, or WildGuard state simulation.",
        "datasets": {
            dataset: replay(matrices / f"{dataset}_item_matrix.json", benign, args.splits, args.max_budget)
            for dataset in ("mj", "lg")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for dataset, spaces in result["datasets"].items():
        print(dataset.upper())
        for space, values in spaces.items():
            best = max(values["methods"].items(), key=lambda pair: pair[1]["normalized_auc_1_to_max_budget"])
            print(space, values["arms"], best[0], f"AUC={best[1]['normalized_auc_1_to_max_budget']:.4f}",
                  f"@{args.max_budget}={best[1]['verified_asr_by_budget'][-1]:.4f}")


if __name__ == "__main__":
    main()
