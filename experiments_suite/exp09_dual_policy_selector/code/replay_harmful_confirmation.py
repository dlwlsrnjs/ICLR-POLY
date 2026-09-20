#!/usr/bin/env python3
"""Historical replay: harmless-prior-only versus GP warm-start with 3--8 harmful batches.

This is deliberately labelled a regression replay.  The historical benign prior is aggregated,
so it cannot simulate the new item-level dynamic harmless probing policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np


ARM = re.compile(r"^g(3|5|8|12)_(ordered|shuffled)_n2__(plain|persona|fiction)$")
FRAMES = ("plain", "persona", "fiction")
U0 = 0.6
BETA = 1.0
SIGMA = 0.35
LENGTH_SCALE = 1.0
NOISE = 0.02


def stable_seed(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def family(tag: str) -> str:
    base = tag.rsplit("_", 1)[0]
    for name in ("qwen", "llama", "gemma", "mistral", "phi", "falcon", "glm"):
        if base.startswith(name):
            return name
    raise ValueError(f"Unknown model family: {tag}")


def features(arm: str) -> np.ndarray:
    match = ARM.fullmatch(arm)
    if not match:
        raise ValueError(arm)
    g, order, frame = match.groups()
    frame_onehot = [float(frame == value) for value in FRAMES]
    return np.array([int(g) / 12.0, float(order == "shuffled"), 2.0 / 8.0, *frame_onehot])


def gp_posterior(
    observed_x: np.ndarray,
    query_x: np.ndarray,
    observed_y: np.ndarray,
    observed_prior: np.ndarray,
    query_prior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    def kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        distance = ((left[:, None, :] - right[None, :, :]) ** 2).sum(-1)
        return SIGMA**2 * np.exp(-distance / (2 * LENGTH_SCALE**2))

    if len(observed_x) == 0:
        return query_prior, np.full(len(query_x), SIGMA)
    covariance = kernel(observed_x, observed_x) + NOISE * np.eye(len(observed_x))
    cross = kernel(query_x, observed_x)
    alpha = np.linalg.solve(covariance, observed_y - observed_prior)
    mean = query_prior + cross @ alpha
    solved = np.linalg.solve(covariance, cross.T)
    variance = np.clip(SIGMA**2 - (cross * solved.T).sum(1), 1e-9, None)
    return mean, np.sqrt(variance)


def sign_flip_p(values: np.ndarray) -> float:
    count = len(values)
    masks = np.arange(1 << count, dtype=np.int64)[:, None]
    signs = 1 - 2 * ((masks >> np.arange(count)) & 1).astype(np.int8)
    null = (signs * values).mean(1)
    return float((np.abs(null) >= abs(values.mean()) - 1e-12).mean())


def replay(matrix_path: Path, benign_dir: Path, splits: int) -> dict:
    data = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix = data["matrix"]
    available = [tag for tag in data["tags"] if (benign_dir / f"{tag}.json").exists()]
    if not available:
        raise ValueError(f"No harmless priors match {matrix_path}")
    arms = sorted(arm for arm in matrix[available[0]] if ARM.fullmatch(arm))
    if len(arms) != 24:
        raise ValueError(f"Expected the frozen 24-arm space, got {len(arms)}")
    priors = {
        tag: json.loads((benign_dir / f"{tag}.json").read_text(encoding="utf-8"))["prior"]
        for tag in available
    }
    arm_features = np.array([features(arm) for arm in arms])
    methods = ["prior_only", "fixed_family_loo", *(f"warm_gp_{n}" for n in range(3, 9))]
    scores = {method: {tag: [] for tag in available} for method in methods}
    selections = {method: {tag: [] for tag in available} for method in methods}

    for split in range(splits):
        indices = list(range(len(data["items"])))
        random.Random(1000 + split).shuffle(indices)
        calibration, test = indices[: len(indices) // 2], indices[len(indices) // 2 :]
        cal = {
            tag: {arm: float(np.mean([matrix[tag][arm][i] for i in calibration])) for arm in arms}
            for tag in available
        }
        tst = {
            tag: {arm: float(np.mean([matrix[tag][arm][i] for i in test])) for arm in arms}
            for tag in available
        }
        for tag in available:
            prior = np.array([priors[tag].get(arm, 0.0) for arm in arms], dtype=float)
            chosen = int(np.argmax(prior))
            scores["prior_only"][tag].append(tst[tag][arms[chosen]])
            selections["prior_only"][tag].append(arms[chosen])

            others = [other for other in available if family(other) != family(tag)]
            fixed_mean = {arm: float(np.mean([cal[other][arm] for other in others])) for arm in arms}
            fixed_arm = max(fixed_mean, key=fixed_mean.get)
            scores["fixed_family_loo"][tag].append(tst[tag][fixed_arm])
            selections["fixed_family_loo"][tag].append(fixed_arm)

            prior_mean = U0 * prior
            queried: list[int] = []
            observations: list[float] = []
            for budget in range(1, 9):
                if queried:
                    mean, deviation = gp_posterior(
                        arm_features[queried],
                        arm_features,
                        np.array(observations),
                        prior_mean[queried],
                        prior_mean,
                    )
                else:
                    mean, deviation = prior_mean.copy(), np.full(len(arms), SIGMA)
                acquisition = mean + BETA * deviation
                acquisition[queried] = -np.inf
                next_arm = int(np.argmax(acquisition))
                queried.append(next_arm)
                observations.append(cal[tag][arms[next_arm]])
                if budget >= 3:
                    method = f"warm_gp_{budget}"
                    best = max(queried, key=lambda index: cal[tag][arms[index]])
                    scores[method][tag].append(tst[tag][arms[best]])
                    selections[method][tag].append(arms[best])

    per_model = {
        method: {tag: float(np.mean(values)) for tag, values in by_model.items()}
        for method, by_model in scores.items()
    }
    families = sorted({family(tag) for tag in available})
    per_family = {
        method: {
            name: float(np.mean([value for tag, value in by_model.items() if family(tag) == name]))
            for name in families
        }
        for method, by_model in per_model.items()
    }
    comparisons = {}
    baseline = np.array([per_family["prior_only"][name] for name in families])
    for budget in range(3, 9):
        method = f"warm_gp_{budget}"
        current = np.array([per_family[method][name] for name in families])
        difference = current - baseline
        comparisons[method] = {
            "mean_delta_vs_prior_only": float(difference.mean()),
            "family_wins": int((difference > 0).sum()),
            "families": len(families),
            "sign_flip_p": sign_flip_p(difference),
        }
    return {
        "schema": "exp09_historical_harmful_confirmation_replay_v1",
        "warning": "prior_only is not the new dynamic harmless-only policy",
        "matrix": str(matrix_path.resolve()),
        "models": available,
        "families": families,
        "items": len(data["items"]),
        "arms": arms,
        "splits": splits,
        "per_model": per_model,
        "per_family": per_family,
        "comparisons": comparisons,
    }


def main() -> None:
    here = Path(__file__).resolve()
    experiment = here.parents[1]
    suite = here.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=int, default=40)
    parser.add_argument("--output", type=Path, default=experiment / "results" / "historical_replay.json")
    args = parser.parse_args()
    if args.splits < 1:
        parser.error("--splits must be positive")
    matrices = suite / "exp07_panel17_selector_replay" / "results"
    benign = suite / "exp02_panel_collect" / "results" / "benign"
    result = {
        dataset: replay(matrices / f"{dataset}_item_matrix.json", benign, args.splits)
        for dataset in ("mj", "lg")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    for dataset, values in result.items():
        print(dataset.upper(), "historical verified replay")
        print("  prior_only", np.mean(list(values["per_family"]["prior_only"].values())))
        for method, comparison in values["comparisons"].items():
            mean = np.mean(list(values["per_family"][method].values()))
            print(f"  {method}: {mean:.4f} delta={comparison['mean_delta_vs_prior_only']:+.4f}")


if __name__ == "__main__":
    main()
