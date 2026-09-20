#!/usr/bin/env python3
"""Replay several 1--8 harmful-confirmation strategies on the frozen 24-arm MJ/LG grid."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from replay_harmful_confirmation import ARM, SIGMA, features, gp_posterior


def run_strategy(strategy, budget, arms, arm_features, prior, observed_values, rng):
    queried, outcomes = [], []
    for _ in range(budget):
        if strategy == "random":
            choices = [i for i in range(len(arms)) if i not in queried]
            nxt = int(rng.choice(choices))
        elif strategy == "prior_order":
            score = prior.copy()
            score[queried] = -np.inf
            nxt = int(np.argmax(score))
        else:
            mean0 = 0.6 * prior
            if queried:
                mean, sd = gp_posterior(arm_features[queried], arm_features, np.array(outcomes), mean0[queried], mean0)
            else:
                mean, sd = mean0.copy(), np.full(len(arms), SIGMA)
            if strategy == "gp_greedy":
                score = mean
            elif strategy.startswith("gp_ucb"):
                beta = float(strategy.rsplit("_", 1)[1])
                score = mean + beta * sd
            elif strategy == "gp_thompson":
                score = rng.normal(mean, sd)
            else:
                raise ValueError(strategy)
            score[queried] = -np.inf
            nxt = int(np.argmax(score))
        queried.append(nxt)
        outcomes.append(observed_values[nxt])
    return max(queried, key=lambda index: observed_values[index])


def replay(matrix_path, benign_dir, splits=40):
    data = json.loads(matrix_path.read_text())
    tags = [tag for tag in data["tags"] if (benign_dir / f"{tag}.json").exists()]
    arms = sorted(arm for arm in data["matrix"][tags[0]] if ARM.fullmatch(arm))
    feats = np.array([features(arm) for arm in arms])
    priors = {tag: json.loads((benign_dir / f"{tag}.json").read_text())["prior"] for tag in tags}
    strategies = ["random", "prior_order", "gp_greedy", "gp_ucb_0.5", "gp_ucb_1.0", "gp_ucb_2.0", "gp_thompson"]
    store = {f"{strategy}@{budget}": {tag: [] for tag in tags}
             for strategy in strategies for budget in range(1, 9)}
    for split in range(splits):
        ids = list(range(len(data["items"])))
        random.Random(9000 + split).shuffle(ids)
        calibration, test = ids[: len(ids)//2], ids[len(ids)//2 :]
        for tag in tags:
            cal = np.array([np.mean([data["matrix"][tag][arm][i] for i in calibration]) for arm in arms])
            tst = np.array([np.mean([data["matrix"][tag][arm][i] for i in test]) for arm in arms])
            prior = np.array([priors[tag].get(arm, 0.0) for arm in arms])
            for strategy in strategies:
                for budget in range(1, 9):
                    rng = np.random.default_rng(100000 * split + 1000 * tags.index(tag) + 10 * budget + strategies.index(strategy))
                    picked = run_strategy(strategy, budget, arms, feats, prior, cal, rng)
                    store[f"{strategy}@{budget}"][tag].append(float(tst[picked]))
    summary = []
    for key, per_tag in store.items():
        model = {tag: float(np.mean(values)) for tag, values in per_tag.items()}
        fams = sorted({tag.split("3", 1)[0] if tag.startswith("phi3") else tag.split("2", 1)[0] for tag in tags})
        # Report model macro here; family-balanced confirmatory inference is performed by the main report.
        summary.append({"method": key, "model_macro": float(np.mean(list(model.values()))), "per_model": model})
    summary.sort(key=lambda row: row["model_macro"], reverse=True)
    return {"matrix": str(matrix_path.resolve()), "models": tags, "arms": arms, "splits": splits,
            "warning": "Historical replay; harmful calibration batches are table lookups.", "ranking": summary}


def main():
    here = Path(__file__).resolve(); experiment = here.parents[1]; suite = here.parents[2]
    matrices = suite / "exp07_panel17_selector_replay" / "results"
    benign = suite / "exp02_panel_collect" / "results" / "benign"
    result = {ds: replay(matrices / f"{ds}_item_matrix.json", benign) for ds in ("mj", "lg")}
    out = experiment / "results" / "harmful_algorithm_sweep.json"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    for ds in result:
        print(ds.upper())
        for row in result[ds]["ranking"][:12]: print(row["method"], f"{row['model_macro']:.4f}")


if __name__ == "__main__":
    main()
