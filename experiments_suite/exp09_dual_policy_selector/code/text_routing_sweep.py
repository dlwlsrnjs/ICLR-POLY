#!/usr/bin/env python3
"""Fast historical upper-bound sweep for text-cosine routing.

The item labels in this replay come from harmful MJ/LG grids.  Consequently this evaluates whether
text-local routing could be useful, not the final harmless331-only selector.  Outer test labels never
enter routing or method selection.  A small inner split selects one routing rule per outer split.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


ARM = re.compile(r"^(g(?:3|5|8|12)_(?:ordered|shuffled)_n2)__(plain|persona|fiction)$")


def family(tag: str) -> str:
    stem = tag.rsplit("_", 1)[0]
    for value in ("qwen", "llama", "gemma", "mistral", "phi", "falcon", "glm"):
        if stem.startswith(value):
            return value
    raise ValueError(tag)


def load_texts(runtime: Path, dataset: str, wanted: list[str]) -> tuple[list[str], list[str]]:
    name = "multijail_v1" if dataset == "mj" else "panel_v2"
    rows = {}
    with (runtime / "data" / name / "harm_grid.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["item_id"]] = (row.get("original") or row["questions"]["English"], str(row["scenario"]))
    missing = set(wanted) - set(rows)
    if missing:
        raise ValueError(f"Missing {len(missing)} texts for {dataset}: {sorted(missing)[:3]}")
    return [rows[item][0] for item in wanted], [rows[item][1] for item in wanted]


def encode(train: list[str], query: list[str], kind: str):
    matrices = []
    queries = []
    if kind in {"word", "combined"}:
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        matrices.append(vectorizer.fit_transform(train))
        queries.append(vectorizer.transform(query))
    if kind in {"char", "combined"}:
        vectorizer = TfidfVectorizer(analyzer="char_wb", lowercase=True, ngram_range=(3, 5), min_df=1, sublinear_tf=True)
        matrices.append(vectorizer.fit_transform(train))
        queries.append(vectorizer.transform(query))
    return normalize(hstack(matrices).tocsr()), normalize(hstack(queries).tocsr())


def weighted_scores(similarity: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(similarity))
    nearest = np.argsort(-similarity)[:k]
    weights = np.maximum(similarity[nearest], 0.0)
    if weights.sum() <= 1e-12:
        weights = np.ones(k)
    weights /= weights.sum()
    return weights @ labels[nearest]


def factorize(scores: np.ndarray, arms: list[str]) -> np.ndarray:
    cells = sorted({ARM.fullmatch(arm).group(1) for arm in arms})
    frames = ("plain", "persona", "fiction")
    table = np.zeros((len(cells), len(frames)))
    for index, arm in enumerate(arms):
        cell, frame = ARM.fullmatch(arm).groups()
        table[cells.index(cell), frames.index(frame)] = scores[index]
    # Both terms remain in [0,1]. Product implements the project's R(cell)*W(frame) decomposition.
    estimate = table.mean(1)[:, None] * table.mean(0)[None, :]
    return np.array([estimate[cells.index(ARM.fullmatch(arm).group(1)), frames.index(ARM.fullmatch(arm).group(2))]
                     for arm in arms])


def route(
    train_text: list[str],
    query_text: list[str],
    train_scenarios: list[str],
    query_scenarios: list[str],
    labels: np.ndarray,
    arms: list[str],
    method: str,
    seed: int,
) -> np.ndarray:
    """Return selected arm indices for each query and model: [query, model]."""
    parts = method.split(":")
    output = np.zeros((len(query_text), labels.shape[1]), dtype=int)
    if parts[0] == "scenario":
        for qi, scenario in enumerate(query_scenarios):
            members = [i for i, value in enumerate(train_scenarios) if value == scenario]
            if not members:
                members = list(range(len(train_text)))
            scores = labels[members].mean(0)
            if parts[1] == "factorized":
                scores = np.stack([factorize(row, arms) for row in scores])
            output[qi] = scores.argmax(1)
        return output

    kind, selector = parts[0], parts[1]
    train_x, query_x = encode(train_text, query_text, kind)
    if selector.startswith("knn"):
        k = int(selector.removeprefix("knn"))
        similarities = (query_x @ train_x.T).toarray()
        for qi, similarity in enumerate(similarities):
            scores = np.stack([weighted_scores(similarity, labels[:, model], k) for model in range(labels.shape[1])])
            if parts[2] == "factorized":
                scores = np.stack([factorize(row, arms) for row in scores])
            output[qi] = scores.argmax(1)
        return output

    clusters = int(selector.removeprefix("kmeans"))
    fitted = KMeans(n_clusters=min(clusters, len(train_text)), n_init=10, random_state=seed).fit(train_x)
    assignments = fitted.predict(query_x)
    for qi, cluster in enumerate(assignments):
        members = np.flatnonzero(fitted.labels_ == cluster)
        scores = labels[members].mean(0)
        if parts[2] == "factorized":
            scores = np.stack([factorize(row, arms) for row in scores])
        output[qi] = scores.argmax(1)
    return output


def candidate_methods() -> list[str]:
    result = []
    for kind in ("word", "char", "combined"):
        for k in (1, 3, 5, 10):
            for axis in ("joint", "factorized"):
                result.append(f"{kind}:knn{k}:{axis}")
        for clusters in (2, 4):
            for axis in ("joint", "factorized"):
                result.append(f"{kind}:kmeans{clusters}:{axis}")
    result += ["scenario:joint", "scenario:factorized"]
    return result


def evaluate_selection(selected: np.ndarray, labels: np.ndarray) -> np.ndarray:
    # labels [query, model, arm], selected [query, model]
    query = np.arange(labels.shape[0])[:, None]
    model = np.arange(labels.shape[1])[None, :]
    return labels[query, model, selected].mean(0)


def replay(matrix_path: Path, runtime: Path, dataset: str, splits: int) -> dict:
    data = json.loads(matrix_path.read_text(encoding="utf-8"))
    arms = sorted(arm for arm in data["matrix"][data["tags"][0]] if ARM.fullmatch(arm))
    tags = data["tags"]
    values = np.array([[[data["matrix"][tag][arm][item] for arm in arms]
                        for tag in tags] for item in range(len(data["items"]))], dtype=float)
    texts, scenarios = load_texts(runtime, dataset, data["items"])
    methods = candidate_methods()
    nested = {tag: [] for tag in tags}
    choices = []
    exploratory = {method: {tag: [] for tag in tags} for method in methods}

    for split in range(splits):
        indices = list(range(len(texts)))
        random.Random(3000 + split).shuffle(indices)
        outer_train = indices[: len(indices) // 2]
        outer_test = indices[len(indices) // 2 :]
        inner_train = outer_train[: len(outer_train) // 2]
        inner_validation = outer_train[len(outer_train) // 2 :]

        # Select one router using inner validation, macro-averaged over model families.
        validation_scores = {}
        for method in methods:
            selected = route(
                [texts[i] for i in inner_train], [texts[i] for i in inner_validation],
                [scenarios[i] for i in inner_train], [scenarios[i] for i in inner_validation],
                values[inner_train], arms, method, 5000 + split,
            )
            per_model = evaluate_selection(selected, values[inner_validation])
            validation_scores[method] = float(np.mean([
                np.mean([per_model[i] for i, tag in enumerate(tags) if family(tag) == name])
                for name in sorted({family(tag) for tag in tags})
            ]))
        chosen = max(methods, key=lambda method: (validation_scores[method], method))
        choices.append(chosen)

        for method in methods:
            selected = route(
                [texts[i] for i in outer_train], [texts[i] for i in outer_test],
                [scenarios[i] for i in outer_train], [scenarios[i] for i in outer_test],
                values[outer_train], arms, method, 7000 + split,
            )
            scores = evaluate_selection(selected, values[outer_test])
            for index, tag in enumerate(tags):
                exploratory[method][tag].append(float(scores[index]))
            if method == chosen:
                for index, tag in enumerate(tags):
                    nested[tag].append(float(scores[index]))

    families = sorted({family(tag) for tag in tags})
    nested_model = {tag: float(np.mean(score)) for tag, score in nested.items()}
    nested_family = {
        name: float(np.mean([nested_model[tag] for tag in tags if family(tag) == name])) for name in families
    }
    exploratory_macro = {}
    for method, by_model in exploratory.items():
        model_mean = {tag: float(np.mean(scores)) for tag, scores in by_model.items()}
        exploratory_macro[method] = float(np.mean([
            np.mean([model_mean[tag] for tag in tags if family(tag) == name]) for name in families
        ]))
    ranking = sorted(exploratory_macro.items(), key=lambda pair: pair[1], reverse=True)
    return {
        "schema": "exp09_text_routing_harmful_supervision_upper_bound_v1",
        "warning": "Historical harmful item labels train the router; this is not harmless-only evidence.",
        "dataset": dataset,
        "models": len(tags),
        "families": families,
        "items": len(texts),
        "arms": len(arms),
        "outer_splits": splits,
        "inner_selected_method_counts": dict(Counter(choices)),
        "nested_selected_macro": float(np.mean(list(nested_family.values()))),
        "nested_selected_per_family": nested_family,
        "exploratory_ranking": [{"method": method, "macro": score} for method, score in ranking],
    }


def main() -> None:
    here = Path(__file__).resolve()
    experiment = here.parents[1]
    suite = here.parents[2]
    project = here.parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=int, default=40)
    parser.add_argument("--output", type=Path, default=experiment / "results" / "text_routing_sweep.json")
    args = parser.parse_args()
    matrices = suite / "exp07_panel17_selector_replay" / "results"
    runtime = project / "incoming-mj-lg" / "restored" / "polyjigsaw_grid_collect_runtime"
    result = {
        dataset: replay(matrices / f"{dataset}_item_matrix.json", runtime, dataset, args.splits)
        for dataset in ("mj", "lg")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    for dataset, values in result.items():
        print(dataset.upper(), "nested", f"{values['nested_selected_macro']:.4f}", values["inner_selected_method_counts"])
        for row in values["exploratory_ranking"][:5]:
            print(" ", row["method"], f"{row['macro']:.4f}")


if __name__ == "__main__":
    main()
