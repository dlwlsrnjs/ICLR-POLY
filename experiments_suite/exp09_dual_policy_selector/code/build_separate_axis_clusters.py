#!/usr/bin/env python3
"""Build frozen canonical-text cosine clusters for the two POLY axes.

The understanding and willingness banks never share clustering observations,
centroids, posterior counts, or representatives.  The only shared object is a
read-only text->embedding cache, because the canonical string can occur in both
banks.  Rendered puzzle prompts and model responses are never embedded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


MODELS = (
    "falcon3_10b", "falcon3_3b", "falcon3_7b", "gemma2_27b", "gemma2_2b_it",
    "gemma2_9b_it", "glm4_9b", "llama31_8b_it", "llama32_3b_it", "mistral24b",
    "mistral7b", "phi35_mini", "phi3_medium_14b", "qwen25_14b", "qwen25_32b",
    "qwen25_3b", "qwen25_7b",
)
K_CANDIDATES = (4, 8, 12, 16)


def jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def normalize(values: np.ndarray) -> np.ndarray:
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def encode_bge(texts: list[str], model_path: Path, batch_size: int, query: bool = False) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_path, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    if query:
        texts = [f"Represent this sentence for searching relevant passages: {text}" for text in texts]
    output = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start:start + batch_size], padding=True, truncation=True,
                max_length=256, return_tensors="pt",
            ).to(device)
            # BGE-large-en-v1.5's published SentenceTransformers config uses CLS pooling.
            pooled = model(**batch).last_hidden_state[:, 0].float()
            output.append(pooled.cpu().numpy())
    return normalize(np.concatenate(output).astype(np.float32))


def spherical_kmeans(x: np.ndarray, k: int, seed: int, iterations: int = 100):
    rng = np.random.default_rng(seed)
    # kmeans++ under cosine distance.
    chosen = [int(rng.integers(len(x)))]
    closest = 1.0 - x @ x[chosen[0]]
    while len(chosen) < k:
        probs = np.maximum(closest, 0) ** 2
        probs = probs / probs.sum() if probs.sum() else np.full(len(x), 1 / len(x))
        nxt = int(rng.choice(len(x), p=probs))
        if nxt in chosen:
            nxt = next(i for i in range(len(x)) if i not in chosen)
        chosen.append(nxt)
        closest = np.minimum(closest, 1.0 - x @ x[nxt])
    centroids = x[chosen].copy()
    labels = np.full(len(x), -1, dtype=int)
    for _ in range(iterations):
        updated = np.argmax(x @ centroids.T, axis=1)
        if np.array_equal(updated, labels):
            break
        labels = updated
        for cluster in range(k):
            members = x[labels == cluster]
            if len(members):
                centroids[cluster] = normalize(members.mean(0, keepdims=True))[0]
            else:
                centroids[cluster] = x[int(rng.integers(len(x)))]
    return labels, centroids


def silhouette_cosine(x: np.ndarray, labels: np.ndarray) -> float:
    similarity = x @ x.T
    distance = 1.0 - similarity
    scores = []
    for index, cluster in enumerate(labels):
        own = np.flatnonzero(labels == cluster)
        a = float(distance[index, own[own != index]].mean()) if len(own) > 1 else 0.0
        alternatives = [float(distance[index, labels == other].mean())
                        for other in np.unique(labels) if other != cluster]
        b = min(alternatives) if alternatives else 0.0
        scores.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(scores))


def choose_clusters(x: np.ndarray, seed: int):
    candidates = []
    for k in K_CANDIDATES:
        if k >= len(x):
            continue
        labels, centroids = spherical_kmeans(x, k, seed + k)
        score = silhouette_cosine(x, labels)
        candidates.append((score, -k, labels, centroids))
    if not candidates:
        return np.zeros(len(x), dtype=int), normalize(x.mean(0, keepdims=True)), [{"k": 1, "silhouette": None}]
    best = max(candidates, key=lambda row: (row[0], row[1]))
    diagnostics = [{"k": -row[1], "silhouette": row[0]} for row in candidates]
    return best[2], best[3], diagnostics


def load_query_texts(runtime: Path):
    result = {}
    for dataset, name in (("mj", "multijail_v1"), ("lg", "panel_v2")):
        path = runtime / "data" / name / "harm_grid.jsonl"
        for row in jsonl(path):
            result[f"{dataset}::{row['item_id']}"] = row.get("original") or row["questions"]["English"]
    return result


def read_banks(root: Path, model: str):
    base = root / model
    understanding = list(jsonl(base / "understanding_axis/reconstruction_success_cases.jsonl"))
    strict = list(jsonl(base / "willingness_axis/strict_success_cases.jsonl"))
    effective = list(jsonl(base / "willingness_axis/technique_effective_cases.jsonl"))
    return understanding, strict, effective


def build_axis(model: str, axis: str, rows: list[dict], effective_keys: set[tuple[str, str, str]],
               embeddings: dict[str, np.ndarray], out: Path):
    arm_field = "understanding_arm" if axis == "understanding" else "willingness_frame"
    by_item = defaultdict(list)
    for row in rows:
        by_item[row["item_id"]].append(row)
    item_ids = sorted(by_item)
    x = np.stack([embeddings[by_item[item][0]["canonical_text"]] for item in item_ids])
    labels, centroids, diagnostics = choose_clusters(x, stable_seed(model, axis))
    records = []
    assignments = []
    for cluster in range(len(centroids)):
        members = np.flatnonzero(labels == cluster)
        counts = Counter()
        unique_support = defaultdict(set)
        effective_counts = Counter()
        for member in members:
            item = item_ids[member]
            assignments.append({
                "model": model, "axis": axis, "item_id": item,
                "canonical_text": by_item[item][0]["canonical_text"], "cluster": cluster,
                "centroid_cosine": float(x[member] @ centroids[cluster]),
            })
            for row in by_item[item]:
                arm = row[arm_field]
                counts[arm] += 1
                unique_support[arm].add(item)
                if (row["model"], item, arm) in effective_keys:
                    effective_counts[arm] += 1
        ranked = sorted(counts, key=lambda arm: (
            effective_counts[arm] if axis == "willingness" else len(unique_support[arm]),
            len(unique_support[arm]), counts[arm], arm,
        ), reverse=True)
        records.append({
            "cluster": cluster,
            "items": len(members),
            "representative_setting": ranked[0] if ranked else None,
            "setting_support": [
                {"setting": arm, "success_rows": counts[arm], "unique_items": len(unique_support[arm]),
                 "positive_effect_rows": effective_counts[arm] if axis == "willingness" else None,
                 "beta_support_mean": (counts[arm] + 1) / (sum(counts.values()) + len(counts))}
                for arm in ranked
            ],
        })
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "centroids.npy", centroids.astype(np.float32))
    with (out / "assignments.jsonl").open("w", encoding="utf-8") as handle:
        for row in assignments:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload = {
        "schema": "poly_separate_axis_cosine_clusters/v1",
        "model": model, "axis": axis,
        "embedding_input": "canonical_text_only",
        "embedding_normalization": "L2",
        "distance": "cosine",
        "joint_axis_state_created": False,
        "k": len(centroids), "k_selection": diagnostics,
        "clusters": records,
    }
    (out / "clusters.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main():
    here = Path(__file__).resolve()
    project = here.parents[4]
    parser = argparse.ArgumentParser()
    parser.add_argument("--banks", type=Path, default=project / "workspaces/prior_axes_mj_lg_20260920/05_method_selection_ready/10_separate_model_frontier_priors")
    parser.add_argument("--runtime", type=Path, default=project / "incoming-mj-lg/restored/polyjigsaw_grid_collect_runtime")
    parser.add_argument("--model-path", type=Path, default=project / "hf-cache/hub/models--BAAI--bge-large-en-v1.5/snapshots/d4aa6901d3a41ba39fb536a557fa166f842b0e09")
    parser.add_argument("--output", type=Path, default=here.parents[1] / "results/separate_axis_clusters")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    loaded = {}
    texts = {}
    for model in MODELS:
        understanding, strict, effective = read_banks(args.banks, model)
        loaded[model] = (understanding, strict, effective)
        for row in understanding + strict:
            texts[row["canonical_text"]] = None
    query_texts = load_query_texts(args.runtime)
    ordered_texts = sorted(texts)
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "canonical_bge_embeddings.npz"
    manifest = args.output / "canonical_bge_texts.json"
    if cache.exists() and manifest.exists() and json.loads(manifest.read_text()) == ordered_texts:
        matrix = np.load(cache)["embeddings"]
    else:
        matrix = encode_bge(ordered_texts, args.model_path, args.batch_size, query=False)
        np.savez_compressed(cache, embeddings=matrix)
        manifest.write_text(json.dumps(ordered_texts, ensure_ascii=False) + "\n", encoding="utf-8")
    embedding = dict(zip(ordered_texts, matrix))
    ordered_queries = sorted(query_texts)
    query_cache = args.output / "query_bge_embeddings.npz"
    query_manifest = args.output / "query_bge_keys.json"
    if query_cache.exists() and query_manifest.exists() and json.loads(query_manifest.read_text()) == ordered_queries:
        query_matrix = np.load(query_cache)["embeddings"]
    else:
        query_matrix = encode_bge([query_texts[key] for key in ordered_queries], args.model_path, args.batch_size, query=True)
        np.savez_compressed(query_cache, embeddings=query_matrix)
        query_manifest.write_text(json.dumps(ordered_queries, ensure_ascii=False) + "\n", encoding="utf-8")
    query_index = {key: index for index, key in enumerate(ordered_queries)}
    (args.output / "query_embedding_index.json").write_text(
        json.dumps(query_index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = []
    for model, (understanding, strict, effective) in loaded.items():
        effective_keys = {(row["model"], row["item_id"], row["willingness_frame"]) for row in effective}
        u = build_axis(model, "understanding", understanding, set(), embedding, args.output / model / "understanding")
        w = build_axis(model, "willingness", strict, effective_keys, embedding, args.output / model / "willingness")
        summary.append({"model": model, "understanding_k": u["k"], "willingness_k": w["k"]})
    # Deployment starts without the target model identity.  These pooled clusters
    # provide the anonymous first probes; per-model banks remain audit/signature
    # references and are never selected by target name in the replay.
    all_understanding = [row for value in loaded.values() for row in value[0]]
    all_strict = [row for value in loaded.values() for row in value[1]]
    all_effective = {(row["model"], row["item_id"], row["willingness_frame"])
                     for value in loaded.values() for row in value[2]}
    global_u = build_axis("global_anonymous", "understanding", all_understanding, set(), embedding,
                          args.output / "global_anonymous/understanding")
    global_w = build_axis("global_anonymous", "willingness", all_strict, all_effective, embedding,
                          args.output / "global_anonymous/willingness")
    index = {
        "schema": "poly_separate_axis_cluster_index/v1",
        "embedding_model": str(args.model_path),
        "pooling": "CLS (model-card 1_Pooling setting)",
        "query_instruction": "Represent this sentence for searching relevant passages: ",
        "normalization": "L2", "similarity": "cosine",
        "joint_prior_created": False, "joint_embedding_created": False,
        "deployment_start": "global_anonymous clusters; target model name is unavailable",
        "global_anonymous": {"understanding_k": global_u["k"], "willingness_k": global_w["k"]},
        "models": summary,
    }
    (args.output / "INDEX.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
