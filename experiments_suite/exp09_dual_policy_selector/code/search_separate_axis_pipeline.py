#!/usr/bin/env python3
"""Selection/validation/test search for the anonymous separate-axis pipeline."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
SPEC = importlib.util.spec_from_file_location("axis_ablation", HERE.with_name("run_separate_axis_ablation.py"))
A = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(A)


def split_name(item: str) -> str:
    value = int.from_bytes(hashlib.sha256(("poly-exp09-split-v1|" + item).encode()).digest()[:8], "big") % 100
    return "selection" if value < 50 else "validation" if value < 75 else "test"


def candidates(limit: int):
    product = list(itertools.product(
        ("P4_stagnation_local_search", "P5_response_state_dynamic", "P6_cosine_ood_gated_dynamic"),
        (20, 40, 80), (5, 10, 20), (.005, .01, .02),
        (.75, .9), (.3, .7), (.5, 1.0, 2.0), (2, 4),
        ((1, 2), (1, 4), (2, 6)),
    ))
    rng = np.random.default_rng(20260920)
    chosen = rng.choice(len(product), size=min(limit, len(product)), replace=False)
    rows = []
    for index in chosen:
        method, cap, patience, epsilon, likelihood, mix, beta, local, penalties = product[int(index)]
        rows.append({
            "method": method, "harmless_cap": cap, "patience": patience, "epsilon": epsilon,
            "likelihood_match": likelihood, "mixture_weight": mix,
            "harmless_ucb_beta": beta, "harmful_ucb_beta": beta,
            "local_top_k": local, "partial_penalty": penalties[0], "full_penalty": penalties[1],
            "ood_threshold": (0.55, 0.58, 0.60, 0.62, 0.65)[int(index) % 5],
        })
    # Always include the documented defaults and boundary controls.
    rows += [
        {"method": "P4_stagnation_local_search", "harmless_cap": 40, "patience": 10, "epsilon": .01,
         "likelihood_match": .9, "mixture_weight": .7, "harmless_ucb_beta": 1., "harmful_ucb_beta": 1.,
         "local_top_k": 4, "partial_penalty": 1, "full_penalty": 4},
        {"method": "P5_response_state_dynamic", "harmless_cap": 40, "patience": 10, "epsilon": .01,
         "likelihood_match": .9, "mixture_weight": .7, "harmless_ucb_beta": 1., "harmful_ucb_beta": 1.,
         "local_top_k": 4, "partial_penalty": 1, "full_penalty": 4},
        {"method": "P6_cosine_ood_gated_dynamic", "harmless_cap": 40, "patience": 10, "epsilon": .01,
         "likelihood_match": .9, "mixture_weight": .7, "harmless_ucb_beta": 1., "harmful_ucb_beta": 1.,
         "local_top_k": 4, "partial_penalty": 1, "full_penalty": 4, "ood_threshold": .60},
    ]
    # Axis-local OOD gates.  Retrieval remains cosine, while the gate compares
    # raw centroid similarity, local density, and in-bank calibrated scores.
    p7_modes = {
        "centroid_cosine": (.55, .60, .65),
        "knn_density": (.45, .55, .65),
        "conformal_density": (.01, .03, .05, .15, .30, .50),
        "conformal_hybrid": (.01, .03, .05, .15, .30, .50),
        "diag_mahalanobis_conformal": (.01, .03, .05, .15, .30),
    }
    p7_base = [
        (20, 10, .01, .75, .3, .5, 4, 1, 2),
        (20, 10, .02, .75, .7, 1., 4, 1, 2),
        (40, 5, .01, .9, .7, 1., 2, 1, 4),
        (40, 10, .01, .9, .7, 1., 4, 1, 4),
        (80, 20, .005, .9, .3, 2., 4, 2, 6),
    ]
    for mode, thresholds in p7_modes.items():
        for threshold in thresholds:
            for cap, patience, epsilon, likelihood, mix, beta, local, partial, full in p7_base[:1]:
                rows.append({
                    "method": "P7_axis_local_calibrated_ood", "harmless_cap": cap,
                    "patience": patience, "epsilon": epsilon, "likelihood_match": likelihood,
                    "mixture_weight": mix, "harmless_ucb_beta": beta, "harmful_ucb_beta": beta,
                    "local_top_k": local, "partial_penalty": partial, "full_penalty": full,
                    "ood_score": mode, "ood_threshold": threshold,
                })
    # Cluster-free soft kNN priors are a direct alternative to hard clustering.
    for mode, thresholds in {
        "conformal_density": (.01, .03, .05, .15),
        "diag_mahalanobis_conformal": (.01, .03, .05, .15),
    }.items():
        for threshold in thresholds:
            for top_k in (8, 16, 32):
                for cap, patience, epsilon, likelihood, mix, beta, local, partial, full in p7_base[:1]:
                    rows.append({
                        "method": "P8_soft_knn_axis_local_ood", "harmless_cap": cap,
                        "patience": patience, "epsilon": epsilon, "likelihood_match": likelihood,
                        "mixture_weight": mix, "harmless_ucb_beta": beta, "harmful_ucb_beta": beta,
                        "local_top_k": local, "partial_penalty": partial, "full_penalty": full,
                        "ood_score": mode, "ood_threshold": threshold, "router_top_k": top_k,
                    })
    # Primary simple design: all valid successes and failures form two soft-kNN
    # priors; invalid rows are missing observations, not forced failures.
    for top_k in (8, 16, 32, 64):
        for cap, patience, epsilon, likelihood, mix, beta, local, partial, full in p7_base:
            rows.append({
                "method": "P9_full_evidence_soft_knn", "harmless_cap": cap,
                "patience": patience, "epsilon": epsilon, "likelihood_match": likelihood,
                "mixture_weight": 0.0, "harmless_ucb_beta": beta, "harmful_ucb_beta": beta,
                "local_top_k": local, "partial_penalty": partial, "full_penalty": full,
                "router_top_k": top_k,
            })
    for top_k in (16, 32):
        for mix in (.3, .7):
            for cap, patience, epsilon, likelihood, _, beta, local, partial, full in p7_base[:4]:
                rows.append({
                    "method": "P10_full_evidence_model_mixture", "harmless_cap": cap,
                    "patience": patience, "epsilon": epsilon, "likelihood_match": likelihood,
                    "mixture_weight": mix, "harmless_ucb_beta": beta, "harmful_ucb_beta": beta,
                    "local_top_k": local, "partial_penalty": partial, "full_penalty": full,
                    "router_top_k": top_k,
                })
    unique = {json.dumps(row, sort_keys=True): row for row in rows}
    return list(unique.values())


def better(summary: dict):
    recovery = summary["oracle_recovery"] or 0
    if recovery >= .9:
        return (1, -summary["mean_harmful_pulls"], summary["harmful_budget_auc"], summary["verified_asr"])
    return (0, summary["harmful_budget_auc"], summary["verified_asr"], -summary["mean_harmful_pulls"])


def macro_summary(rows: list[dict]):
    """Equal-dataset macro used to choose one deployable policy."""
    keys = ("verified_asr", "oracle_verified_asr", "oracle_recovery", "harmful_budget_auc",
            "mean_harmful_pulls", "mean_harmless_requests")
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def prepare(project: Path, experiment: Path, suite: Path, clusters: Path, dataset: str):
    text_list = json.loads((clusters / "canonical_bge_texts.json").read_text())
    text_index = {text: i for i, text in enumerate(text_list)}
    embeddings = np.load(clusters / "canonical_bge_embeddings.npz")["embeddings"]
    query_embeddings = np.load(clusters / "query_bge_embeddings.npz")["embeddings"]
    query_index = json.loads((clusters / "query_embedding_index.json").read_text())
    data = json.loads((suite / f"exp07_panel17_selector_replay/results/{dataset}_item_matrix.json").read_text())
    arms = sorted(arm for arm in data["matrix"][data["tags"][0]] if A.ARM_RE.fullmatch(arm))
    states_payload = A.build_state_matrix(project, data, dataset, experiment / f"results/{dataset}_response_state_matrix.json")
    models = sorted({tag.rsplit("_", 1)[0] for tag in data["tags"]})
    harmless_by_model = {model: A.load_harmless(project, model) for model in models}
    reference_harmless = [harmless_by_model[model] for model in models]
    reference_global = {
        axis: np.stack([A.global_priors(harmless_by_model[model])[axis] for model in models])
        for axis in ("understanding", "willingness")
    }
    global_prior = {axis: reference_global[axis].mean(0) for axis in reference_global}
    global_clusters = {axis: A.load_clusters(clusters, "global_anonymous", axis) for axis in ("understanding", "willingness")}
    reference_clusters = {model: {axis: A.load_clusters(clusters, model, axis) for axis in ("understanding", "willingness")} for model in models}
    route_cache = {}
    for item in data["items"]:
        query = query_embeddings[query_index[f"{dataset}::{item}"]]
        routed = {}
        for axis, settings in (("understanding", A.CELLS), ("willingness", A.FRAMES)):
            routed[axis] = np.stack([
                A.routed_prior(query, reference_clusters[model][axis][0], reference_clusters[model][axis][1], settings)[0]
                for model in models
            ])
        scores = {
            axis: A.route_confidence_scores(query, global_clusters[axis][1], global_clusters[axis][2], embeddings, text_index)
            for axis in ("understanding", "willingness")
        }
        soft_priors = {
            top_k: {
                "understanding": A.soft_knn_success_prior(query, "understanding", A.CELLS,
                    global_clusters["understanding"][2], embeddings, text_index, reference_harmless, top_k),
                "willingness": A.soft_knn_success_prior(query, "willingness", A.FRAMES,
                    global_clusters["willingness"][2], embeddings, text_index, reference_harmless, top_k),
            }
            for top_k in (8, 16, 32)
        }
        for top_k in (8, 16, 32, 64):
            soft_priors[f"full_{top_k}"] = {
                "understanding": A.soft_knn_full_evidence_prior(query, "understanding", A.CELLS,
                    global_clusters["understanding"][2], embeddings, text_index, reference_harmless, top_k),
                "willingness": A.soft_knn_full_evidence_prior(query, "willingness", A.FRAMES,
                    global_clusters["understanding"][2], embeddings, text_index, reference_harmless, top_k),
            }
        full_reference = {
            top_k: {
                "understanding": A.soft_knn_full_evidence_reference_priors(
                    query, "understanding", A.CELLS, global_clusters["understanding"][2],
                    embeddings, text_index, reference_harmless, top_k),
                "willingness": A.soft_knn_full_evidence_reference_priors(
                    query, "willingness", A.FRAMES, global_clusters["understanding"][2],
                    embeddings, text_index, reference_harmless, top_k),
            }
            for top_k in (16, 32)
        }
        route_cache[item] = (query, routed, scores, soft_priors, full_reference)
    cases = []
    for tag in data["tags"]:
        model = tag.rsplit("_", 1)[0]
        for item_index, item in enumerate(data["items"]):
            verified = np.array([data["matrix"][tag][arm][item_index] for arm in arms], dtype=bool)
            states = [states_payload["states"][tag][arm][item_index] for arm in arms]
            query, routed, route_scores, soft_priors, full_reference = route_cache[item]
            cases.append({"split": split_name(item), "tag": tag, "item": item, "verified": verified,
                          "states": states, "query": query, "routed": routed,
                          "route_scores": route_scores,
                          "soft_knn_priors": soft_priors,
                          "full_reference_priors": full_reference,
                          "harmless": harmless_by_model[model]})
    shared = dict(arms=arms, cluster_data=global_clusters, global_prior=global_prior,
                  embeddings=embeddings, text_index=text_index, reference_harmless=reference_harmless,
                  reference_global_priors=reference_global)
    return cases, shared


def evaluate(cases, shared, method: str, policy: dict, budget: int, seed_count: int = 1):
    records = []; oracle = []
    for case in cases:
        oracle.append(float(case["verified"].any()))
        seeds = seed_count if method == "B0_random_harmful" else 1
        for seed in range(seeds):
            rng = np.random.default_rng(A.stable_seed(case["tag"], case["item"], method, seed, json.dumps(policy, sort_keys=True)))
            success, pulls, harmless = A.run_item(
                method, shared["arms"], case["verified"], case["states"], case["query"],
                shared["cluster_data"], case["harmless"], shared["global_prior"],
                shared["embeddings"], shared["text_index"], shared["reference_harmless"],
                case["routed"], shared["reference_global_priors"], budget, rng, policy=policy,
                route_scores=case["route_scores"],
                soft_knn_priors=case["soft_knn_priors"],
                full_reference_priors=case["full_reference_priors"].get(int(policy.get("router_top_k", 16))),
            )
            records.append({"success": success, "pulls": pulls, "harmless": harmless,
                            "state_exact": float(np.mean([state["exact"] for state in case["states"]]))})
    return A.summarize(records, float(np.mean(oracle)), budget)


def main():
    project = HERE.parents[4]; experiment = HERE.parents[1]; suite = HERE.parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", type=Path, default=experiment / "results/separate_axis_clusters")
    parser.add_argument("--candidates", type=int, default=96)
    parser.add_argument("--shortlist", type=int, default=12)
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--output", type=Path, default=experiment / "results/separate_axis_pipeline_search.json")
    args = parser.parse_args()
    output = {"schema": "poly_separate_axis_pipeline_search/v1", "split": "SHA256 item-level 50/25/25", "datasets": {}}
    grid = candidates(args.candidates)
    prepared = {}
    selection_by_dataset = {}
    for dataset in ("mj", "lg"):
        cases, shared = prepare(project, experiment, suite, args.clusters, dataset)
        partitions = {name: [case for case in cases if case["split"] == name] for name in ("selection", "validation", "test")}
        selection = []
        for index, policy in enumerate(grid):
            summary = evaluate(partitions["selection"], shared, policy["method"], policy, args.budget)
            selection.append({"candidate": index, "policy": policy, "summary": summary})
        selection.sort(key=lambda row: better(row["summary"]), reverse=True)
        prepared[dataset] = (partitions, shared)
        selection_by_dataset[dataset] = {row["candidate"]: row for row in selection}
        validation = []
        for row in selection[:args.shortlist]:
            summary = evaluate(partitions["validation"], shared, row["policy"]["method"], row["policy"], args.budget)
            validation.append({"candidate": row["candidate"], "policy": row["policy"], "summary": summary})
        validation.sort(key=lambda row: better(row["summary"]), reverse=True)
        chosen = validation[0]
        test = evaluate(partitions["test"], shared, chosen["policy"]["method"], chosen["policy"], args.budget)
        baselines = {
            method: evaluate(partitions["test"], shared, method, {}, args.budget, seed_count=20)
            for method in ("B0_random_harmful", "B1_priorless_joint_gp_ucb_harmful", "B2_priorless_separate_axis_harmful")
        }
        output["datasets"][dataset] = {
            "item_counts": {name: len({case["item"] for case in rows}) for name, rows in partitions.items()},
            "candidate_count": len(grid), "selection_top": selection[:args.shortlist],
            "validation_ranking": validation, "chosen_policy": chosen["policy"],
            "heldout_test": test, "heldout_baselines": baselines,
        }
        print(dataset, "chosen", json.dumps(chosen["policy"], sort_keys=True), "test", test)
    global_selection = []
    for index, policy in enumerate(grid):
        summaries = [selection_by_dataset[dataset][index]["summary"] for dataset in ("mj", "lg")]
        global_selection.append({"candidate": index, "policy": policy, "summary": macro_summary(summaries)})
    global_selection.sort(key=lambda row: better(row["summary"]), reverse=True)
    global_validation = []
    for row in global_selection[:args.shortlist]:
        summaries = []
        for dataset in ("mj", "lg"):
            partitions, shared = prepared[dataset]
            summaries.append(evaluate(partitions["validation"], shared, row["policy"]["method"], row["policy"], args.budget))
        global_validation.append({"candidate": row["candidate"], "policy": row["policy"],
                                  "summary": macro_summary(summaries),
                                  "per_dataset_validation": dict(zip(("mj", "lg"), summaries))})
    global_validation.sort(key=lambda row: better(row["summary"]), reverse=True)
    frozen = global_validation[0]
    global_test = {}
    for dataset in ("mj", "lg"):
        partitions, shared = prepared[dataset]
        global_test[dataset] = evaluate(partitions["test"], shared, frozen["policy"]["method"],
                                        frozen["policy"], args.budget)
    output["global_policy"] = {
        "selection_top": global_selection[:args.shortlist],
        "validation_ranking": global_validation,
        "chosen_policy": frozen["policy"],
        "heldout_test_by_dataset": global_test,
    }
    print("global chosen", json.dumps(frozen["policy"], sort_keys=True), "test", global_test)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
