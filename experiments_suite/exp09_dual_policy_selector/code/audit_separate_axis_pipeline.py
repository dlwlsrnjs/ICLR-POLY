#!/usr/bin/env python3
"""Mechanical integrity audit for the separate-axis cluster and replay artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip(): yield json.loads(line)


def split_name(item):
    value = int.from_bytes(hashlib.sha256(("poly-exp09-split-v1|" + item).encode()).digest()[:8], "big") % 100
    return "selection" if value < 50 else "validation" if value < 75 else "test"


def main():
    here = Path(__file__).resolve(); experiment = here.parents[1]; suite = here.parents[2]
    clusters = experiment / "results/separate_axis_clusters"
    index = json.loads((clusters / "INDEX.json").read_text())
    models = [row["model"] for row in index["models"]]
    checks, failures = [], []
    def check(name, ok, detail):
        row = {"check": name, "ok": bool(ok), "detail": detail}; checks.append(row)
        if not ok: failures.append(row)
    texts = json.loads((clusters / "canonical_bge_texts.json").read_text())
    embeddings = np.load(clusters / "canonical_bge_embeddings.npz")["embeddings"]
    query_embeddings = np.load(clusters / "query_bge_embeddings.npz")["embeddings"]
    query_index = json.loads((clusters / "query_embedding_index.json").read_text())
    check("canonical_embedding_rows", len(texts) == len(embeddings), f"{len(texts)} texts / {len(embeddings)} vectors")
    check("canonical_embedding_norm", np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-4), "L2 norms")
    check("query_embedding_rows", len(query_index) == len(query_embeddings), f"{len(query_index)} queries / {len(query_embeddings)} vectors")
    check("query_embedding_norm", np.allclose(np.linalg.norm(query_embeddings, axis=1), 1, atol=1e-4), "L2 norms")
    for model in [*models, "global_anonymous"]:
        for axis in ("understanding", "willingness"):
            root = clusters / model / axis
            payload = json.loads((root / "clusters.json").read_text())
            centroids = np.load(root / "centroids.npy")
            assignments = list(jsonl(root / "assignments.jsonl"))
            ids = [row["item_id"] for row in assignments]
            check(f"{model}/{axis}/centroid_norm", np.allclose(np.linalg.norm(centroids, axis=1), 1, atol=1e-4), f"k={len(centroids)}")
            check(f"{model}/{axis}/nonempty", all(row["items"] > 0 for row in payload["clusters"]), f"assignments={len(assignments)}")
            check(f"{model}/{axis}/unique_item_assignment", len(ids) == len(set(ids)), f"unique={len(set(ids))}")
            check(f"{model}/{axis}/canonical_only", all(set(row) <= {"model","axis","item_id","canonical_text","cluster","centroid_cosine"} for row in assignments), "no prompt/response fields")
    check("no_joint_artifacts", not any("joint" in path.name.lower() for path in clusters.rglob("*") if path.is_file()), "no joint-named files")
    for dataset in ("mj", "lg"):
        state = json.loads((experiment / f"results/{dataset}_response_state_matrix.json").read_text())
        check(f"{dataset}/state_count", 0 <= state["exact_state_rows"] <= state["total_rows"], f"{state['exact_state_rows']}/{state['total_rows']}")
        data = json.loads((suite / f"exp07_panel17_selector_replay/results/{dataset}_item_matrix.json").read_text())
        parts = {name: {item for item in data["items"] if split_name(item) == name} for name in ("selection","validation","test")}
        disjoint = not (parts["selection"] & parts["validation"] or parts["selection"] & parts["test"] or parts["validation"] & parts["test"])
        check(f"{dataset}/split_disjoint", disjoint and sum(map(len, parts.values())) == len(data["items"]), {k: len(v) for k,v in parts.items()})
    search = json.loads((experiment / "results/separate_axis_pipeline_search.json").read_text())
    check("search_has_heldout_test", all("heldout_test" in row for row in search["datasets"].values()), list(search["datasets"]))
    check("search_has_global_frozen_policy",
          "global_policy" in search and set(search["global_policy"].get("heldout_test_by_dataset", {})) == {"mj", "lg"},
          search.get("global_policy", {}).get("chosen_policy", {}).get("method"))
    full_config = json.loads((experiment / "config/full_evidence_two_axis_online.json").read_text())
    contract = full_config["separation_contract"]
    check("full_evidence_no_joint_state",
          not any(contract[key] for key in ("joint_prior", "joint_posterior", "joint_cluster", "joint_gp")),
          contract)
    final_config = json.loads((experiment / "config/final_p5_response_state_dynamic.json").read_text())
    check("canonical_final_is_p5",
          final_config.get("status") == "canonical_final"
          and final_config.get("method") == "P5_response_state_dynamic"
          and final_config.get("joint_prior") is False
          and final_config.get("joint_posterior") is False,
          {key: final_config.get(key) for key in ("status", "method", "joint_prior", "joint_posterior")})
    evidence = json.loads((experiment / "results/full_evidence_prior_audit.json").read_text())["totals"]
    u = evidence["understanding"]; w = evidence["willingness"]
    check("understanding_evidence_partition",
          u["rows"] == u["valid_success"] + u["valid_failure"] + u["invalid"], u)
    check("willingness_evidence_partition",
          w["rows"] == w["eligible_r1"] + w["reconstruction_failure"] + w["invalid_or_missing_judge"]
          and w["eligible_r1"] == w["valid_success"] + w["valid_failure"], w)
    output = {"schema": "poly_separate_axis_pipeline_audit/v1", "passed": not failures,
              "checks": checks, "failures": failures}
    (experiment / "results/separate_axis_pipeline_audit.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    lines = ["# Separate-axis pipeline audit", "", f"Overall: **{'PASS' if not failures else 'FAIL'}**", "", "| Check | Status | Detail |", "|---|---|---|"]
    for row in checks:
        lines.append(f"| `{row['check']}` | {'PASS' if row['ok'] else 'FAIL'} | `{row['detail']}` |")
    (experiment / "results/separate_axis_pipeline_audit.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"passed": not failures, "checks": len(checks), "failures": len(failures)}))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__": main()
