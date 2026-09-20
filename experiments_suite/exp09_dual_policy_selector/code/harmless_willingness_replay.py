#!/usr/bin/env python3
"""Repeated harmless willingness probes on the completed 92daeab five-frame panel."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


FRAMES = ("plain", "persona", "fiction", "pap", "persona+fiction")


def load_model(path: Path):
    responses = {row["key"]: row for row in map(json.loads, path.joinpath("responses.jsonl").open())}
    recon = {row["key"]: row for row in map(json.loads, path.joinpath("judge/reconstruction/restricted_reconstruction_audit.jsonl").open())}
    guard = {row["key"]: row for row in map(json.loads, path.joinpath("judge/outputs/wildguard.jsonl").open())}
    values = {}
    for key, response in responses.items():
        r, g = recon.get(key), guard.get(key)
        if not r or not g or not r.get("reconstruction_parse_valid") or not r.get("semantic_reconstruction_equivalent"):
            continue
        refusal = g.get("fields", {}).get("response_refusal", {})
        if not g.get("valid") or not refusal.get("valid"):
            continue
        values[(response["id"], response["frame"])] = float(refusal.get("label") == "no")
    items = sorted({item for item, _ in values if all((item, frame) in values for frame in FRAMES)})
    return items, np.array([[values[(item, frame)] for frame in FRAMES] for item in items])


def choose(strategy, observations, counts, successes, rng, step):
    unseen = np.flatnonzero(counts == 0)
    if len(unseen): return int(unseen[step % len(unseen)])
    mean = successes / counts
    if strategy == "round_robin": return step % len(FRAMES)
    if strategy == "ucb1": return int(np.argmax(mean + np.sqrt(2 * np.log(step + 1) / counts)))
    if strategy == "thompson": return int(np.argmax(rng.beta(successes + 1, counts - successes + 1)))
    if strategy == "epsilon_greedy":
        return int(rng.integers(len(FRAMES))) if rng.random() < 0.15 else int(np.argmax(mean))
    if strategy == "successive_halving":
        active = np.flatnonzero(mean >= np.max(mean) - np.sqrt(2 * np.log(step + 1) / counts))
        return int(active[np.argmin(counts[active])])
    raise ValueError(strategy)


def main():
    here = Path(__file__).resolve(); experiment = here.parents[1]; project = here.parents[4]
    root = project / "workspaces/prior_axes_mj_lg_20260920/01_harmless331/willingness_prior_fixed_g3_ordered_n2_20260920T123409Z/run/panel"
    panels = {path.name: load_model(path) for path in sorted(root.iterdir()) if (path / "responses.jsonl").exists()}
    strategies = ("round_robin", "ucb1", "thompson", "epsilon_greedy", "successive_halving")
    budgets = (5, 10, 20, 40, 80)
    result = {f"{s}@{b}": [] for s in strategies for b in budgets}
    oracle = []
    for model_index, (tag, (items, matrix)) in enumerate(panels.items()):
        if len(items) < 20: continue
        for split in range(40):
            rng = np.random.default_rng(200000 + 1000 * model_index + split)
            order = rng.permutation(len(items)); train, test = order[:len(order)//2], order[len(order)//2:]
            oracle.append(float(matrix[test].mean(0).max()))
            for strategy in strategies:
                for budget in budgets:
                    counts = np.zeros(len(FRAMES)); successes = np.zeros(len(FRAMES)); cursor = np.zeros(len(FRAMES), int)
                    per_frame = {frame: rng.permutation(train) for frame in range(len(FRAMES))}
                    for step in range(budget):
                        arm = choose(strategy, None, counts, successes, rng, step)
                        item = per_frame[arm][cursor[arm] % len(train)]; cursor[arm] += 1
                        counts[arm] += 1; successes[arm] += matrix[item, arm]
                    recommended = int(np.argmax((successes + 1) / (counts + 2)))
                    result[f"{strategy}@{budget}"].append(float(matrix[test, recommended].mean()))
    ranking = sorted(({"method": key, "mean_test_nonrefusal": float(np.mean(values))} for key, values in result.items()),
                     key=lambda row: row["mean_test_nonrefusal"], reverse=True)
    payload = {"schema": "exp09_harmless_willingness_replay_v1", "models": len(panels), "frames": FRAMES,
               "metric": "nonrefusal conditional on valid successful reconstruction", "oracle": float(np.mean(oracle)),
               "ranking": ranking}
    out = experiment / "results/harmless_willingness_replay.json"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print("oracle", payload["oracle"])
    for row in ranking[:15]: print(row["method"], f"{row['mean_test_nonrefusal']:.4f}")


if __name__ == "__main__":
    main()
