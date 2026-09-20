#!/usr/bin/env python3
"""Hybrid replay: free repeated harmless probes, then optional costly harmful confirmation."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from replay_harmful_confirmation import ARM, BETA, SIGMA, features, gp_posterior
from harmless_willingness_replay import FRAMES, load_model


CELL_FRAME = re.compile(r"^(g(?:3|5|8|12)_(?:ordered|shuffled)_n2)__(plain|persona|fiction)$")


def harmless_posterior(matrix, train, rng, mode):
    counts = np.zeros(len(FRAMES)); success = np.zeros(len(FRAMES)); cursor = np.zeros(len(FRAMES), int)
    queues = [rng.permutation(train) for _ in FRAMES]
    local_candidates = None
    if mode.startswith("fixed"):
        maximum = int(mode.removeprefix("fixed")); patience = None; epsilon = None; local_budget = None
    else:
        parts = mode.split("_")
        patience = int(parts[1].removeprefix("p")); epsilon = float(parts[2].removeprefix("e"))
        local_budget = int(parts[3].removeprefix("local")) if len(parts) == 4 else None
        maximum = local_budget or 80
    best_history, value_history = [], []
    for step in range(maximum):
        allowed = np.arange(len(FRAMES)) if local_candidates is None else local_candidates
        unseen = allowed[counts[allowed] == 0]
        if len(unseen):
            arm = int(unseen[0])
        else:
            draws = rng.beta(success[allowed] + 1, counts[allowed] - success[allowed] + 1)
            arm = int(allowed[np.argmax(draws)])
        item = queues[arm][cursor[arm] % len(train)]; cursor[arm] += 1
        counts[arm] += 1; success[arm] += matrix[item, arm]
        posterior = (success + 1) / (counts + 2)
        best_history.append(int(np.argmax(posterior))); value_history.append(float(np.max(posterior)))
        if patience and local_candidates is None and step + 1 >= max(10, patience):
            stable = len(set(best_history[-patience:])) == 1
            plateau = abs(value_history[-1] - value_history[-patience]) <= epsilon
            if stable and plateau:
                if local_budget is None:
                    break
                # Harmless queries are free: keep the two strongest posterior arms and
                # spend the remaining harmless budget locally before any harmful check.
                local_candidates = np.argsort(posterior)[-2:]
    return (success + 1) / (counts + 2), int(counts.sum())


def arm_prior(arms, benign, willingness):
    cells = benign.get("benign_recon_by_cell", {})
    fallback = float(np.mean(list(cells.values()))) if cells else 0.5
    values = []
    for arm in arms:
        cell, frame = CELL_FRAME.fullmatch(arm).groups()
        values.append(cells.get(cell, fallback) * willingness[FRAMES.index(frame)])
    return np.array(values, float)


def harmful_refine(arms, feats, prior, calibration, rng, mode):
    if mode == "none": return int(np.argmax(prior)), 0
    adaptive = mode.startswith("adaptive")
    maximum = int(re.search(r"\d+$", mode).group())
    queried, outcomes = [], []
    mean0 = 0.6 * prior
    for step in range(maximum):
        if queried:
            mean, sd = gp_posterior(feats[queried], feats, np.array(outcomes), mean0[queried], mean0)
        else:
            mean, sd = mean0.copy(), np.full(len(arms), SIGMA)
        acquisition = mean + BETA * sd; acquisition[queried] = -np.inf
        nxt = int(np.argmax(acquisition)); queried.append(nxt); outcomes.append(float(calibration[nxt]))
        if adaptive and step + 1 >= 1:
            order = np.argsort(mean)[::-1]
            confident = mean[order[0]] - mean[order[1]] >= 0.10
            succeeded = max(outcomes) >= 0.50
            if confident or succeeded: break
    # BAI recommendation follows the repository's frozen method: the harmful
    # observations correct the harmless warm-start residual, then we recommend
    # the posterior-mean maximizer (which may be an unqueried neighbour).
    mean, _ = gp_posterior(feats[queried], feats, np.array(outcomes), mean0[queried], mean0)
    return int(np.argmax(mean)), len(queried)


def main():
    here = Path(__file__).resolve(); experiment = here.parents[1]; suite = here.parents[2]; project = here.parents[4]
    panel_root = project / "workspaces/prior_axes_mj_lg_20260920/01_harmless331/willingness_prior_fixed_g3_ordered_n2_20260920T123409Z/run/panel"
    harmless = {}
    for path in panel_root.iterdir():
        if not (path / "responses.jsonl").exists():
            continue
        loaded = load_model(path)
        if len(loaded[0]) >= 20:
            harmless[path.name] = loaded
    benign_root = suite / "exp02_panel_collect/results/benign"
    harmless_modes = (
        "fixed20", "fixed40", "fixed80",
        "plateau_p5_e0.01", "plateau_p10_e0.01", "plateau_p10_e0.03",
        "plateau_p10_e0.01_local40", "plateau_p10_e0.01_local80",
    )
    harmful_modes = ("none", "fixed3", "fixed5", "fixed8", "adaptive3", "adaptive5", "adaptive8")
    output = {}
    for dataset in ("mj", "lg"):
        data = json.loads((suite / f"exp07_panel17_selector_replay/results/{dataset}_item_matrix.json").read_text())
        tags = [tag for tag in data["tags"] if tag.rsplit("_", 1)[0] in harmless and (benign_root / f"{tag}.json").exists()]
        arms = sorted(arm for arm in data["matrix"][tags[0]] if ARM.fullmatch(arm)); feats = np.array([features(a) for a in arms])
        records = {f"{h}+{d}": [] for h in harmless_modes for d in harmful_modes}
        hcounts = {key: [] for key in records}; dcounts = {key: [] for key in records}
        for split in range(40):
            ids = np.random.default_rng(30000 + split).permutation(len(data["items"])); cal_ids, test_ids = ids[:len(ids)//2], ids[len(ids)//2:]
            for ti, tag in enumerate(tags):
                base = tag.rsplit("_", 1)[0]; items, hmatrix = harmless[base]
                hids = np.random.default_rng(40000 + 100 * split + ti).permutation(len(items)); htrain = hids[:len(hids)//2]
                benign = json.loads((benign_root / f"{tag}.json").read_text())
                calibration = np.array([np.mean([data["matrix"][tag][arm][i] for i in cal_ids]) for arm in arms])
                test = np.array([np.mean([data["matrix"][tag][arm][i] for i in test_ids]) for arm in arms])
                for hi, hmode in enumerate(harmless_modes):
                    rng = np.random.default_rng(500000 + 10000 * split + 100 * ti + hi)
                    willingness, harmless_n = harmless_posterior(hmatrix, htrain, rng, hmode)
                    prior = arm_prior(arms, benign, willingness)
                    for di, dmode in enumerate(harmful_modes):
                        picked, harmful_n = harmful_refine(arms, feats, prior, calibration, rng, dmode)
                        key = f"{hmode}+{dmode}"; records[key].append(float(test[picked])); hcounts[key].append(harmless_n); dcounts[key].append(harmful_n)
        rows = [{"method": key, "verified": float(np.mean(values)), "harmless_queries": float(np.mean(hcounts[key])),
                 "harmful_batches": float(np.mean(dcounts[key]))} for key, values in records.items()]
        # Pareto: no other row is at least as accurate with no more harmful batches.
        for row in rows:
            row["pareto"] = not any(other["verified"] >= row["verified"] and other["harmful_batches"] <= row["harmful_batches"]
                                      and (other["verified"] > row["verified"] or other["harmful_batches"] < row["harmful_batches"])
                                      for other in rows)
        rows.sort(key=lambda row: (row["verified"], -row["harmful_batches"]), reverse=True)
        output[dataset] = {"models": tags, "ranking": rows}
    out = experiment / "results/hybrid_free_harmless_then_harmful.json"; out.write_text(json.dumps(output, indent=2) + "\n")
    for ds, result in output.items():
        print(ds.upper())
        for row in result["ranking"][:12]: print(row)


if __name__ == "__main__": main()
