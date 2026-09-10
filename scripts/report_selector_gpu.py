#!/usr/bin/env python3
"""Aggregate completed GPU selector folds without treating episodes as targets."""
import argparse
import fcntl
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def summarize(records):
    result = []
    for method in sorted({r["method"] for r in records}):
        for budget in sorted({r["budget"] for r in records}):
            rows = [r for r in records if r["method"] == method and r["budget"] == budget]
            per_target = defaultdict(list)
            target_family = {}
            for row in rows:
                per_target[row["model"]].append(row["mean_proxy"])
                target_family[row["model"]] = row["family"]
            means = {m:float(np.mean(v)) for m,v in per_target.items()}
            by_family = defaultdict(list)
            for model, mean in means.items():
                by_family[target_family[model]].append(mean)
            family_means = {g:float(np.mean(v)) for g,v in by_family.items()}
            values = np.array(list(family_means.values()))
            draws = np.random.default_rng(71).choice(values, size=(5000,len(values)), replace=True).mean(1)
            result.append({"method":method,"budget":budget,
                           "family_macro_proxy":float(values.mean()),
                           "target_macro_proxy":float(np.mean(list(means.values()))),
                           "family_bootstrap95":np.quantile(draws,[.025,.975]).tolist(),
                           "replayed_item_calls":float(np.mean([r["replayed_item_calls"] for r in rows])),
                           "per_family":family_means,"per_target":means})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    a = ap.parse_args()
    report_lock = (a.results/".report.lock").open("a")
    fcntl.flock(report_lock.fileno(), fcntl.LOCK_EX)
    stages = defaultdict(list)
    for folder in a.results.glob("train*_gpu*"):
        progress = folder/"progress.json"
        if progress.exists() and json.loads(progress.read_text()).get("status") == "completed":
            stages[folder.name.split("_gpu")[0]].append(folder)
    all_summaries = {}
    lines = ["# GPU adaptive selector results", "",
             "Only completed training partitions are included. Values are lexical reconstruction proxies, not ASR or semantic-equivalence rates.", ""]
    for stage, folders in sorted(stages.items()):
        records, configurations = [], []
        signatures = set()
        for folder in sorted(folders):
            config = json.loads((folder/"config.json").read_text())
            configurations.append(config)
            for r in json.loads((folder/"test_results.json").read_text()):
                key = (r["fold"],r["training_seed"],r["model"],r["method"],r["budget"])
                if key in signatures:
                    raise ValueError("Repeated completed fold: select intended run rather than double-counting retries")
                signatures.add(key)
                records.append(r)
        summary = summarize(records)
        all_summaries[stage] = {"results":summary,"training_configurations":configurations,
                                "completed_family_folds":len({r["test_family"] for r in records}),
                                "completed_training_seeds":sorted({r["training_seed"] for r in records})}
        lines += [f"## {stage}", "", f"Dataset shape: {configurations[0]['dataset_shape']} (targets × source items × configurations).",
                  f"Completed held-out family folds: {all_summaries[stage]['completed_family_folds']}.", "",
                  "| Method | Extra configurations | Replayed item calls | Family mean | Family bootstrap 95% |",
                  "|---|---:|---:|---:|---|" ]
        for r in summary:
            lo, hi = r["family_bootstrap95"]
            lines.append(f"| {r['method']} | {r['budget']} | {r['replayed_item_calls']:.0f} | {r['family_macro_proxy']:.4f} | [{lo:.4f}, {hi:.4f}] |")
        lines.append("")
    lines += ["## Scope and interpretation", "",
              "- Target models were executed on GPUs to create the observation data. PPO trains the selector; target model weights stay frozen.",
              "- Family-level sampling prevents numerous Qwen sizes from dominating training. Entire target families and source items are held apart.",
              "- Initial calibration costs 2 configurations × 8 items = 16 replayed calls. Every additional configuration costs 8 calls.",
              "- Final performance uses separate test items; calibration and train reward items are disjoint within each training episode.",
              "- Full candidate scores are retrospective diagnostics. Policies only observe paid calibration results.",
              "- The continued-supervised baseline matches PPO optimizer-update count, not state-example count or GPU seconds; these are recorded separately.",
              "- Bootstrap resamples family groups after averaging seeds and checkpoints inside each group. It does not measure new-item, judge, or decoding uncertainty.",
              "- Additional budget 16 extrapolates beyond the training budgets 2/4/8.",
              "- 64 real configurations are supported. Generalization to new unseen compositions is not demonstrated by this experiment.",
              "- Lexical similarity can miss semantic changes and favor easy configurations. This does not validate a reconstruction-gated safety objective.", ""]
    (a.results/"summary.json").write_text(json.dumps(all_summaries,indent=2)+"\n")
    (a.results/"REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"stages":list(all_summaries),"report":str(a.results/"REPORT.md")}),flush=True)


if __name__ == "__main__":
    main()
