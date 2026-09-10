#!/usr/bin/env python3
"""Train and evaluate an explicitly benign finite-grid selector pilot on CPU."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from benign_selector import (
    BUDGETS, CONDITIONS, INITIAL, Selector, aggregate, digest, evaluate,
    fit_rl, fit_supervised, load_corpus, split_items, validate_split,
)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def make_report(path, summary):
    config = summary["config"]
    lines = [
        "# Benign adaptive selector pilot", "",
        "Status: completed CPU experiment on previously recorded benign FLORES outputs.",
        "This is not a new target-model run, expanded-space validation, or jailbreak-policy result.", "",
        "## Protocol", "",
        "- Five leave-one-target-out folds; source items split 20 train/calibration, 8 validation, 12 test.",
        "- Training uses four target systems only. The excluded target is observed only through counted calibration queries.",
        "- Calibration and terminal training reward items are disjoint within every training episode.",
        "- Every evaluated method uses the same calibration support per episode.",
        "- Initial probe: two fixed configurations, four items each (8 replayed target calls).",
        "- Additional budgets: 0, 1, 2, 4, 8 configurations; each costs four replayed target calls.",
        "- Full grid diagnostic is retrospective and unavailable to the policy.",
        "- Target models were not called during this experiment. Calls below are replayed calibration cost, not new API calls.",
        "- RL: terminal-reward actor-critic, initialized by supervised training; fixed horizon, no learned stopping.",
        "- RL also retains an auxiliary supervised reconstruction prediction loss.",
        "- Negative-feedback ablation hides post-initial-probe outcomes at inference; it is not a separately retrained policy.",
        "- Fixed and one-shot methods may use less than the permitted budget; their actual costs are shown.", "",
        "## Results", "",
        "| Method | Extra configuration budget | Replayed item calls | Test reconstruction | Target bootstrap 95% | Grid regret |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for method, by_budget in summary["results"].items():
        for budget, entry in by_budget.items():
            lo, hi = entry["target_cluster_bootstrap95"]
            lines.append(
                f"| {method} | {budget} | {entry['mean_replayed_calibration_item_calls']:.0f} | "
                f"{entry['macro_test_reconstruction']:.4f} | [{lo:.4f}, {hi:.4f}] | "
                f"{entry['mean_test_grid_regret']:.4f} |")
    lines += [
        "", "## Interpretation and limits", "",
        "- Report all seeds, budgets, and targets; no held-out target was used to select model weights or training steps.",
        "- Bootstrap resamples the five target systems, after averaging episodes and seeds within a target.",
        "- These intervals do not include new-item, judge, or model-population uncertainty; only five target systems are available.",
        "- The 40 source items were already used in earlier research. This computational split is not a fresh independent confirmation dataset.",
        "- All 400 item/configuration keys have at least one prompt difference across targets. Target and prompt-template effects are confounded.",
        "- Ten recorded configurations are supported. No outcomes were imputed for unseen language sets or permutations.",
        "- Fidelity-only benign reward may favor easy configurations; success here would not establish an optimal safety-evaluation policy.",
        "- The initial benign-probe results are part of the charged calibration budget, not free target features.",
        "- Calibration item reuse across configurations permits paired comparison, not independent new measurements.",
        "- Fixed decoding observations have no fresh response variance in replay.",
        "- Actor-critic updates on resampled finite tables do not create new independent targets or real historical trajectories.", "",
        "## Reproduce", "",
        "Interpreter: " + summary["runtime"]["python_executable"],
        "Command:", "",
        "    " + summary["command"], "",
        "Outputs include input hashes, split IDs, all episode traces, validation diagnostics, training curves, and model checkpoints.",
        "No raw prompts or responses are copied to the output directory.", "",
        f"Wall time: {summary['elapsed_seconds']:.1f} seconds.",
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    ap.add_argument("--pretrain-steps", type=int, default=200)
    ap.add_argument("--rl-steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    if min(args.pretrain_steps, args.rl_steps, args.batch_size, args.episodes, args.threads) < 1:
        ap.error("Training sizes and thread count must be positive")
    if args.outdir.exists():
        ap.error("Use a fresh output directory; experiments are not overwritten")
    args.outdir.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    began = time.monotonic()
    corpus = load_corpus(args.root)
    split = split_items(len(corpus.items))
    validate_split(split)
    config = {
        "seeds": args.seeds, "pretrain_steps": args.pretrain_steps,
        "rl_steps": args.rl_steps, "batch_size": args.batch_size,
        "evaluation_episodes_per_fold_seed_budget": args.episodes, "threads": args.threads,
        "support_items": 4, "initial_arms": list(INITIAL), "budgets": list(BUDGETS),
        "split_seed": 20260903, "selection": "fixed training schedule; no test-based model selection",
        "reward": "held-apart benign training reconstruction rate; terminal only; fixed horizon",
        "training_algorithm": "supervised warm start + actor-critic with auxiliary supervised loss",
        "new_target_calls": 0, "new_judge_calls": 0, "new_gpu_work": False,
    }
    manifest = dict(corpus.manifest)
    manifest["split"] = {name: [corpus.items[int(i)] for i in ids] for name, ids in split.items()}
    manifest["code_sha256"] = {p.name: digest(p) for p in
                               [Path(__file__), Path(__file__).with_name("benign_selector.py")]}
    write_json(args.outdir / "manifest.json", manifest)
    write_json(args.outdir / "config.json", config)
    all_records, validation, learning = [], [], []
    for heldout, name in enumerate(corpus.models):
        train_models = [m for m in range(len(corpus.models)) if m != heldout]
        prior = corpus.values[train_models][:, split["train_calibration"]].mean((0, 1))
        for seed in args.seeds:
            print(f"START heldout={name} seed={seed}", flush=True)
            start = time.monotonic()
            supervised, history_s = fit_supervised(
                corpus.values, train_models, split["train_calibration"], prior, seed,
                steps=args.pretrain_steps, batch=args.batch_size)
            policy, history_r = fit_rl(
                supervised, corpus.values, train_models, split["train_calibration"], prior, seed,
                steps=args.rl_steps, batch=args.batch_size)
            methods = {"train_fixed": None, "supervised_once": supervised, "random": None,
                       "gp_ucb": None, "supervised_sequential": supervised,
                       "rl_history": policy, "rl_no_feedback": policy}
            records = evaluate(
                corpus.values, heldout, split["test"], split["train_calibration"], prior,
                methods, BUDGETS, args.episodes, seed=8900+heldout*10000)
            for record in records:
                record.update({"heldout_model": name, "training_seed": seed})
                record["calibration_item_ids"] = [corpus.items[i] for i in record["calibration_item_ids"]]
            all_records.extend(records)
            # Diagnostics on training targets + validation items; never choose a checkpoint on test.
            val = []
            for train_model in train_models:
                vr = evaluate(corpus.values, train_model, split["validation"], split["train_calibration"], prior,
                              {"supervised_once": supervised, "rl_history": policy}, [4],
                              min(args.episodes, 16), seed=6700+train_model*10000)
                val.extend(vr)
            validation.append({"heldout_model": name, "seed": seed,
                               "mean_by_method": {method: float(np.mean([r["test_reconstruction"] for r in val if r["method"] == method]))
                                                  for method in ("supervised_once", "rl_history")},
                               "used_for_checkpoint_selection": False})
            learning.append({"heldout_model": name, "seed": seed, "supervised": history_s, "rl": history_r})
            torch.save({"supervised": supervised.state_dict(), "rl": policy.state_dict(),
                        "prior": torch.from_numpy(prior), "conditions": CONDITIONS,
                        "heldout_model": name, "seed": seed, "config": config},
                       args.outdir / f"fold{heldout}_seed{seed}.pt")
            write_json(args.outdir / "progress.json",
                       {"completed_fold_seeds": len(learning), "planned": len(corpus.models)*len(args.seeds)})
            print(f"DONE heldout={name} seed={seed} seconds={time.monotonic()-start:.1f}", flush=True)
    summary = {
        "config": config, "results": aggregate(all_records),
        "elapsed_seconds": time.monotonic()-began,
        "runtime": {"python": platform.python_version(), "python_executable": sys.executable,
                    "numpy": np.__version__, "torch": torch.__version__, "device": "cpu"},
        "command": " ".join([sys.executable, *sys.argv]),
        "status": "completed", "scope": "benign recorded-grid pilot",
    }
    write_json(args.outdir / "summary.json", summary)
    write_json(args.outdir / "validation.json", validation)
    write_json(args.outdir / "training_curves.json", learning)
    with (args.outdir / "episodes.jsonl").open("w") as handle:
        for row in all_records:
            handle.write(json.dumps(row) + "\n")
    make_report(args.outdir / "REPORT.md", summary)
    print(json.dumps({"status": "completed", "output": str(args.outdir),
                      "seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()

