#!/usr/bin/env python3
"""Train the configuration selector on the JOINT reconstruction-gated ASR reward.

Reuses the exact policy / PPO / rollout / evaluation machinery from
train_selector_gpu.py, but the reward tensor is J = R AND U (reconstruction-gated ASR)
built by build_joint_selector_data.py, and holdout is per-TARGET leave-one-out because
this joint pilot has only three families (qwen, phi, falcon).

Baselines share the same candidate set, prior, calibration budget and evaluation items:
  fixed  random  gp_ucb  one_shot  supervised  supervised_continued  ppo  no_feedback
Final numbers are joint gated ASR on the held-out target's held-out (test) items.
No target-model or judge calls happen here; this trains a small selector on the
already-measured joint table.
"""
from __future__ import annotations
import argparse, copy, json, hashlib, time
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_selector_gpu import (Policy, supervised_updates, ppo_updates, evaluate, write_json)  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("results/joint_selector_train_20260903"))
    ap.add_argument("--outdir", type=Path, default=Path("results/joint_selector_run_20260903"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    ap.add_argument("--supervised-steps", type=int, default=500)
    ap.add_argument("--ppo-updates", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--episodes", type=int, default=128)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA; no silent CPU fallback")
    a.outdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_float32_matmul_precision("high")

    manifest = json.loads((a.data / "manifest.json").read_text())
    blob = np.load(a.data / "joint_values.npz", allow_pickle=True)
    J = blob["J"]; feats = blob["features"]; split_labels = blob["splits"]
    targets = manifest["targets"]; families = [t["family"] for t in targets]
    T = len(targets)
    values = torch.tensor(J, device="cuda")
    features = torch.tensor(feats, device="cuda")
    splits = {s: torch.tensor(np.where(split_labels == s)[0], device="cuda")
              for s in ("train", "validation", "test")}

    run_config = dict(reward="joint reconstruction-gated ASR (R AND U); per-item 0/1",
                      holdout="per-target leave-one-out (3 families only)",
                      targets=[t["tag"] for t in targets], families=sorted(set(families)),
                      dataset_shape=list(J.shape), item_splits=manifest["item_splits"],
                      anchors=manifest["anchors"], support_items=8, terminal_reward_items=32,
                      supervised_optimizer_updates=a.supervised_steps,
                      ppo_optimizer_updates=a.ppo_updates * 4, seeds=a.seeds,
                      device=torch.cuda.get_device_name(), torch_version=str(torch.__version__),
                      code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(a.outdir / "config.json", run_config)

    records, validation, curves = [], [], []
    started = time.monotonic()
    for held in range(T):
        test_ids = [held]
        val_target = (held + 1) % T
        val_ids = [val_target]
        train_ids = [i for i in range(T) if i not in (held, val_target)]
        # each training target is its own balanced-sampling group
        groups = [torch.tensor([i], device="cuda") for i in train_ids]
        prior = torch.stack([values[g][:, splits["train"]].mean((0, 1)) for g in groups]).mean(0)
        for seed in a.seeds:
            print(json.dumps({"stage": "fold_start", "held_out": targets[held]["tag"],
                              "validation": targets[val_target]["tag"], "seed": seed}), flush=True)
            torch.manual_seed(seed)
            model = Policy(features, prior).cuda()
            initial = {k: v.clone() for k, v in model.state_dict().items()}
            s_curve = supervised_updates(model, values, groups, splits["train"], a.supervised_steps, a.batch_size, seed)
            supervised = copy.deepcopy(model).eval()
            r_curve = ppo_updates(model, values, groups, splits["train"], a.ppo_updates, a.batch_size, seed + 10000)
            rl = model.eval()
            continued = copy.deepcopy(supervised)
            c_curve = supervised_updates(continued, values, groups, splits["train"], a.ppo_updates * 4,
                                         a.batch_size, seed + 20000, learning_rate=1e-4)
            policies = {"fixed": supervised, "random": supervised, "gp_ucb": supervised,
                        "one_shot": supervised, "supervised": supervised,
                        "supervised_continued": continued, "ppo": rl, "no_feedback": rl}
            val = evaluate(policies, values, val_ids, splits["train"], splits["validation"], targets, a.episodes, 6700)
            test = evaluate(policies, values, test_ids, splits["train"], splits["test"], targets, a.episodes, 8900)
            for dest, rows in ((validation, val), (records, test)):
                for r in rows:
                    r.update(held_out=targets[held]["tag"], validation=targets[val_target]["tag"], seed=seed)
                    dest.append(r)
            assert any(not torch.equal(initial[k], rl.state_dict()[k]) for k in initial)
            curves.append({"held_out": targets[held]["tag"], "seed": seed,
                           "supervised": s_curve, "ppo": r_curve, "continued": c_curve})
            write_json(a.outdir / "test_results.json", records)
            write_json(a.outdir / "validation.json", validation)
            write_json(a.outdir / "training_curves.json", curves)

    # aggregate: joint gated ASR by method x budget over held-out targets (mean of seeds per target)
    def agg(rows):
        out = {}
        keys = sorted({(r["method"], r["budget"]) for r in rows})
        for method, budget in keys:
            by_t = {}
            for r in rows:
                if r["method"] == method and r["budget"] == budget:
                    by_t.setdefault(r["held_out"], []).append(r["mean_proxy"])
            per_t = np.array([np.mean(v) for v in by_t.values()])
            regret = {}
            reg_rows = [r for r in rows if r["method"] == method and r["budget"] == budget]
            reg_by_t = {}
            for r in reg_rows:
                reg_by_t.setdefault(r["held_out"], []).append(r["mean_grid_regret"])
            per_t_reg = np.array([np.mean(v) for v in reg_by_t.values()])
            b = np.array([np.random.default_rng(s).choice(per_t, len(per_t)).mean() for s in range(2000)])
            out[f"{method}@{budget}"] = dict(gated=float(per_t.mean()),
                                             ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                                             mean_regret_vs_target_oracle=float(per_t_reg.mean()),
                                             n_targets=len(per_t))
        return out
    summary = dict(scope="Joint reconstruction-gated ASR selector, per-target LOO.",
                   reward=run_config["reward"], targets=[t["tag"] for t in targets],
                   test=agg(records), validation=agg(validation),
                   elapsed_seconds=time.monotonic() - started)
    write_json(a.outdir / "summary.json", summary)
    print("\n== TEST: joint gated ASR on held-out target's test items (mean over 5 LOO targets) ==")
    for budget in (0, 2, 4, 8, 16):
        line = []
        for m in ("fixed", "random", "gp_ucb", "one_shot", "supervised", "ppo", "no_feedback"):
            v = summary["test"].get(f"{m}@{budget}")
            if v: line.append(f"{m}={v['gated']:.3f}")
        print(f"  budget={budget:2d}  " + "  ".join(line))
    print(json.dumps({"elapsed_seconds": round(summary["elapsed_seconds"], 1)}))


if __name__ == "__main__":
    main()
