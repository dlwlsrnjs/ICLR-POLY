#!/usr/bin/env python3
"""CUDA supervised initialization and PPO on newly measured benign configurations.

The policy chooses the next probe and finally recommends a configuration. Targets
are frozen environments. Family-held-out evaluation and source-item splits are
independent; unqueried support values never enter the policy state.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2)+"\n")
    tmp.replace(path)


def load_data(data, collection, cohort=None):
    manifest = json.loads((data/"manifest.json").read_text())
    items = json.loads((data/"items.json").read_text())
    if cohort is not None:
        selected = set(json.loads(cohort.read_text()))
        items = [r for r in items if r["item_id"] in selected]
    configs = json.loads((data/"configurations.json").read_text())
    item_index = {r["item_id"]: i for i, r in enumerate(items)}
    config_index = {r["config_id"]: i for i, r in enumerate(configs)}
    prompts = {(r["item_id"], r["config_id"]): r["prompt_sha256"]
               for r in map(json.loads, (data/"prompts.jsonl").open())}
    tables, targets, fingerprints, diagnostics = [], [], [], []
    for model_index, target in enumerate(manifest["targets"]):
        directory = collection/f"target_{model_index:02d}"
        progress_path = directory/"progress.json"
        if not progress_path.exists():
            continue
        progress = json.loads(progress_path.read_text())
        if cohort is None and progress["status"] != "completed":
            continue
        table = np.full((len(items), len(configs)), np.nan, np.float32)
        seen, token_input, token_output, truncated, exact = set(), 0, 0, 0, 0
        for path in sorted(directory.glob("chunk_*.jsonl")):
            fingerprint = hashlib.sha256()
            with path.open("rb") as handle:
                for line in handle:
                    fingerprint.update(line)
                    row = json.loads(line)
                    if row["item_id"] not in item_index:
                        continue
                    key = row["item_id"], row["config_id"]
                    if key in seen or key not in prompts:
                        raise ValueError("Duplicate or unknown observation")
                    seen.add(key)
                    if row["target_model"] != target["model_id"] or row["revision"] != target["revision"]:
                        raise ValueError("Mixed target revisions")
                    if row["risk_type"] != "benign" or prompts[key] != row["prompt_sha256"]:
                        raise ValueError("Wrong prompt or source provenance")
                    if hashlib.sha256(row["raw_output"].encode()).hexdigest() != row["response_sha256"]:
                        raise ValueError("Response hash mismatch")
                    i, c = item_index[key[0]], config_index[key[1]]
                    if items[i]["source_sha256"] != row["source_sha256"] or items[i]["split"] != row["item_split"]:
                        raise ValueError("Original source or split mismatch")
                    value = row["reconstruction_proxy"]
                    if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                        raise ValueError("Invalid proxy measurement")
                    table[i, c] = value
                    token_input += row["input_tokens"]
                    token_output += row["output_tokens"]
                    truncated += row["finish_reason"] == "length"
                    exact += row["normalized_exact"]
            fingerprints.append({"path": str(path), "sha256": fingerprint.hexdigest()})
        if not np.isfinite(table).all():
            raise ValueError("Claimed complete target has missing cells; do not impute")
        tables.append(table)
        targets.append(target)
        diagnostics.append({"model": target["model_id"], "family": target["family"],
                            "rows": len(seen), "input_tokens": token_input,
                            "output_tokens": token_output, "truncated": truncated,
                            "normalized_exact": exact, "mean_proxy": float(table.mean())})
    if len({t["family"] for t in targets}) < 4:
        raise ValueError("Need at least four completed family groups for train/validation/test")
    splits = {s: np.array([i for i, r in enumerate(items) if r["split"] == s])
              for s in ("train", "validation", "test")}
    if len(splits["train"]) < 40 or min(len(splits["validation"]), len(splits["test"])) < 16:
        raise ValueError("Insufficient independent source-item partitions")
    return np.stack(tables), targets, configs, items, splits, fingerprints, diagnostics


class Policy(nn.Module):
    """Shared candidate encoder; observed candidates are pooled as target context."""
    def __init__(self, features, prior, width=128):
        super().__init__()
        self.register_buffer("features", features)
        self.register_buffer("prior", prior)
        self.encoder = nn.Sequential(nn.Linear(features.shape[1]+4, width), nn.Tanh(),
                                     nn.Linear(width, width), nn.Tanh())
        self.mix = nn.Sequential(nn.Linear(width*2, width), nn.Tanh())
        self.query = nn.Linear(width, 1)
        self.recommend = nn.Linear(width, 1)
        self.critic = nn.Linear(width, 1)

    def forward(self, obs, mask, remaining):
        b, a = mask.shape
        p = self.prior.expand(b, a)
        # Explicit masking prevents future observations from entering the encoder.
        visible = torch.where(mask.bool(), obs, p)
        x = torch.cat([self.features.expand(b, -1, -1), p[..., None], visible[..., None],
                       mask[..., None], remaining[:, None, None].expand(b, a, 1)/16], -1)
        z = self.encoder(x)
        context = (z * mask[..., None]).sum(1) / mask.sum(1).clamp_min(1)[:, None]
        z = self.mix(torch.cat([z, context[:, None].expand_as(z)], -1))
        return self.query(z).squeeze(-1), self.recommend(z).squeeze(-1), self.critic(context).squeeze(-1)


def sample_episode(values, groups, item_ids, batch, support_size=8, reward_size=32):
    device = values.device
    group_draw = torch.randint(len(groups), (batch,), device=device)
    model_ids = torch.empty(batch, dtype=torch.long, device=device)
    for g, ids in enumerate(groups):
        selected = group_draw == g
        n = int(selected.sum())
        model_ids[selected] = ids[torch.randint(len(ids), (n,), device=device)]
    chosen = torch.rand(batch, len(item_ids), device=device).topk(support_size+reward_size, dim=1).indices
    chosen = item_ids[chosen]
    cells = values[model_ids[:, None], chosen]
    return cells[:, :support_size].mean(1), cells[:, support_size:].mean(1)


def initial_state(support):
    mask = torch.zeros_like(support)
    mask[:, [0, support.shape[1]-1]] = 1
    return support * mask, mask


def supervised_updates(policy, values, groups, items, steps, batch, seed, learning_rate=1e-3):
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    started = time.monotonic()
    curve = []
    for update in range(steps):
        support, reward = sample_episode(values, groups, items, batch)
        obs, mask = initial_state(support)
        budget = (0, 2, 4, 8)[update % 4]
        reveal = update % (budget+1)
        if reveal:
            random_order = torch.rand_like(mask).masked_fill(mask.bool(), -1).topk(reveal, dim=1).indices
            mask.scatter_(1, random_order, 1)
            obs = support * mask
        rem = torch.full((batch,), budget-reveal, device=values.device)
        q, r, _ = policy(obs, mask, rem)
        loss = F.binary_cross_entropy_with_logits(r, reward)
        if reveal < budget:
            available_q = q.masked_fill(mask.bool(), -1e9)
            soft = torch.softmax((reward/.05).masked_fill(mask.bool(), -1e9), dim=1)
            loss = loss - .25*(soft*torch.log_softmax(available_q, 1)).sum(1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1)
        optimizer.step()
        if (update+1) % 250 == 0 or update+1 == steps:
            row = {"stage": "supervised", "step": update+1, "loss": float(loss.detach()),
                   "seconds": time.monotonic()-started}
            curve.append(row)
            print(json.dumps(row), flush=True)
    return curve


def ppo_updates(policy, values, groups, items, updates, batch, seed, epochs=4):
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    device = values.device
    curve = []
    started = time.monotonic()
    for update in range(updates):
        budget = (2, 4, 8)[update % 3]
        support, final_scores = sample_episode(values, groups, items, batch)
        obs, mask = initial_state(support)
        snapshots, actions, old_logps, old_values, kinds = [], [], [], [], []
        with torch.no_grad():
            for step in range(budget+1):
                remaining = torch.full((batch,), budget-step, device=device)
                q, r, v = policy(obs, mask, remaining)
                terminal = step == budget
                distribution = Categorical(logits=r if terminal else q.masked_fill(mask.bool(), -1e9))
                action = distribution.sample()
                snapshots.append((obs.clone(), mask.clone(), remaining))
                actions.append(action)
                old_logps.append(distribution.log_prob(action))
                old_values.append(v)
                kinds.append(torch.full((batch,), terminal, dtype=torch.bool, device=device))
                if not terminal:
                    mask.scatter_(1, action[:, None], 1)
                    obs = support * mask
            reward = final_scores.gather(1, actions[-1][:, None]).squeeze(1)
        obs_all = torch.cat([s[0] for s in snapshots])
        mask_all = torch.cat([s[1] for s in snapshots])
        remaining_all = torch.cat([s[2] for s in snapshots])
        action_all, logp_old, value_old = map(torch.cat, (actions, old_logps, old_values))
        terminal_all = torch.cat(kinds)
        returns = reward.repeat(budget+1)
        advantage = returns - value_old
        advantage = (advantage-advantage.mean())/(advantage.std()+1e-6)
        # Four complete on-policy PPO epochs. No score from test/validation enters updates.
        for _ in range(epochs):
            q, r, value = policy(obs_all, mask_all, remaining_all)
            logits = torch.where(terminal_all[:, None], r, q.masked_fill(mask_all.bool(), -1e9))
            dist = Categorical(logits=logits)
            ratio = (dist.log_prob(action_all)-logp_old).exp()
            clipped = torch.clamp(ratio, .8, 1.2)
            policy_loss = -torch.minimum(ratio*advantage, clipped*advantage).mean()
            critic_loss = F.mse_loss(value, returns)
            auxiliary = F.binary_cross_entropy_with_logits(r[-batch:], final_scores)
            loss = policy_loss + .5*critic_loss + .1*auxiliary - .01*dist.entropy().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1)
            optimizer.step()
        if (update+1) % 100 == 0 or update+1 == updates:
            row = {"stage": "ppo", "update": update+1, "reward": float(reward.mean()),
                   "loss": float(loss.detach()), "extra_config_budget": budget,
                   "approx_kl": float((logp_old-dist.log_prob(action_all)).mean().detach()),
                   "seconds": time.monotonic()-started}
            curve.append(row)
            print(json.dumps(row), flush=True)
    return curve



@torch.no_grad()
def gp_posterior(policy, support, mask):
    """Fixed structural kernel and conservative observation noise; no target fitting."""
    batch, arms = support.shape
    features = policy.features.clone()
    features[:, 10:12] /= .5
    kernel = .04*torch.exp(-.5*torch.cdist(features, features).square())
    count = int(mask[0].sum())
    if not bool((mask.sum(1) == count).all()):
        raise ValueError("Batched posterior requires equal query counts")
    indices = mask.topk(count, dim=1).indices
    small = kernel[indices[:, :, None], indices[:, None, :]]
    cross = kernel[:, indices].permute(1, 0, 2)
    prior_selected = policy.prior[indices]
    noise = .0025+prior_selected.clamp(.05,.95)*(1-prior_selected.clamp(.05,.95))/8
    system = small+torch.diag_embed(noise)
    # Only queried values are gathered.
    residual = support.gather(1, indices)-prior_selected
    rhs = torch.cat([residual[:, :, None], cross.transpose(1,2)], -1)
    solved = torch.linalg.solve(system, rhs)
    mean = policy.prior+(cross*solved[:, None, :, 0]).sum(-1)
    variance = (kernel.diag()-(cross*solved[:, :, 1:].transpose(1,2)).sum(-1)).clamp_min(0)
    return mean, variance.sqrt()


@torch.no_grad()
def rollout(policy, support, budget, method):
    obs, mask = initial_state(support)
    initial_obs, initial_mask = obs.clone(), mask.clone()
    batch = len(obs)
    device = obs.device
    if method == "fixed":
        return policy.prior.argmax().expand(batch), 0
    if method == "gp_ucb":
        for _ in range(budget):
            mean, std = gp_posterior(policy, support, mask)
            action = (mean+1.5*std).masked_fill(mask.bool(), -1e9).argmax(1)
            mask.scatter_(1, action[:, None], 1)
        mean, _ = gp_posterior(policy, support, mask)
        return mean.argmax(1), (2+budget)*8
    if method == "random":
        for _ in range(budget):
            action = torch.rand_like(mask).masked_fill(mask.bool(), -1).argmax(1)
            mask.scatter_(1, action[:, None], 1)
        score = (4*policy.prior + 8*support*mask)/(4+8*mask)
        return score.argmax(1), (2+budget)*8
    steps = 0 if method == "one_shot" else budget
    for step in range(steps):
        visible_obs, visible_mask = (initial_obs, initial_mask) if method == "no_feedback" else (obs, mask)
        q, _, _ = policy(visible_obs, visible_mask, torch.full((batch,), budget-step, device=device))
        action = q.masked_fill(mask.bool(), -1e9).argmax(1)
        mask.scatter_(1, action[:, None], 1)
        obs = support*mask
    visible_obs, visible_mask = (initial_obs, initial_mask) if method == "no_feedback" else (obs, mask)
    _, r, _ = policy(visible_obs, visible_mask, torch.zeros(batch, device=device))
    return r.argmax(1), (2+steps)*8


@torch.no_grad()
def evaluate(policies, values, model_ids, calibration, evaluation, targets, episodes, seed):
    records = []
    for m in model_ids:
        # Same episodes across methods, budgets and training seeds.
        generator = torch.Generator(device=values.device).manual_seed(seed+m)
        draw = torch.rand(episodes, len(calibration), device=values.device, generator=generator).topk(8, dim=1).indices
        support_ids = calibration[draw]
        support = values[m, support_ids].mean(1)
        rates = values[m, evaluation].mean(0)
        for method, policy in policies.items():
            for budget in (0, 2, 4, 8, 16):
                torch.manual_seed(seed+m*100+budget)
                action, calls = rollout(policy, support, budget, method)
                selected = rates[action]
                records.append({"model": targets[m]["model_id"], "family": targets[m]["family"],
                                "method": method, "budget": budget, "mean_proxy": float(selected.mean()),
                                "mean_grid_regret": float((rates.max()-selected).mean()),
                                "measured_grid_best": float(rates.max()), "replayed_item_calls": calls,
                                "selected_configs": action.cpu().tolist(),
                                "calibration_item_indices": support_ids.cpu().tolist()})
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--collection", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--folds", type=int, nargs="+")
    ap.add_argument("--cohort", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    ap.add_argument("--supervised-steps", type=int, default=1000)
    ap.add_argument("--ppo-updates", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--episodes", type=int, default=128)
    ap.add_argument("--require-targets", type=int, default=12)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA; no silent CPU fallback")
    a.outdir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    started = time.monotonic()
    arrays, targets, configs, items, split_arrays, sources, diagnostics = load_data(a.data, a.collection, a.cohort)
    if len(targets) < a.require_targets:
        raise ValueError(f"Only {len(targets)} complete targets; expected {a.require_targets}")
    values = torch.tensor(arrays, device="cuda")
    features = torch.tensor([c["features"] for c in configs], device="cuda")
    splits = {s: torch.tensor(v, device="cuda") for s, v in split_arrays.items()}
    families = sorted({t["family"] for t in targets})
    folds = list(range(len(families))) if a.folds is None else a.folds
    run_config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()}
    run_config.update(device=torch.cuda.get_device_name(), torch_version=str(torch.__version__),
                      family_order=families, metric="lexical reconstruction proxy; not semantic ASR",
                      support_items=8, terminal_reward_items=32, ppo_epochs=4,
                      selection="fixed training schedules; validation diagnostic only; test never selects weights",
                      dataset_shape=list(arrays.shape), balanced_family_sampling=True,
                      new_target_calls_during_policy_training=0,
                      supervised_optimizer_updates=a.supervised_steps,
                      ppo_optimizer_updates=a.ppo_updates*4,
                      continued_supervised_optimizer_updates=a.ppo_updates*4,
                      ppo_training_state_examples=sum((2,4,8)[i%3]+1 for i in range(a.ppo_updates))*a.batch_size*4,
                      continued_supervised_state_examples=a.ppo_updates*4*a.batch_size,
                      budget16_is_extrapolation=True)
    write_json(a.outdir/"config.json", run_config)
    write_json(a.outdir/"data_manifest.json", {"sources": sources, "targets": targets,
               "item_splits": {s: [items[i]["item_id"] for i in ids] for s,ids in split_arrays.items()},
               "configurations": configs, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    write_json(a.outdir/"collection_diagnostics.json", diagnostics)
    records, validation, curves = [], [], []
    for fold in folds:
        test_family = families[fold]
        val_family = families[(fold+1) % len(families)]
        train_ids = [i for i,t in enumerate(targets) if t["family"] not in (test_family, val_family)]
        test_ids = [i for i,t in enumerate(targets) if t["family"] == test_family]
        val_ids = [i for i,t in enumerate(targets) if t["family"] == val_family]
        groups = [torch.tensor([i for i in train_ids if targets[i]["family"] == g], device="cuda")
                  for g in families if g not in (test_family, val_family)]
        prior = torch.stack([values[group][:, splits["train"]].mean((0,1)) for group in groups]).mean(0)
        for seed in a.seeds:
            print(json.dumps({"stage":"fold_start", "test_family":test_family,
                              "validation_family":val_family, "seed":seed}), flush=True)
            torch.manual_seed(seed)
            model = Policy(features, prior).cuda()
            initial = {k:v.clone() for k,v in model.state_dict().items()}
            s_curve = supervised_updates(model, values, groups, splits["train"], a.supervised_steps, a.batch_size, seed)
            supervised = copy.deepcopy(model).eval()
            r_curve = ppo_updates(model, values, groups, splits["train"], a.ppo_updates, a.batch_size, seed+10000)
            rl = model.eval()
            # A continuation controls optimizer-update count, while reporting its different state count/cost.
            continued = copy.deepcopy(supervised)
            c_curve = supervised_updates(continued, values, groups, splits["train"], a.ppo_updates*4,
                                          a.batch_size, seed+20000, learning_rate=1e-4)
            policies = {"fixed": supervised, "random": supervised, "gp_ucb": supervised, "one_shot": supervised,
                        "supervised": supervised, "supervised_continued": continued,
                        "ppo": rl, "no_feedback": rl}
            val = evaluate(policies, values, val_ids, splits["train"], splits["validation"], targets, a.episodes, 6700)
            test = evaluate(policies, values, test_ids, splits["train"], splits["test"], targets, a.episodes, 8900)
            for destination, rows in ((validation, val), (records, test)):
                for r in rows:
                    r.update(fold=fold, training_seed=seed, test_family=test_family, validation_family=val_family)
                    destination.append(r)
            checkpoint = {"supervised": supervised.state_dict(), "ppo": rl.state_dict(),
                          "supervised_continued": continued.state_dict(), "config": run_config,
                          "train_model_ids": [targets[i]["model_id"] for i in train_ids],
                          "test_family": test_family, "validation_family": val_family, "seed":seed}
            assert any(not torch.equal(initial[k], rl.state_dict()[k]) for k in initial)
            assert any(not torch.equal(supervised.state_dict()[k], rl.state_dict()[k]) for k in initial)
            torch.save(checkpoint, a.outdir/f"fold{fold}_seed{seed}.pt")
            curves.append({"fold":fold,"seed":seed,"supervised":s_curve,"ppo":r_curve,"continued":c_curve})
            write_json(a.outdir/"test_results.json", records)
            write_json(a.outdir/"validation.json", validation)
            write_json(a.outdir/"training_curves.json", curves)
            write_json(a.outdir/"progress.json", {"completed_runs":len(curves),"planned_runs":len(folds)*len(a.seeds),
                       "elapsed_seconds":time.monotonic()-started,"status":"training"})
    write_json(a.outdir/"progress.json", {"completed_runs":len(curves),"planned_runs":len(folds)*len(a.seeds),
               "elapsed_seconds":time.monotonic()-started,"status":"completed"})
    subprocess.run([sys.executable, str(Path(__file__).with_name("report_selector_gpu.py")),
                    "--results", str(a.outdir.parent)], check=True)
    print(json.dumps({"status":"completed","runs":len(curves),"seconds":time.monotonic()-started}), flush=True)


if __name__ == "__main__":
    main()
