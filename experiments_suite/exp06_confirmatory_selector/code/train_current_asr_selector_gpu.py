#!/usr/bin/env python3
"""GPU-trained contextual ASR selector for the current 17-model/160-arm panel.

The context is built exclusively from harmless FLORES reconstruction and
FalseReject over-refusal probes.  Training and evaluation are family-held-out;
target arm feedback comes from calibration items and recommendations are scored
on disjoint test items.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent


def load_base():
    path = HERE / "pilot_family_loo_asr_selector.py"
    spec = importlib.util.spec_from_file_location("poly_rl_base", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["poly_rl_base"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base = load_base()


def context_vector(tag, benign_root: Path, false_root: Path, dataset="joint"):
    benign_objects = []
    datasets = ("mj", "lg") if dataset == "joint" else (dataset,)
    for dataset_name in datasets:
        path = benign_root / f"{tag}_{dataset_name}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        benign_objects.append(json.loads(path.read_text(encoding="utf-8")))

    cell_keys = sorted(benign_objects[0]["benign_recon_by_cell"])
    vector = [
        float(np.mean([obj["benign_recon_by_cell"][key] for obj in benign_objects]))
        for key in cell_keys
    ]
    for frame in ("persona", "fiction", "pap"):
        for metric in ("nonrefusal", "fiction_hold", "detail"):
            vector.append(
                float(np.mean([obj["frame_signals"][frame][metric] for obj in benign_objects]))
            )

    false_path = false_root / f"willingness_prior_falsereject_{tag}.json"
    if not false_path.exists():
        raise FileNotFoundError(false_path)
    false = json.loads(false_path.read_text(encoding="utf-8"))
    for frame in ("plain", "persona", "fiction", "pap"):
        for metric in ("nonref", "hedge", "fic", "words"):
            vector.append(float(false["per_frame"][frame][metric]))
    for frame in ("persona", "fiction", "pap"):
        for metric in ("dnonref", "dwords", "hedge_ratio", "fiction_struct"):
            vector.append(float(false["frame_effect_vs_plain"][frame][metric]))
    return np.asarray(vector, dtype=np.float32)


def tensor_panel(panel, arms, selected_dataset="joint"):
    tags = sorted(panel)
    item_keys = []
    datasets = ("mj", "lg") if selected_dataset == "joint" else (selected_dataset,)
    for dataset in datasets:
        exemplar = panel[tags[0]][dataset][arms[0]]
        item_keys.extend((dataset, item_id) for item_id in sorted(exemplar))
    values = np.zeros((len(tags), len(item_keys), len(arms)), dtype=np.float32)
    for model_index, tag in enumerate(tags):
        for item_index, (dataset, item_id) in enumerate(item_keys):
            values[model_index, item_index] = [
                panel[tag][dataset][arm][item_id] for arm in arms
            ]
    return tags, item_keys, values


def item_splits(item_keys, seed):
    output = {"train": [], "calibration": [], "test": []}
    for dataset in ("mj", "lg"):
        positions = [i for i, key in enumerate(item_keys) if key[0] == dataset]
        rng = np.random.default_rng(seed + (100_000 if dataset == "lg" else 0))
        rng.shuffle(positions)
        n_train = len(positions) // 2
        n_calib = len(positions) // 4
        output["train"].extend(positions[:n_train])
        output["calibration"].extend(positions[n_train : n_train + n_calib])
        output["test"].extend(positions[n_train + n_calib :])
    return output


class Policy(nn.Module):
    def __init__(self, arm_features, prior, context_dim, width=96):
        super().__init__()
        self.register_buffer("arm_features", arm_features)
        self.register_buffer("prior", prior)
        input_dim = arm_features.shape[1] + context_dim + 4
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh()
        )
        self.fuse = nn.Sequential(nn.Linear(width * 2, width), nn.Tanh())
        self.query = nn.Linear(width, 1)
        self.recommend = nn.Linear(width, 1)
        self.critic = nn.Sequential(nn.Linear(width, width // 2), nn.Tanh(), nn.Linear(width // 2, 1))

    def forward(self, observed, mask, remaining, context):
        batch, arms = mask.shape
        prior = self.prior[None].expand(batch, arms)
        visible = torch.where(mask.bool(), observed, prior)
        ctx = context[:, None].expand(batch, arms, context.shape[1])
        features = self.arm_features[None].expand(batch, arms, -1)
        rem = remaining[:, None, None].expand(batch, arms, 1) / 16.0
        x = torch.cat(
            [features, prior[..., None], visible[..., None], mask[..., None], rem, ctx], dim=-1
        )
        z = self.encoder(x)
        denominator = mask.sum(1, keepdim=True)
        masked_pool = (z * mask[..., None]).sum(1) / denominator.clamp_min(1)
        global_pool = z.mean(1)
        pooled = torch.where((denominator > 0), masked_pool, global_pool)
        fused = self.fuse(torch.cat([z, pooled[:, None].expand_as(z)], dim=-1))
        return (
            self.query(fused).squeeze(-1),
            self.recommend(fused).squeeze(-1),
            self.critic(pooled).squeeze(-1),
        )


def episode(values, model_ids, item_ids, batch, support_count, reward_count, generator):
    models = model_ids[
        torch.randint(len(model_ids), (batch,), device=values.device, generator=generator)
    ]
    random = torch.rand(batch, len(item_ids), device=values.device, generator=generator)
    chosen = item_ids[random.topk(support_count + reward_count, dim=1).indices]
    cells = values[models[:, None], chosen]
    support = cells[:, :support_count].mean(1)
    reward = cells[:, support_count:].mean(1)
    return support, reward, models


def supervised_train(policy, values, contexts, model_ids, items, steps, batch, seed, use_context):
    optimizer = torch.optim.AdamW(policy.parameters(), lr=8e-4, weight_decay=1e-4)
    generator = torch.Generator(device=values.device).manual_seed(seed)
    for step in range(steps):
        support_count = min(8, max(2, len(items) // 3))
        reward_count = min(16, len(items) - support_count)
        support, reward, models = episode(
            values, model_ids, items, batch, support_count, reward_count, generator
        )
        context = contexts[models] if use_context else torch.zeros_like(contexts[models])
        mask = torch.zeros_like(support)
        reveal = step % 9
        if reveal:
            indices = torch.rand_like(mask).topk(reveal, dim=1).indices
            mask.scatter_(1, indices, 1)
        observed = support * mask
        query, recommend, _ = policy(
            observed,
            mask,
            torch.full((batch,), 8 - reveal, device=values.device),
            context,
        )
        recommend_loss = F.binary_cross_entropy_with_logits(recommend, reward)
        query_logits = query.masked_fill(mask.bool(), -1e9)
        target = torch.softmax((reward / 0.08).masked_fill(mask.bool(), -1e9), dim=1)
        query_loss = -(target * torch.log_softmax(query_logits, dim=1)).sum(1).mean()
        loss = recommend_loss + 0.25 * query_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()


def ppo_train(policy, values, contexts, model_ids, items, updates, batch, seed, use_context):
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=1e-4)
    generator = torch.Generator(device=values.device).manual_seed(seed)
    for update in range(updates):
        budget = (2, 4, 8, 16)[update % 4]
        support_count = min(8, max(2, len(items) // 3))
        reward_count = min(16, len(items) - support_count)
        support, final, models = episode(
            values, model_ids, items, batch, support_count, reward_count, generator
        )
        context = contexts[models] if use_context else torch.zeros_like(contexts[models])
        observed = torch.zeros_like(support)
        mask = torch.zeros_like(support)
        snapshots, actions, old_logprobs, old_values, terminals = [], [], [], [], []
        with torch.no_grad():
            for step in range(budget + 1):
                remaining = torch.full((batch,), budget - step, device=values.device)
                query, recommend, value = policy(observed, mask, remaining, context)
                terminal = step == budget
                logits = recommend if terminal else query.masked_fill(mask.bool(), -1e9)
                distribution = Categorical(logits=logits)
                action = distribution.sample()
                snapshots.append((observed.clone(), mask.clone(), remaining))
                actions.append(action)
                old_logprobs.append(distribution.log_prob(action))
                old_values.append(value)
                terminals.append(
                    torch.full((batch,), terminal, device=values.device, dtype=torch.bool)
                )
                if not terminal:
                    mask.scatter_(1, action[:, None], 1)
                    observed = support * mask
            terminal_reward = final.gather(1, actions[-1][:, None]).squeeze(1)

        all_observed = torch.cat([s[0] for s in snapshots])
        all_masks = torch.cat([s[1] for s in snapshots])
        all_remaining = torch.cat([s[2] for s in snapshots])
        all_context = context.repeat(budget + 1, 1)
        all_actions = torch.cat(actions)
        all_old_logprobs = torch.cat(old_logprobs)
        all_old_values = torch.cat(old_values)
        all_terminal = torch.cat(terminals)
        returns = terminal_reward.repeat(budget + 1)
        advantages = returns - all_old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        for _ in range(4):
            query, recommend, value = policy(
                all_observed, all_masks, all_remaining, all_context
            )
            logits = torch.where(
                all_terminal[:, None], recommend, query.masked_fill(all_masks.bool(), -1e9)
            )
            distribution = Categorical(logits=logits)
            ratio = (distribution.log_prob(all_actions) - all_old_logprobs).exp()
            policy_loss = -torch.minimum(
                ratio * advantages, torch.clamp(ratio, 0.8, 1.2) * advantages
            ).mean()
            value_loss = F.mse_loss(value, returns)
            entropy = distribution.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()


@torch.no_grad()
def rollout(policy, support, context, budget, feedback=True):
    observed = torch.zeros_like(support)
    mask = torch.zeros_like(support)
    for step in range(budget):
        query, _, _ = policy(
            observed if feedback else torch.zeros_like(observed),
            mask,
            torch.full((len(support),), budget - step, device=support.device),
            context,
        )
        action = query.masked_fill(mask.bool(), -1e9).argmax(1)
        mask.scatter_(1, action[:, None], 1)
        if feedback:
            observed = support * mask
    _, recommend, _ = policy(
        observed if feedback else torch.zeros_like(observed),
        mask,
        torch.zeros(len(support), device=support.device),
        context,
    )
    return recommend.argmax(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--benign-root", type=Path, required=True)
    parser.add_argument("--false-root", type=Path, required=True)
    parser.add_argument("--held-family", required=True)
    parser.add_argument("--dataset", choices=("joint", "mj", "lg"), default="joint")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-seed", type=int, default=47)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--supervised-steps", type=int, default=600)
    parser.add_argument("--ppo-updates", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--random-repetitions", type=int, default=128)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)

    panel, arms = base.load_complete_panel(args.raw_root)
    tags, item_keys, values_np = tensor_panel(panel, arms, args.dataset)
    families = [base.model_family(tag) for tag in tags]
    if args.held_family not in families:
        raise ValueError(f"unknown held family {args.held_family}; choices={sorted(set(families))}")
    contexts_np = np.stack(
        [context_vector(tag, args.benign_root, args.false_root, args.dataset) for tag in tags]
    )
    train_models = np.asarray(
        [i for i, family in enumerate(families) if family != args.held_family], dtype=int
    )
    test_models = np.asarray(
        [i for i, family in enumerate(families) if family == args.held_family], dtype=int
    )
    mean = contexts_np[train_models].mean(0)
    std = contexts_np[train_models].std(0) + 1e-5
    contexts_np = np.clip((contexts_np - mean) / std, -8, 8)
    splits = item_splits(item_keys, args.split_seed)

    values = torch.tensor(values_np, device=device)
    contexts = torch.tensor(contexts_np, device=device)
    arm_features = torch.tensor(
        np.stack([base.arm_features(arm) for arm in arms]).astype(np.float32),
        device=device,
    )
    # Equalize the total episode probability of every source family.  The
    # current panel has 1--3 models per family, so model-uniform sampling would
    # otherwise over-represent the largest families.
    source_counts = {
        family: sum(families[i] == family for i in train_models)
        for family in sorted({families[i] for i in train_models})
    }
    common_count = int(np.lcm.reduce(list(source_counts.values())))
    balanced_models = []
    for index in train_models:
        balanced_models.extend([int(index)] * (common_count // source_counts[families[index]]))
    train_ids = torch.tensor(balanced_models, device=device)
    train_items = torch.tensor(splits["train"], device=device)
    calibration_items = torch.tensor(splits["calibration"], device=device)
    test_items = torch.tensor(splits["test"], device=device)
    family_priors = []
    for family in sorted(source_counts):
        indices = torch.tensor(
            [i for i in train_models if families[i] == family], device=device
        )
        family_priors.append(values[indices][:, train_items].mean((0, 1)))
    prior = torch.stack(family_priors).mean(0)

    supervised_context = Policy(arm_features, prior, contexts.shape[1]).to(device)
    supervised_train(
        supervised_context,
        values,
        contexts,
        train_ids,
        train_items,
        args.supervised_steps,
        args.batch_size,
        args.seed,
        True,
    )
    ppo_context = copy.deepcopy(supervised_context)
    ppo_train(
        ppo_context,
        values,
        contexts,
        train_ids,
        train_items,
        args.ppo_updates,
        args.batch_size,
        args.seed + 10_000,
        True,
    )
    supervised_no_context = Policy(arm_features, prior, contexts.shape[1]).to(device)
    supervised_train(
        supervised_no_context,
        values,
        contexts,
        train_ids,
        train_items,
        args.supervised_steps,
        args.batch_size,
        args.seed + 20_000,
        False,
    )
    ppo_no_context = copy.deepcopy(supervised_no_context)
    ppo_train(
        ppo_no_context,
        values,
        contexts,
        train_ids,
        train_items,
        args.ppo_updates,
        args.batch_size,
        args.seed + 30_000,
        False,
    )

    records = []
    budgets = (0, 2, 4, 8, 16)
    for model_index in test_models:
        support = values[model_index, calibration_items].mean(0, keepdim=True)
        final = values[model_index, test_items].mean(0)
        context = contexts[model_index : model_index + 1]
        zero_context = torch.zeros_like(context)
        for budget in budgets:
            methods = {
                "supervised_context": rollout(supervised_context, support, context, budget),
                "ppo_context": rollout(ppo_context, support, context, budget),
                "ppo_no_feedback": rollout(ppo_context, support, context, budget, feedback=False),
                "ppo_no_context": rollout(ppo_no_context, support, zero_context, budget),
            }
            fixed = int(prior.argmax())
            methods["best_fixed_other_families"] = torch.tensor([fixed], device=device)
            for method, action in methods.items():
                index = int(action.item())
                records.append(
                    {
                        "method": method,
                        "budget": budget,
                        "held_family": args.held_family,
                        "dataset": args.dataset,
                        "model": tags[model_index],
                        "seed": args.seed,
                        "split_seed": args.split_seed,
                        "arm": arms[index],
                        "test_asr": float(final[index]),
                        "oracle_test_asr": float(final.max()),
                    }
                )
            generator = np.random.default_rng(
                args.seed * 1_000_003 + model_index * 1009 + budget
            )
            for repetition in range(args.random_repetitions):
                if budget == 0:
                    index = fixed
                else:
                    candidates = generator.choice(len(arms), size=budget, replace=False)
                    support_np = support[0].cpu().numpy()
                    index = int(candidates[np.argmax(support_np[candidates])])
                records.append(
                    {
                        "method": "random_probe_best_observed",
                        "budget": budget,
                        "held_family": args.held_family,
                        "dataset": args.dataset,
                        "model": tags[model_index],
                        "seed": args.seed,
                        "split_seed": args.split_seed,
                        "repetition": repetition,
                        "arm": arms[index],
                        "test_asr": float(final[index]),
                        "oracle_test_asr": float(final.max()),
                    }
                )

    payload = {
        "protocol": {
            "metric": "Qwen3Guard unsafe ASR (U)",
            "training": "supervised warm start followed by PPO actor-critic",
            "outer_split": "held-out model family",
            "item_split": "50% source train / 25% target calibration / 25% target test",
            "context": "harmless FLORES reconstruction + FalseReject over-refusal fingerprints",
            "harmful_target_feedback_in_context": False,
            "target_test_used_for_training_or_selection": False,
            "held_family": args.held_family,
            "dataset": args.dataset,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "arms": len(arms),
            "context_dimensions": int(contexts.shape[1]),
            "supervised_steps": args.supervised_steps,
            "ppo_updates": args.ppo_updates,
            "batch_size": args.batch_size,
            "source_family_balancing": "equal total sampling mass and equal prior weight per family",
        },
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(args.out)
    checkpoint = args.out.with_suffix(".pt")
    torch.save(
        {
            "policy": ppo_context.state_dict(),
            "arms": arms,
            "held_family": args.held_family,
            "dataset": args.dataset,
            "seed": args.seed,
            "context_mean": mean,
            "context_std": std,
        },
        checkpoint,
    )
    print(
        json.dumps(
            {
                "held_family": args.held_family,
                "dataset": args.dataset,
                "seed": args.seed,
                "records": len(records),
                "out": str(args.out),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
