"""CPU-only selector pilot on recorded, explicitly benign FLORES reconstructions.

No target generation, harmful-result loader, or simulated unobserved configuration
is provided. A fixed-support replay exposes only queried cells to each policy.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

TAGS = ["panel_qwen3", "panel_phi", "panel_gpt4omini", "panel_mistral", "panel_qwen32"]
CONDITIONS = [f"interleave_{o}_n{n}" for o in ("ordered", "shuffled") for n in (2, 4, 6, 8, 10)]
INITIAL = (0, 9)
BUDGETS = (0, 1, 2, 4, 8)


def binary(value):
    if type(value) not in (int, bool) or value not in (0, 1):
        raise ValueError("Expected a JSON boolean or integer 0/1")
    return int(value)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Corpus:
    values: np.ndarray
    models: list
    items: list
    manifest: dict


def load_corpus(root):
    """Read only the five benign arms; fail closed on non-benign/duplicate rows."""
    root = Path(root)
    matrices, model_names, item_sets, hashes, origins, prompts = [], [], [], [], {}, {}
    for tag in TAGS:
        folder = root / "private_artifacts/paper_main" / tag
        source = folder / "benign_gen/restricted_target_outputs.jsonl"
        audit = folder / "benign_recon/restricted_reconstruction_audit.jsonl"
        raw = {}
        for line in source.open():
            row = json.loads(line)
            if row["condition"] not in CONDITIONS:
                continue
            if row.get("risk_type") != "benign" or not re.fullmatch(r"benign_\d+", row["item_id"]):
                raise ValueError(f"Non-benign provenance in {source}")
            key = row["item_id"], row["condition"]
            if key in raw:
                raise ValueError(f"Duplicate source key in {source}")
            raw[key] = row
        table, names = {}, set()
        for line in audit.open():
            row = json.loads(line)
            if row["condition"] not in CONDITIONS:
                continue
            key = row["item_id"], row["condition"]
            if key in table:
                raise ValueError(f"Duplicate audit key in {audit}")
            if row.get("risk_type") != "benign" or key not in raw:
                raise ValueError(f"Invalid benign audit provenance in {audit}")
            target = raw[key]
            if not row.get("response_sha256") or row["response_sha256"] != target["response_sha256"]:
                raise ValueError(f"Unmatched response hash in {audit}")
            if row["target_model"] != target["target_model"]:
                raise ValueError("Target mismatch")
            if not binary(row.get("reconstruction_parse_valid")):
                raise ValueError("Invalid judge parse; do not silently treat it as failure")
            table[key] = binary(row.get("semantic_reconstruction_equivalent"))
            names.add(row["target_model"])
            original_hash = hashlib.sha256(target["original"].strip().encode()).hexdigest()
            if key[0] in origins and origins[key[0]] != original_hash:
                raise ValueError("Same item ID refers to different original texts")
            origins[key[0]] = original_hash
            prompts.setdefault(key, set()).add(hashlib.sha256(target["prompt"].encode()).hexdigest())
        if len(names) != 1 or set(table) != set(raw):
            raise ValueError("Incomplete or mixed-model run")
        ids = sorted({key[0] for key in table})
        if any((item, condition) not in table for item in ids for condition in CONDITIONS):
            raise ValueError("Missing configuration cell; no imputation is allowed")
        matrices.append(table)
        model_names.append(names.pop())
        item_sets.append(set(ids))
        hashes.append({"tag": tag, "source": str(source.relative_to(root)),
                       "source_sha256": digest(source), "judge": str(audit.relative_to(root)),
                       "judge_sha256": digest(audit)})
    if any(s != item_sets[0] for s in item_sets):
        raise ValueError("Targets must have exactly matched benign item IDs")
    ids = sorted(item_sets[0])
    values = np.array([[[table[item, c] for c in CONDITIONS] for item in ids]
                       for table in matrices], dtype=np.float32)
    manifest = {
        "dataset": "recorded benign FLORES reconstruction pilot",
        "shape_models_items_configs": list(values.shape), "models": model_names,
        "conditions": CONDITIONS, "item_ids": ids, "sources": hashes,
        "prompt_keys_varying_across_targets": sum(len(x) > 1 for x in prompts.values()),
        "limitations": ["Only recorded configurations are supported.",
                        "Judge labels are measurements, not human ground truth.",
                        "Decoding and token costs are not fully recorded.",
                        "Original text hashes match across target runs."],
    }
    return Corpus(values, model_names, ids, manifest)


def split_items(n, seed=20260903):
    if n < 30:
        raise ValueError("This pilot requires at least 30 distinct source items")
    ids = np.random.default_rng(seed).permutation(n)
    # Fixed source-item split, shared across all target folds and training seeds.
    a, b = n // 2, n // 5
    return {"train_calibration": ids[:a], "validation": ids[a:a+b], "test": ids[a+b:]}


def validate_split(split):
    sets = [set(v) for v in split.values()]
    if any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i)):
        raise ValueError("Source-item partitions overlap")


class BenignReplay:
    """A single finite, deterministic support table, with counted reveals."""
    def __init__(self, support, initial=INITIAL):
        self._support = np.array(support, dtype=np.float32, copy=True)
        if self._support.ndim != 2 or not np.isin(self._support, [0, 1]).all():
            raise ValueError("Support must contain recorded binary reconstructions")
        self.observed = np.zeros(self._support.shape[1], dtype=np.float32)
        self.mask = np.zeros_like(self.observed)
        self.calls = 0
        self.trace = []
        for arm in initial:
            self.query(arm)

    def query(self, arm):
        if arm < 0 or arm >= len(self.mask) or self.mask[arm]:
            raise ValueError("Invalid or already observed configuration")
        value = float(self._support[:, arm].mean())
        self.observed[arm], self.mask[arm] = value, 1
        self.calls += len(self._support)
        self.trace.append({"arm": int(arm), "observed_reconstruction": value,
                           "replayed_item_calls": len(self._support)})
        return value

    def snapshot(self):
        return self.observed.copy(), self.mask.copy()


def state(prior, obs, mask, remaining, budget):
    """No model identity, future labels, or test measurements enter this state."""
    prior = np.asarray(prior, dtype=np.float32)
    obs, mask = np.asarray(obs, dtype=np.float32), np.asarray(mask, dtype=np.float32)
    single = obs.ndim == 1
    if single:
        obs, mask = obs[None], mask[None]
    batch, arms = obs.shape
    p = np.broadcast_to(prior, (batch, arms))
    structural = np.array([[int(c.rsplit("n", 1)[1]) / 10, float("shuffled" in c)]
                           for c in CONDITIONS], dtype=np.float32)
    s = np.broadcast_to(structural, (batch, arms, 2))
    x = np.concatenate([s, p[..., None],
                        np.where(mask > 0, obs, p)[..., None], mask[..., None]], axis=-1)
    rem = np.full((batch, 1), remaining / 8, dtype=np.float32)
    total = np.full((batch, 1), budget / 8, dtype=np.float32)
    result = np.concatenate([x.reshape(batch, -1), rem, total], axis=1)
    return result[0] if single else result


class Selector(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(52, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.query_head = nn.Linear(64, 10)
        self.recommend_head = nn.Linear(64, 10)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        z = self.body(x)
        return self.query_head(z), self.recommend_head(z), self.value_head(z).squeeze(-1)


def sample_training(values, model_ids, item_ids, batch, support_size, rng):
    """Support and terminal reward use disjoint items, all from training only."""
    supports, rewards = [], []
    for _ in range(batch):
        model = int(rng.choice(model_ids))
        perm = rng.permutation(item_ids)
        supports.append(values[model, perm[:support_size]].mean(axis=0))
        rewards.append(values[model, perm[support_size:]].mean(axis=0))
    return np.array(supports, np.float32), np.array(rewards, np.float32)


def fit_supervised(values, model_ids, items, prior, seed, steps=200, batch=64, support_size=4):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Selector()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    for step in range(steps):
        observed_all, evaluation = sample_training(values, model_ids, items, batch, support_size, rng)
        mask = np.zeros_like(observed_all)
        mask[:, INITIAL] = 1
        budget = int(rng.choice(BUDGETS))
        revealed = int(rng.integers(budget + 1))
        for row in mask:
            if revealed:
                row[rng.choice(np.flatnonzero(row == 0), revealed, replace=False)] = 1
        x = torch.from_numpy(state(prior, observed_all, mask, budget-revealed, budget))
        query, recommendation, _ = model(x)
        target = torch.from_numpy(evaluation)
        loss = nn.functional.binary_cross_entropy_with_logits(recommendation, target)
        if revealed < budget:
            available = torch.from_numpy(mask == 0)
            logits = query.masked_fill(~available, -1e9)
            soft_target = torch.softmax((target / .08).masked_fill(~available, -1e9), dim=-1)
            loss = loss + .25 * -(soft_target * torch.log_softmax(logits, -1)).sum(-1).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if step == steps-1:
            history.append({"step": step+1, "loss": float(loss.detach())})
    return model, history


def fit_rl(pretrained, values, model_ids, items, prior, seed, steps=400, batch=64, support_size=4):
    """Actor-critic on a benign finite-table replay; gamma=1, terminal reward only."""
    import copy
    torch.manual_seed(seed + 100000)
    rng = np.random.default_rng(seed + 100000)
    model = copy.deepcopy(pretrained)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    history = []
    row_index = np.arange(batch)
    for update in range(steps):
        observed_all, evaluation = sample_training(values, model_ids, items, batch, support_size, rng)
        mask = np.zeros_like(observed_all)
        mask[:, INITIAL] = 1
        obs = np.zeros_like(observed_all)
        obs[:, INITIAL] = observed_all[:, INITIAL]
        budget = int(rng.choice(BUDGETS))
        logprobs, baselines, entropies = [], [], []
        for t in range(budget):
            x = torch.from_numpy(state(prior, obs, mask, budget-t, budget))
            query, _, value = model(x)
            dist = torch.distributions.Categorical(logits=query.masked_fill(torch.from_numpy(mask > 0), -1e9))
            action = dist.sample()
            arms = action.detach().numpy()
            logprobs.append(dist.log_prob(action))
            baselines.append(value)
            entropies.append(dist.entropy())
            obs[row_index, arms] = observed_all[row_index, arms]
            mask[row_index, arms] = 1
        _, recommendation, value = model(torch.from_numpy(state(prior, obs, mask, 0, budget)))
        dist = torch.distributions.Categorical(logits=recommendation)
        action = dist.sample()
        reward = torch.from_numpy(evaluation[row_index, action.detach().numpy()])
        logprobs.append(dist.log_prob(action))
        baselines.append(value)
        entropies.append(dist.entropy())
        logp = torch.stack(logprobs)
        baseline = torch.stack(baselines)
        advantage = reward.unsqueeze(0) - baseline.detach()
        actor_loss = -(logp * advantage).mean()
        critic_loss = ((baseline - reward.unsqueeze(0)) ** 2).mean()
        auxiliary = nn.functional.binary_cross_entropy_with_logits(recommendation, torch.from_numpy(evaluation))
        loss = actor_loss + .5 * critic_loss + .1 * auxiliary - .01 * torch.stack(entropies).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if (update+1) % 100 == 0 or update == steps-1:
            history.append({"step": update+1, "reward": float(reward.mean()),
                            "loss": float(loss.detach()), "extra_config_budget": budget})
    return model, history


def gp_posterior(prior, obs, mask, support_size):
    # Fixed kernel and noise; no fitting or selection on the held-out target.
    xy = np.array([[int(c.rsplit("n", 1)[1]) / 10, float("shuffled" in c)]
                   for c in CONDITIONS])
    dist2 = (((xy[:, None] - xy[None]) / np.array([.4, .7])) ** 2).sum(-1)
    kernel = .04 * np.exp(-.5 * dist2)
    selected = np.flatnonzero(mask)
    if not len(selected):
        return prior.copy(), np.sqrt(kernel.diagonal())
    noise = np.clip(prior[selected], .05, .95)
    noise = noise * (1-noise) / support_size + 1e-6
    inverse = np.linalg.solve(kernel[np.ix_(selected, selected)] + np.diag(noise),
                              np.column_stack([obs[selected]-prior[selected], kernel[selected]]))
    mu = prior + kernel[:, selected] @ inverse[:, 0]
    var = np.maximum(kernel.diagonal() - (kernel[:, selected] * inverse[:, 1:].T).sum(-1), 0)
    return mu, np.sqrt(var)


def model_outputs(model, prior, obs, mask, remaining, budget):
    with torch.no_grad():
        q, r, _ = model(torch.from_numpy(state(prior, obs, mask, remaining, budget)).unsqueeze(0))
    return q[0].numpy(), r[0].numpy()


def rollout(method, support, prior, budget, rng, model=None):
    if budget not in BUDGETS:
        raise ValueError("Unsupported budget")
    if method == "train_fixed":
        return int(np.argmax(prior)), 0, []
    env = BenignReplay(support)
    initial_obs, _ = env.snapshot()
    initial_mask = np.zeros_like(env.mask)
    initial_mask[list(INITIAL)] = 1
    once = method == "supervised_once"
    for t in range(0 if once else budget):
        obs, mask = env.snapshot()
        if method == "random":
            arm = int(rng.choice(np.flatnonzero(mask == 0)))
        elif method == "gp_ucb":
            mu, sd = gp_posterior(prior, obs, mask, len(support))
            score = mu + 1.5 * sd
            arm = int(np.argmax(np.where(mask == 0, score, -np.inf)))
        else:
            visible_mask = mask
            if method == "rl_no_feedback":
                obs = initial_obs
                visible_mask = initial_mask
            query, _ = model_outputs(model, prior, obs, visible_mask, budget-t, budget)
            arm = int(np.argmax(np.where(mask == 0, query, -np.inf)))
        env.query(arm)
    obs, mask = env.snapshot()
    if method == "random":
        score = (4 * prior + len(support) * obs) / (4 + len(support) * mask)
    elif method == "gp_ucb":
        score, _ = gp_posterior(prior, obs, mask, len(support))
    else:
        if method == "rl_no_feedback":
            obs = initial_obs
            mask = initial_mask
        _, score = model_outputs(model, prior, obs, mask, 0, 0 if once else budget)
    return int(np.argmax(score)), env.calls, env.trace


def evaluate(values, heldout, test_ids, calibration_ids, prior, methods, budgets, episodes, seed, support_size=4):
    records = []
    for episode in range(episodes):
        # Same support items for every budget, method, and training seed.
        support_ids = np.random.default_rng(seed + episode).choice(calibration_ids, support_size, replace=False)
        support = values[heldout, support_ids]
        test_rates = values[heldout, test_ids].mean(axis=0)
        for budget in budgets:
            for method, model in methods.items():
                rng = np.random.default_rng(seed + episode * 100 + budget)
                arm, calls, trace = rollout(method, support, prior, budget, rng, model)
                records.append({
                    "episode": episode, "extra_config_budget": budget, "method": method,
                    "selected_arm": arm, "test_reconstruction": float(test_rates[arm]),
                    "test_grid_best": float(test_rates.max()),
                    "test_grid_regret": float(test_rates.max()-test_rates[arm]),
                    "replayed_calibration_item_calls": calls,
                    "calibration_item_ids": [int(i) for i in support_ids],
                    "trace": trace,
                })
    return records


def aggregate(records, seed=44):
    output = {}
    for method in sorted({r["method"] for r in records}):
        by_budget = {}
        for budget in BUDGETS:
            rows = [r for r in records if r["method"] == method and r["extra_config_budget"] == budget]
            if not rows:
                continue
            models = sorted({r["heldout_model"] for r in rows})
            means = np.array([np.mean([r["test_reconstruction"] for r in rows if r["heldout_model"] == m])
                              for m in models])
            rng = np.random.default_rng(seed)
            samples = means[rng.integers(len(means), size=(2000, len(means)))].mean(1)
            by_budget[str(budget)] = {
                "macro_test_reconstruction": float(means.mean()),
                "target_cluster_bootstrap95": np.quantile(samples, [.025, .975]).tolist(),
                "mean_test_grid_regret": float(np.mean([r["test_grid_regret"] for r in rows])),
                "mean_replayed_calibration_item_calls": float(np.mean([r["replayed_calibration_item_calls"] for r in rows])),
                "per_target": dict(zip(models, means.tolist())),
            }
        output[method] = by_budget
    return output

