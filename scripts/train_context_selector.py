#!/usr/bin/env python3
"""Contextual joint-ASR selector: policy maps (target behavioral fingerprint, probe
history, remaining budget) -> next config, then a final recommendation.

The universal claim is tested by FAMILY-held-out leave-one-out: train on some families,
recommend configs for an unseen family's unseen targets. The value of the fingerprint is
isolated by a no_context ablation (same policy, context zeroed) and by context-free
baselines (fixed, random, gp_ucb). Reward = joint J (reconstruction-gated ASR). Success is
read at LOW budget (0-2 probes): a good fingerprint should give a warm start.

No target-model or judge calls here; trains a small selector on the measured joint table.
"""
from __future__ import annotations
import argparse, copy, json, hashlib, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical


class ContextPolicy(nn.Module):
    def __init__(self, features, prior, ctx_dim, width=128):
        super().__init__()
        self.register_buffer("features", features)     # [A, F]
        self.register_buffer("prior", prior)           # [A]
        self.encoder = nn.Sequential(nn.Linear(features.shape[1] + 4 + ctx_dim, width), nn.Tanh(),
                                     nn.Linear(width, width), nn.Tanh())
        self.mix = nn.Sequential(nn.Linear(width * 2, width), nn.Tanh())
        self.query = nn.Linear(width, 1)
        self.recommend = nn.Linear(width, 1)
        self.critic = nn.Linear(width, 1)

    def forward(self, obs, mask, remaining, context):
        b, a = mask.shape
        p = self.prior.expand(b, a)
        visible = torch.where(mask.bool(), obs, p)
        ctx = context[:, None, :].expand(b, a, context.shape[1])
        x = torch.cat([self.features.expand(b, -1, -1), p[..., None], visible[..., None],
                       mask[..., None], remaining[:, None, None].expand(b, a, 1) / 10, ctx], -1)
        z = self.encoder(x)
        pooled = (z * mask[..., None]).sum(1) / mask.sum(1).clamp_min(1)[:, None]
        z = self.mix(torch.cat([z, pooled[:, None].expand_as(z)], -1))
        return self.query(z).squeeze(-1), self.recommend(z).squeeze(-1), self.critic(pooled).squeeze(-1)


def initial_state(support):
    mask = torch.zeros_like(support)
    mask[:, [0, support.shape[1] - 1]] = 1          # two cheap anchor arms pre-observed
    return support * mask, mask


def sample_episode(values, ctx, model_ids, item_ids, batch, support=8, reward=32, gen=None):
    device = values.device
    m = model_ids[torch.randint(len(model_ids), (batch,), device=device, generator=gen)]
    chosen = torch.rand(batch, len(item_ids), device=device, generator=gen).topk(support + reward, 1).indices
    chosen = item_ids[chosen]
    cells = values[m[:, None], chosen]
    return cells[:, :support].mean(1), cells[:, support:].mean(1), ctx[m]


def supervised(policy, values, ctx, model_ids, items, steps, batch, seed, lr=1e-3, use_ctx=True):
    torch.manual_seed(seed); opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    gen = torch.Generator(device=values.device).manual_seed(seed)
    for step in range(steps):
        sup, rew, c = sample_episode(values, ctx, model_ids, items, batch, gen=gen)
        if not use_ctx: c = torch.zeros_like(c)
        obs, mask = initial_state(sup)
        budget = (0, 2, 4, 8)[step % 4]; reveal = step % (budget + 1)
        if reveal:
            idx = torch.rand_like(mask).masked_fill(mask.bool(), -1).topk(reveal, 1).indices
            mask.scatter_(1, idx, 1); obs = sup * mask
        rem = torch.full((batch,), budget - reveal, device=values.device)
        q, r, _ = policy(obs, mask, rem, c)
        loss = F.binary_cross_entropy_with_logits(r, rew)
        if reveal < budget:
            aq = q.masked_fill(mask.bool(), -1e9)
            soft = torch.softmax((rew / .05).masked_fill(mask.bool(), -1e9), 1)
            loss = loss - .25 * (soft * torch.log_softmax(aq, 1)).sum(1).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1); opt.step()


def ppo(policy, values, ctx, model_ids, items, updates, batch, seed, epochs=4, use_ctx=True):
    torch.manual_seed(seed); opt = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    dev = values.device; gen = torch.Generator(device=dev).manual_seed(seed)
    for up in range(updates):
        budget = (2, 4, 8)[up % 3]
        sup, final, c = sample_episode(values, ctx, model_ids, items, batch, gen=gen)
        if not use_ctx: c = torch.zeros_like(c)
        obs, mask = initial_state(sup)
        snaps, acts, logps, vals, kinds = [], [], [], [], []
        with torch.no_grad():
            for step in range(budget + 1):
                rem = torch.full((batch,), budget - step, device=dev)
                q, r, v = policy(obs, mask, rem, c)
                terminal = step == budget
                dist = Categorical(logits=r if terminal else q.masked_fill(mask.bool(), -1e9))
                act = dist.sample()
                snaps.append((obs.clone(), mask.clone(), rem)); acts.append(act)
                logps.append(dist.log_prob(act)); vals.append(v)
                kinds.append(torch.full((batch,), terminal, dtype=torch.bool, device=dev))
                if not terminal:
                    mask.scatter_(1, act[:, None], 1); obs = sup * mask
            reward = final.gather(1, acts[-1][:, None]).squeeze(1)
        oa = torch.cat([s[0] for s in snaps]); ma = torch.cat([s[1] for s in snaps])
        ra = torch.cat([s[2] for s in snaps]); ca = c.repeat(budget + 1, 1)
        aa, lo, vo = map(torch.cat, (acts, logps, vals)); ta = torch.cat(kinds)
        ret = reward.repeat(budget + 1); adv = ret - vo; adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        for _ in range(epochs):
            q, r, value = policy(oa, ma, ra, ca)
            logits = torch.where(ta[:, None], r, q.masked_fill(ma.bool(), -1e9))
            dist = Categorical(logits=logits); ratio = (dist.log_prob(aa) - lo).exp()
            pl = -torch.minimum(ratio * adv, torch.clamp(ratio, .8, 1.2) * adv).mean()
            cl = F.mse_loss(value, ret); aux = F.binary_cross_entropy_with_logits(r[-batch:], final)
            loss = pl + .5 * cl + .1 * aux - .01 * dist.entropy().mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1); opt.step()


@torch.no_grad()
def gp_posterior(policy, support, mask):
    feats = policy.features.clone(); feats[:, 10:11] /= .5
    kernel = .04 * torch.exp(-.5 * torch.cdist(feats, feats).square())
    count = int(mask[0].sum())
    idx = mask.topk(count, 1).indices
    small = kernel[idx[:, :, None], idx[:, None, :]]
    cross = kernel[:, idx].permute(1, 0, 2)
    ps = policy.prior[idx]; noise = .0025 + ps.clamp(.05, .95) * (1 - ps.clamp(.05, .95)) / 8
    sysm = small + torch.diag_embed(noise)
    resid = support.gather(1, idx) - ps
    rhs = torch.cat([resid[:, :, None], cross.transpose(1, 2)], -1)
    sol = torch.linalg.solve(sysm, rhs)
    mean = policy.prior + (cross * sol[:, None, :, 0]).sum(-1)
    var = (kernel.diag() - (cross * sol[:, :, 1:].transpose(1, 2)).sum(-1)).clamp_min(0)
    return mean, var.sqrt()


@torch.no_grad()
def rollout(policy, support, ctx, budget, method):
    obs, mask = initial_state(support); b = len(obs); dev = obs.device
    i0, m0 = obs.clone(), mask.clone()
    if method == "fixed":
        return policy.prior.argmax().expand(b)
    if method == "gp_ucb":
        for _ in range(budget):
            mean, std = gp_posterior(policy, support, mask)
            mask.scatter_(1, (mean + 1.5 * std).masked_fill(mask.bool(), -1e9).argmax(1)[:, None], 1)
        mean, _ = gp_posterior(policy, support, mask); return mean.argmax(1)
    if method == "random":
        for _ in range(budget):
            mask.scatter_(1, torch.rand_like(mask).masked_fill(mask.bool(), -1).argmax(1)[:, None], 1)
        return ((4 * policy.prior + 8 * support * mask) / (4 + 8 * mask)).argmax(1)
    use_ctx = method != "no_context"
    c = ctx if use_ctx else torch.zeros_like(ctx)
    steps = 0 if method == "one_shot" else budget
    for step in range(steps):
        vo, vm = (i0, m0) if method == "no_feedback" else (obs, mask)
        q, _, _ = policy(vo, vm, torch.full((b,), budget - step, device=dev), c)
        mask.scatter_(1, q.masked_fill(mask.bool(), -1e9).argmax(1)[:, None], 1); obs = support * mask
    vo, vm = (i0, m0) if method == "no_feedback" else (obs, mask)
    _, r, _ = policy(vo, vm, torch.zeros(b, device=dev), c)
    return r.argmax(1)


@torch.no_grad()
def evaluate(policy_ctx, policy_noctx, values, ctx, model_ids, calib, evalitems, episodes, seed):
    rows = []
    methods = {"fixed": policy_ctx, "random": policy_ctx, "gp_ucb": policy_ctx,
               "one_shot": policy_ctx, "ppo": policy_ctx, "no_feedback": policy_ctx,
               "no_context": policy_noctx}
    for m in model_ids:
        gen = torch.Generator(device=values.device).manual_seed(seed + int(m))
        draw = torch.rand(episodes, len(calib), device=values.device, generator=gen).topk(8, 1).indices
        support = values[m, calib[draw]].mean(1)
        rates = values[m, evalitems].mean(0)
        cvec = ctx[m][None].expand(episodes, -1)
        for name, pol in methods.items():
            for budget in (0, 2, 4, 8):
                action = rollout(pol, support, cvec, budget, "ppo" if name == "no_context" else name)
                sel = rates[action]
                rows.append(dict(model=int(m), method=name, budget=budget,
                                 gated=float(sel.mean()), regret=float((rates.max() - sel).mean())))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("results/context_selector_train_20260903"))
    ap.add_argument("--outdir", type=Path, default=Path("results/context_selector_run_20260903"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    ap.add_argument("--supervised-steps", type=int, default=800)
    ap.add_argument("--ppo-updates", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--episodes", type=int, default=128)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA")
    a.outdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    man = json.loads((a.data / "manifest.json").read_text())
    blob = np.load(a.data / "context_values.npz")
    Jt = blob["J"]; ctx_np = blob["context"]; feats = blob["features"]; split_labels = blob["splits"]
    # standardize context features across targets
    ctx_np = (ctx_np - ctx_np.mean(0)) / (ctx_np.std(0) + 1e-6)
    families = man["families"]; tags = man["targets"]; T = len(tags)
    values = torch.tensor(Jt, device="cuda"); features = torch.tensor(feats, device="cuda")
    ctx = torch.tensor(ctx_np, device="cuda")
    splits = {s: torch.tensor(np.where(split_labels == s)[0], device="cuda") for s in ("train", "validation", "test")}
    fam_list = sorted(set(families))
    run_cfg = dict(reward="joint reconstruction-gated ASR", holdout="family leave-one-out",
                   targets=tags, families=families, family_list=fam_list, dataset_shape=list(Jt.shape),
                   arms=man["arms"], context_features=man["context_features"], seeds=a.seeds,
                   code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.outdir / "config.json").write_text(json.dumps(run_cfg, indent=2) + "\n")

    records = []
    for fi, test_fam in enumerate(fam_list):
        val_fam = fam_list[(fi + 1) % len(fam_list)]
        train_ids = [i for i in range(T) if families[i] not in (test_fam, val_fam)]
        test_ids = [i for i in range(T) if families[i] == test_fam]
        if not train_ids or not test_ids:
            continue
        prior = values[train_ids][:, splits["train"]].mean((0, 1))
        model_ids = torch.tensor(train_ids, device="cuda")
        for seed in a.seeds:
            torch.manual_seed(seed)
            pol = ContextPolicy(features, prior, ctx.shape[1]).cuda()
            supervised(pol, values, ctx, model_ids, splits["train"], a.supervised_steps, a.batch_size, seed)
            noctx = copy.deepcopy(pol)
            ppo(pol, values, ctx, model_ids, splits["train"], a.ppo_updates, a.batch_size, seed + 10000, use_ctx=True)
            ppo(noctx, values, ctx, model_ids, splits["train"], a.ppo_updates, a.batch_size, seed + 10000, use_ctx=False)
            test = evaluate(pol.eval(), noctx.eval(), values, ctx, torch.tensor(test_ids, device="cuda"),
                            splits["train"], splits["test"], a.episodes, 8900)
            for r in test:
                r.update(test_family=test_fam, seed=seed, model_tag=tags[r["model"]])
                records.append(r)
            print(json.dumps({"fold": test_fam, "seed": seed, "test_targets": len(test_ids)}), flush=True)
            (a.outdir / "test_results.json").write_text(json.dumps(records, indent=2) + "\n")

    def agg(method, budget):
        v = np.array([r["gated"] for r in records if r["method"] == method and r["budget"] == budget])
        reg = np.array([r["regret"] for r in records if r["method"] == method and r["budget"] == budget])
        b = np.array([np.random.default_rng(s).choice(v, len(v)).mean() for s in range(2000)]) if len(v) else np.array([np.nan])
        return dict(gated=float(v.mean()) if len(v) else None,
                    ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if len(v) else None,
                    regret=float(reg.mean()) if len(reg) else None, n=len(v))
    methods = ["fixed", "random", "gp_ucb", "one_shot", "no_feedback", "no_context", "ppo"]
    summary = dict(scope="Contextual joint-ASR selector, family leave-one-out.",
                   targets=tags, families_held=fam_list,
                   results={m: {b: agg(m, b) for b in (0, 2, 4, 8)} for m in methods})
    (a.outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n== TEST joint gated ASR (family-LOO, held-out family's targets) ==")
    for b in (0, 2, 4, 8):
        print(f"  budget={b}: " + "  ".join(f"{m}={summary['results'][m][b]['gated']:.3f}"
                                             for m in methods if summary['results'][m][b]['gated'] is not None))

if __name__ == "__main__":
    main()
