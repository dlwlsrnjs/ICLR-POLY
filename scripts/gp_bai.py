#!/usr/bin/env python3
"""Prior-warm-started fixed-budget Bayesian Best-Arm Identification (GP-BAI) for our setting.

Our problem: black-box target, expensive noisy binary joint reward, structured finite arms
(quasi-concave over the (n,F,arr) grid), few queries, and a behavioral fingerprint that
predicts a good arm. The matching algorithm (Nguyen et al. 2402.05878 prior-dependent
fixed-budget BAI in structured bandits; Atsidakou et al. 2211.08572; Combes-Proutiere
unimodal structure) is:

  prior mean  m0(arm) = structured/extended policy J_hat(arm | fingerprint)   [warm start]
  GP surrogate over arm FEATURES with a smoothness kernel                    [quasi-concavity]
  fixed-budget allocation: at each step probe the arm maximizing an
     information/UCB score under the GP posterior; after the budget, RECOMMEND
     argmax posterior mean (best-arm identification, not cumulative regret).

Evaluated in replay on the measured 32-arm joint table: for each held-out target, run the
loop querying calibration items (deterministic temp-0 => replay == live), and score the
recommended arm's true gated ASR on eval items. Compared to: fixed, random, GP-UCB (no
prior mean, cumulative-regret style), and structured-only (query-0). Reports gated ASR and
regret vs oracle at matched query budgets.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np

CELL = re.compile(r"frag(\d+)_(ordered|shuffled)_n(\d+)")


def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def arm_feats(arm_names):
    F = np.array([int(CELL.match(a).group(1)) for a in arm_names], float) / 12.0
    n = np.array([int(CELL.match(a).group(3)) for a in arm_names], float) / 10.0
    shuf = np.array([CELL.match(a).group(2) == "shuffled" for a in arm_names], float)
    return np.stack([n, F, shuf], 1)                       # [A, 3]


def structured_prior_mean(theta, arm_names, A, C_lo, C_hi, M):
    """Extended-structured J_hat over arms for one target's fingerprint (the warm-start prior)."""
    a_n, a_F, a_shuf, c0, c1, k, u0, u_a, u_d, u_m, u_am = theta
    n = np.array([int(CELL.match(a).group(3)) for a in arm_names], float)
    F = np.array([int(CELL.match(a).group(1)) for a in arm_names], float)
    shuf = np.array([CELL.match(a).group(2) == "shuffled" for a in arm_names], float)
    D = a_n * (n / 10) + a_F * (F / 12) + a_shuf * shuf
    cap = C_lo + (C_hi - C_lo) * ((n - 2) / 8.0)
    R = sigmoid(k * (c0 + c1 * cap - D))
    U = sigmoid(u0 + u_a * A + u_d * D + u_m * M + u_am * A * M)
    return R * U


def gp_bai(prior_mean, feats, probe_fn, budget, ls=0.35, sig=0.18, noise=0.02, explore=1.0):
    """Fixed-budget BAI with GP posterior on residual over prior_mean. Returns recommended arm."""
    A = len(prior_mean)
    K = sig**2 * np.exp(-0.5 * ((feats[:, None, :] - feats[None, :, :])**2).sum(-1) / ls**2)
    obs_idx, obs_y = [], []
    def posterior():
        if not obs_idx:
            return prior_mean.copy(), np.sqrt(np.diag(K))
        idx = np.array(obs_idx); y = np.array(obs_y)
        Koo = K[np.ix_(idx, idx)] + noise * np.eye(len(idx))
        Koo_inv = np.linalg.inv(Koo)
        resid = y - prior_mean[idx]
        Kxi = K[:, idx]
        mean = prior_mean + Kxi @ (Koo_inv @ resid)
        var = np.clip(np.diag(K) - np.einsum("ai,ij,aj->a", Kxi, Koo_inv, Kxi), 1e-9, None)
        return mean, np.sqrt(var)
    for _ in range(budget):
        mean, sd = posterior()
        score = mean + explore * sd
        for j in obs_idx:
            score[j] = -1e9                                # BAI: don't re-probe (pure exploration budget)
        a = int(np.argmax(score))
        obs_idx.append(a); obs_y.append(probe_fn(a))
    mean, _ = posterior()
    return int(np.argmax(mean)), obs_idx                   # RECOMMEND posterior-mean argmax


def gp_bai_adaptive(prior_mean, feats, probe_fn, b_min=2, b_max=6, stop_tau=0.5,
                    ls=0.35, sig=0.18, noise=0.02, explore=1.0):
    """Difficulty-adaptive GP-BAI. Same acquisition and recommendation as gp_bai, but the number of
    queries is not fixed: after at least b_min queries it STOPS as soon as an arm has been observed at
    verified success >= stop_tau (a strong attack was found -> an easy target), and otherwise keeps
    querying up to b_max (a hard target earns more attempts). This spends the budget where it is
    needed -- big-hard models get more queries, big-easy ones fewer -- rather than by raw model size.
    stop_tau / b_min / b_max are fixed a priori, not tuned per target. Returns (rec_arm, obs_idx);
    len(obs_idx) is the queries actually used."""
    A = len(prior_mean)
    K = sig**2 * np.exp(-0.5 * ((feats[:, None, :] - feats[None, :, :])**2).sum(-1) / ls**2)
    obs_idx, obs_y = [], []
    def posterior():
        if not obs_idx:
            return prior_mean.copy(), np.sqrt(np.diag(K))
        idx = np.array(obs_idx); y = np.array(obs_y)
        Koo = K[np.ix_(idx, idx)] + noise * np.eye(len(idx))
        Koo_inv = np.linalg.inv(Koo)
        resid = y - prior_mean[idx]
        Kxi = K[:, idx]
        mean = prior_mean + Kxi @ (Koo_inv @ resid)
        var = np.clip(np.diag(K) - np.einsum("ai,ij,aj->a", Kxi, Koo_inv, Kxi), 1e-9, None)
        return mean, np.sqrt(var)
    for step in range(b_max):
        mean, sd = posterior()
        score = mean + explore * sd
        for j in obs_idx:
            score[j] = -1e9
        a = int(np.argmax(score))
        obs_idx.append(a); obs_y.append(probe_fn(a))
        if step + 1 >= b_min and max(obs_y) >= stop_tau:
            break
    mean, _ = posterior()
    return int(np.argmax(mean)), obs_idx


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="results/context_selector_full_train_20260904")
    ap.add_argument("--prior", default="results/structured_policy_ext_20260904/summary.json")
    ap.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--calib", type=int, default=25)
    ap.add_argument("--outdir", default="results/gp_bai_20260904")
    a = ap.parse_args()
    blob = np.load(Path(a.data) / "context_values.npz")
    man = json.loads((Path(a.data) / "manifest.json").read_text())
    J = blob["J"]; ctx = blob["context"]; splits = blob["splits"]
    arm_names = man["arms"]; tags = man["targets"]; fams = man["families"]; T = len(tags)
    feats = arm_feats(arm_names)
    theta = json.loads(Path(a.prior).read_text())["theta_global"]
    Afp = ctx[:, 0]; Mfp = ctx[:, 1]; Clo = 0.2 + ctx[:, 3]; Chi = 0.2 + ctx[:, 4]
    tr_i = np.where(splits == "train")[0]; ev_i = np.where(splits == "test")[0]
    Jeval = J[:, ev_i, :].mean(1)                          # [T, A] true gated on eval items
    rng = np.random.default_rng(20260904)

    def probe_env(ti):
        # replay probe: mean joint on a random calibration-item batch (temp-0 => == live)
        def f(arm):
            items = rng.choice(tr_i, size=min(a.calib, len(tr_i)), replace=False)
            return float(np.nanmean(J[ti, items, arm]))
        return f

    methods = ["gp_bai_prior", "gp_ucb_noprior", "random", "fixed", "structured_q0"]
    records = []
    for held in range(T):
        trn = [i for i in range(T) if fams[i] != fams[held]]     # family hold-out (prior fit excludes held family via its own LOO; here prior is global theta)
        prior_m = structured_prior_mean(theta, arm_names, Afp[held], Clo[held], Chi[held], Mfp[held])
        gfix = int(Jeval[trn].mean(0).argmax())
        orc = float(Jeval[held].max())
        pf = probe_env(held)
        for b in a.budgets:
            # GP-BAI with structured prior mean
            rec, _ = gp_bai(prior_m, feats, pf, b)
            g_bai = float(Jeval[held, rec])
            # GP-UCB without prior (flat prior mean = training-arm average), cumulative-style recommend best observed
            flat = Jeval[trn].mean(0)
            rec2, obs2 = gp_bai(flat, feats, pf, b, explore=1.5)
            g_ucb = float(Jeval[held, rec2])
            # random probe then best observed
            ridx = list(rng.choice(len(arm_names), min(b, len(arm_names)), replace=False))
            robs = {i: pf(i) for i in ridx}
            g_rand = float(Jeval[held, max(robs, key=robs.get)]) if robs else float(Jeval[held, gfix])
            # structured query-0 (no probing)
            g_q0 = float(Jeval[held, int(np.argmax(prior_m))])
            records.append(dict(target=tags[held], family=fams[held], budget=b,
                                gp_bai_prior=g_bai, gp_ucb_noprior=g_ucb, random=g_rand,
                                fixed=float(Jeval[held, gfix]), structured_q0=g_q0, oracle=orc))

    def agg(method, b):
        rs = [r for r in records if r["budget"] == b]
        v = np.array([r[method] for r in rs]); orc = np.array([r["oracle"] for r in rs])
        bt = np.array([np.random.default_rng(s).choice(v, len(v)).mean() for s in range(2000)])
        return dict(gated=float(v.mean()), ci=[float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
                    regret=float((orc - v).mean()))
    summary = dict(scope="Prior-warm-started fixed-budget GP-BAI vs baselines, family LOO, replay.",
                   n_targets=T, budgets=a.budgets,
                   results={m: {b: agg(m, b) for b in a.budgets} for m in methods},
                   oracle=float(np.mean([r["oracle"] for r in records if r["budget"] == a.budgets[0]])))
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"오라클(사후 상한): {summary['oracle']:.3f}\n")
    print(f"{'budget':>6} " + "  ".join(f"{m:>16}" for m in methods))
    for b in a.budgets:
        print(f"{b:>6} " + "  ".join(f"{summary['results'][m][b]['gated']:>16.3f}" for m in methods))
    print("\nregret vs oracle:")
    for b in a.budgets:
        print(f"  b={b}: " + "  ".join(f"{m}={summary['results'][m][b]['regret']:.3f}" for m in methods))


if __name__ == "__main__":
    main()
