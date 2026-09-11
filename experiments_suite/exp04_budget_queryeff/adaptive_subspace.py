#!/usr/bin/env python3
"""EXP04g ONLINE adaptive-subspace BO (ARD / SAAS-style): learn which arm-feature axes matter FROM the
harmful observations as they accrue, and prune the uninformative ones online -- instead of a one-shot
benign-probe drop. This is the deployment method: subspace BO with axis-aligned relevance learned online.

Mechanism (tractable SAAS-lite, no external deps):
  every step (after >= warm obs) fit sparse axis relevances w_d from (X_obs, y_obs) by ridge + soft-
  threshold (L1 surrogate). Per-dim inverse-lengthscale rho_d ∝ |w_d| (pruned when ~0). The GP kernel
  uses rho, so exploration concentrates on the active subspace; the benign prior only warm-starts.
Feature axes: [F, n, shuffled | persona, fiction, pap, role, plain]. Logs the active axes per step.

Compares (replay over the stored matrix): random, gp_ucb_fixed, benign_drop (one-shot), adaptive_subspace.
Usage: python adaptive_subspace.py --root <exp02 results> --suffix _mj [--budget 8] [--reps 24]"""
import argparse, json, glob
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import search_sota as S

DIM = ["F", "n", "shuffled", "persona", "fiction", "pap", "role", "plain"]


def feats8(a):
    c, w = S.feats(a); return np.concatenate([c, w])


def ridge_soft(Xo, yo, lam=0.15, l2=1.0):
    Xc = Xo - Xo.mean(0); yc = yo - yo.mean()
    d = Xc.shape[1]
    beta = np.linalg.solve(Xc.T @ Xc + l2 * np.eye(d), Xc.T @ yc)
    s = np.sign(beta) * np.clip(np.abs(beta) - lam, 0, None)   # soft-threshold -> sparsity
    return s


def kernel(X, rho):
    d = (X[:, None, :] - X[None, :, :]) * rho[None, None, :]
    return np.exp(-(d ** 2).sum(-1) / 2)


def adaptive_run(arms, X, ver, prior, budget, beta, rng, warm=3, base_rho=1.2):
    n, d = X.shape
    pv = np.array([prior.get(a, 0.5) for a in arms]); pv = pv / (pv.max() + 1e-9)
    m = np.full(n, 0.5); obs, ys = [], []; best = 0.0; traj = []
    rho = np.full(d, base_rho)
    for t in range(budget):
        if len(obs) >= warm:
            w = ridge_soft(X[obs], np.array(ys))
            aw = np.abs(w)
            if aw.max() > 1e-6:
                rho = base_rho * (aw / aw.max())          # relevant dims -> large rho (short ls); pruned -> 0
                rho = np.clip(rho, 0.05, None)
        K = kernel(X, rho)
        if not obs:
            mu = m.copy(); sd = np.ones(n)
        else:
            I = np.array(obs); Kinv = np.linalg.inv(K[np.ix_(I, I)] + 0.02 * np.eye(len(I)))
            Ks = K[:, I]; mu = m + Ks @ Kinv @ (np.array(ys) - m[I])
            sd = np.sqrt(np.clip(1 - np.einsum("ij,jk,ik->i", Ks, Kinv, Ks), 1e-6, None))
        acq = mu + beta * sd + rng.normal(0, 1e-6, n)
        for j in obs:
            acq[j] = -1e9
        pick = int(rng.integers(n)) if not obs else int(np.argmax(acq))
        y = ver[arms[pick]]; obs.append(pick); ys.append(y); best = max(best, y)
        active = [DIM[i] for i in range(d) if rho[i] > 0.1 * base_rho]
        traj.append(dict(step=t + 1, arm=arms[pick], verified=round(float(y), 3),
                         best=round(best, 3), active_axes=active))
    return best, traj


def avg_adaptive(arms, X, ver, prior, budget, beta, reps):
    bs = []; last = None
    for s in range(reps):
        b, tr = adaptive_run(arms, X, ver, prior, budget, beta, np.random.default_rng(s))
        bs.append([x["best"] for x in tr]); last = tr
    return np.array(bs).mean(0), last


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True); ap.add_argument("--suffix", default="_mj")
    ap.add_argument("--budget", type=int, default=8); ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--beta", type=float, default=2.0)
    a = ap.parse_args()
    ver_all, pri = S.load(a.root, a.suffix)          # saturation-aware prior (benign drop baseline)
    out = {}
    for tag in sorted(ver_all):
        ver = ver_all[tag]; arms = list(ver)
        if len(arms) < 292:
            print(f"skip {tag}: incomplete ({len(arms)}/292)"); continue
        X = np.array([feats8(x) for x in arms]); oracle = max(ver.values())
        rng = np.random.default_rng(0)
        rand = [round(float(np.mean([max(ver[x] for x in list(np.random.default_rng(s).choice(arms, k, replace=False)))
                for s in range(200)])), 3) for k in range(1, a.budget + 1)]
        fixed = S.avg(arms, ver, pri[tag], a.budget, a.reps, acq="ucb", additive=True, use_pibo=True)
        adap, last = avg_adaptive(arms, X, ver, pri[tag], a.budget, a.beta, a.reps)
        adap = [round(float(x), 3) for x in adap]
        out[tag] = dict(oracle=round(oracle, 3), random=rand, gp_ucb_fixed_bnprior=fixed,
                        adaptive_subspace=adap, example_trajectory=last)
        print(f"[{tag}] oracle={out[tag]['oracle']}")
        print(f"   random           {rand}")
        print(f"   gp_ucb+bn-drop   {fixed}")
        print(f"   adaptive_subspace{adap}")
        print(f"   final active axes (one run): {last[-1]['active_axes']}")
    Path(a.root, f"adaptive_subspace{a.suffix}.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
