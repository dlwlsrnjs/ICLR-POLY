#!/usr/bin/env python3
"""Honest, leak-free re-evaluation of the Lingua selector: the warm-start prior is the GENUINE benign
reconstruction (harmless FLORES puzzles, string metric, disjoint items) from benign_arms_probe, NOT the
recon field of the harmful evals. Compares priors {genuine-benign, leaked-benign(recon-of-harmful),
flat, loo-data} under realistic binomial probe noise. Also reports the benign-recon vs gated Spearman."""
import json, sys, io, contextlib
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, "scripts")
from gp_bai import gp_bai
with contextlib.redirect_stdout(io.StringIO()):
    import lingua_bandit_full as LG

M = LG.MODELS; G = LG.G; RC = LG.RC; FE = LG.FE; names = LG.names
BEN = {}
for t in M:
    d = json.load(open(f"results/benign_arms_20260907/{t}.json"))["arm_recon"]
    BEN[t] = np.array([d[n] for n in names])           # genuine benign recon per arm, in bandit arm order

n_items = 40
rng = np.random.default_rng(0)


def run(prior_kind, B, reps=300, noise="binom"):
    accs = []
    for _ in range(reps):
        vs = []
        for t in M:
            if prior_kind == "genuine-benign":
                pr = BEN[t] * 0.6
            elif prior_kind == "leaked-benign":
                pr = RC[t] * 0.6
            elif prior_kind == "flat":
                pr = np.full(len(names), float(np.mean([G[x].mean() for x in M])))
            else:
                pr = np.mean([G[x] for x in M if x != t], axis=0)
            if noise == "binom":
                probe = lambda a, t=t: float(np.clip(G[t][a] + rng.normal(0, np.sqrt(max(G[t][a]*(1-G[t][a]), .01)/n_items)), 0, 1))
            else:
                probe = lambda a, t=t: float(G[t][a] + rng.normal(0, 0.02))
            rec, _ = gp_bai(pr, FE, probe, B); vs.append(G[t][rec])
        accs.append(np.mean(vs))
    return float(np.mean(accs))


print("=== Lingua honest re-eval: prior = GENUINE benign recon (leak-free), binomial probe noise ===")
print(f"{'budget':7s}{'genuine-benign':16s}{'leaked-benign':15s}{'flat':8s}{'loo-data':10s}")
for B in (1, 2, 3, 4, 6):
    print(f"{B:<7}{run('genuine-benign', B):.3f}           {run('leaked-benign', B):.3f}          {run('flat', B):.3f}   {run('loo-data', B):.3f}")
b = json.load(open("results/lingua_bandit_full_20260906.json"))
print(f"\noracle {b['oracle']:.3f} | best fixed arm {b['fixed']:.3f}")

# genuine benign recon -> gated Spearman, per model
print("\n=== genuine-benign recon vs gated Spearman (per target) ===")
rs = []
for t in M:
    r = float(spearmanr(BEN[t], G[t]).correlation); rs.append(r)
    print(f"  {t:15s} rho={r:+.3f}")
print(f"  mean rho = {np.nanmean(rs):+.3f}")

summ = {"budget": {B: {k: round(run(k, B), 3) for k in ("genuine-benign", "leaked-benign", "flat", "loo-data")} for B in (1, 2, 3, 4, 6)},
        "oracle": b["oracle"], "fixed": b["fixed"],
        "benign_gated_spearman": {t: round(float(spearmanr(BEN[t], G[t]).correlation), 3) for t in M}}
Path("results/lingua_benign_honest_20260907.json").write_text(json.dumps(summ, indent=2))
print("\nsaved results/lingua_benign_honest_20260907.json")
