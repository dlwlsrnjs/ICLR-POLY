#!/usr/bin/env python3
"""GP-BAI over the unified amount+disorder arm set, comparing PRIORS for warm-start:
  flat        : constant prior (= plain GP-UCB, no domain knowledge).
  benign-recon: prior_mean(arm) = recon(arm) * u0  -- uses ONLY the reconstruction rate
                (benign-observable, no harmful supervision) to rank arms. This is the
                benign-driven warm start: solvable arms get higher expected joint.
  loo-data    : leave-one-out mean over other targets (data prior, needs a prior corpus).
Reports the budget-accuracy curve (mean recommended J vs query budget) for each prior, so we can
show benign warm-start reaching near-oracle at a smaller budget. Replay over collected data."""
import json, glob, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gp_bai import gp_bai

SEQ = {}
for f in glob.glob("results/sequential_resource_20260906/*.json"):
    b = os.path.basename(f)
    if b.startswith("panel_") or b.endswith("aggregate.json") or "_20260906.json" in b: continue
    d = json.loads(open(f).read())
    if "steps" in d: SEQ[d["target"]] = {s["k"]: s for s in d["steps"]}
DIS = {}
for f in glob.glob("results/disorder_sweep_20260906/*.json"):
    b = os.path.basename(f)
    if b.startswith("panel_"): continue
    d = json.loads(open(f).read()); DIS[d["target"]] = {r["delta"]: r for r in d["rows"]}

tags = sorted(set(SEQ) & set(DIS))
AMT = [("amt", k) for k in range(1, 10)]
DISO = [("dis", dv) for dv in (0.25, 0.5, 0.75, 1.0)]
ARMS = AMT + DISO
def feat(a):
    kind, lv = a
    return [(lv + 1) / 10.0, 0.0, 0.0] if kind == "amt" else [0.4, lv, 1.0]
FE = np.array([feat(a) for a in ARMS], float)
def gated(t, a):
    kind, lv = a
    return SEQ[t][lv]["gated"] if kind == "amt" else DIS[t][lv]["gated"]
def recon(t, a):
    kind, lv = a
    return SEQ[t][lv]["recon"] if kind == "amt" else DIS[t][lv]["recon"]
V = {t: np.array([gated(t, a) for a in ARMS]) for t in tags}
RC = {t: np.array([recon(t, a) for a in ARMS]) for t in tags}
U0 = 0.6   # typical unsafe among solvable arms (fixed constant, not fit on harmful)

def prior_of(t, kind):
    if kind == "flat":
        return np.full(len(ARMS), float(np.mean([V[x].mean() for x in tags])))
    if kind == "benign-recon":
        return RC[t] * U0                       # benign-observable recon -> expected joint
    if kind == "loo-data":
        return np.mean([V[x] for x in tags if x != t], axis=0)
    raise ValueError(kind)

oracle = np.mean([V[t].max() for t in tags])
fixed_arm = int(np.argmax(np.mean([V[t] for t in tags], axis=0)))
fixed = np.mean([V[t][fixed_arm] for t in tags])
rng = np.random.default_rng(0)

def run(kind, B, reps=20):
    accs = []
    for _ in range(reps):
        vs = []
        for t in tags:
            pr = prior_of(t, kind)
            probe = lambda a, t=t: float(V[t][a] + rng.normal(0, 0.02))
            rec, _ = gp_bai(pr, FE, probe, B)
            vs.append(V[t][rec])
        accs.append(np.mean(vs))
    return float(np.mean(accs))

print(f"대상 {len(tags)} | arm {len(ARMS)}(양9+혼란도4) | 오라클 {oracle:.3f} | 고정 {fixed:.3f}\n")
print(f"{'예산':6s}{'flat(UCB)':12s}{'benign-recon':14s}{'loo-data'}")
for B in (2, 3, 4, 5, 6, 8):
    row = {k: run(k, B) for k in ("flat", "benign-recon", "loo-data")}
    print(f"{B:<6}{row['flat']:.3f}       {row['benign-recon']:.3f}         {row['loo-data']:.3f}")
print(f"\n오라클={oracle:.3f} 고정={fixed:.3f}. benign-recon = 무해관측(재구성)만으로 warm-start(유해감독 0).")
