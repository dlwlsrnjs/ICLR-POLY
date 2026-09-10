#!/usr/bin/env python3
"""GP-BAI over the FULL arm set incl. combination arms: amount(n2..10) + disorder(δ.25..1 @n4)
+ combo(persona, incept @n4). benign-recon warm-start. Shows the new ceiling + budget curve."""
import json, glob, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gp_bai import gp_bai

def load(P, key):
    D = {}
    for f in glob.glob(str(Path(P) / "*.json")):
        b = os.path.basename(f)
        if b.startswith("panel_") or b.endswith("aggregate.json") or "_20260906.json" in b: continue
        d = json.loads(open(f).read()); D[d["target"]] = d
    return D
SEQ = {t: {s["k"]: s for s in d["steps"]} for t, d in load("results/sequential_resource_20260906", "steps").items() if "steps" in d}
DIS = {t: {r["delta"]: r for r in d["rows"]} for t, d in load("results/disorder_sweep_20260906", "rows").items()}
CMB = {t: d["variants"] for t, d in load("results/combo_20260906", "variants").items()}
tags = sorted(set(SEQ) & set(DIS) & set(CMB))
AMT = [("amt", k) for k in range(1, 10)]
DISO = [("dis", dv) for dv in (0.25, 0.5, 0.75, 1.0)]
COMBO = [("combo", "ours_persona"), ("combo", "ours_incept")]
ARMS = AMT + DISO + COMBO
def feat(a):
    k, lv = a
    if k == "amt": return [(lv + 1) / 10, 0, 0, 0]
    if k == "dis": return [0.4, lv, 1, 0]
    return [0.4, 0, 0, 1]
FE = np.array([feat(a) for a in ARMS], float)
def gv(t, a):
    k, lv = a
    if k == "amt": return SEQ[t][lv]["gated"]
    if k == "dis": return DIS[t][lv]["gated"]
    return CMB[t][lv]["gated"]
def rv(t, a):
    k, lv = a
    if k == "amt": return SEQ[t][lv]["recon"]
    if k == "dis": return DIS[t][lv]["recon"]
    return CMB[t][lv]["recon"]
V = {t: np.array([gv(t, a) for a in ARMS]) for t in tags}
RC = {t: np.array([rv(t, a) for a in ARMS]) for t in tags}
oracle = np.mean([V[t].max() for t in tags])
fixed_arm = int(np.argmax(np.mean([V[t] for t in tags], axis=0)))
fixed = np.mean([V[t][fixed_arm] for t in tags])
rng = np.random.default_rng(0)
def run(prior_kind, B, reps=20):
    accs = []
    for _ in range(reps):
        vs = []
        for t in tags:
            if prior_kind == "benign-recon": pr = RC[t] * 0.6
            elif prior_kind == "flat": pr = np.full(len(ARMS), float(np.mean([V[x].mean() for x in tags])))
            else: pr = np.mean([V[x] for x in tags if x != t], axis=0)
            probe = lambda a, t=t: float(V[t][a] + rng.normal(0, 0.02))
            rec, _ = gp_bai(pr, FE, probe, B); vs.append(V[t][rec])
        accs.append(np.mean(vs))
    return float(np.mean(accs))
print(f"대상 {len(tags)} | arm {len(ARMS)} (양9+혼란도4+결합2) | 오라클 {oracle:.3f} | 고정 {fixed:.3f}\n")
# per-model oracle incl combo
print("모델별 전-arm 오라클 (결합 포함):")
for t in tags:
    best_a = ARMS[int(np.argmax(V[t]))]
    print(f"  {t:16s} {V[t].max():.3f}  best_arm={best_a}")
print(f"\n{'예산':6s}{'flat':10s}{'benign-recon':14s}{'loo-data'}")
for B in (2, 3, 4, 6, 8):
    print(f"{B:<6}{run('flat',B):.3f}     {run('benign-recon',B):.3f}         {run('loo-data',B):.3f}")
print(f"\n오라클(결합포함)={oracle:.3f}. 결합 arm 추가로 천장이 크게 상승.")
