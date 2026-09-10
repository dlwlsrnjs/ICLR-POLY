#!/usr/bin/env python3
"""Run the fixed-budget GP-BAI bandit on the NEW arm set (amount n=2..10 at delta=0, plus
disorder delta=0.25..1.0 at n=4), using the collected resource+disorder data as the arm-value
oracle. Compares GP-BAI (budget 4 and 6) to the recon-greedy policy, the single fixed arm, and
the full oracle. Prior = leave-one-out mean over the other targets (structured/data prior)."""
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

tags = sorted(set(SEQ) & set(DIS))          # models with both axes
# arm set
AMT = [("amt", k) for k in range(1, 10)]     # n = k+1, delta=0
DISO = [("dis", dv) for dv in (0.25, 0.5, 0.75, 1.0)]  # n=4, delta=dv
ARMS = AMT + DISO
def feat(arm):
    kind, lv = arm
    if kind == "amt":
        return [(lv + 1) / 10.0, 0.0, 0.0]
    return [4 / 10.0, lv, 1.0]
FE = np.array([feat(a) for a in ARMS], float)

def value(t, arm):
    kind, lv = arm
    if kind == "amt":
        return SEQ[t][lv]["gated"]
    return DIS[t][lv]["gated"]

V = {t: np.array([value(t, a) for a in ARMS]) for t in tags}

def recon_greedy(t):
    """recon-gated greedy on amount arms only (the interpretable policy)."""
    R = SEQ[t]; ks = sorted(R)
    good = [k for k in ks if R[k]["recon"] >= 0.6]; k0 = max(good) if good else ks[0]
    bk = k0; best = R[k0]["gated"]; bad = 0; k = k0
    while k < ks[-1]:
        k += 1
        if R[k]["recon"] < 0.4: break
        g = R[k]["gated"]
        if g > best: best = g; bk = k; bad = 0
        else:
            bad += 1
            if bad >= 1: break
    if k0 - 1 >= ks[0] and R[k0 - 1]["gated"] > best: best = R[k0 - 1]["gated"]
    return best

# single fixed arm = best-on-average over targets
fixed_arm = int(np.argmax(np.mean([V[t] for t in tags], axis=0)))
rng = np.random.default_rng(0)

def run(budget):
    vals = []
    for t in tags:
        others = [x for x in tags if x != t]
        prior = np.mean([V[x] for x in others], axis=0)          # LOO data prior
        probe = lambda a, t=t: float(V[t][a] + rng.normal(0, 0.02))
        rec, _ = gp_bai(prior, FE, probe, budget)
        vals.append(V[t][rec])
    return np.array(vals)

print(f"대상 {len(tags)} (양+혼란도 둘 다): {tags}")
print(f"arm 수 {len(ARMS)} (양9 + 혼란도4) | 고정 최선 arm = {ARMS[fixed_arm]}\n")
oracle = np.array([V[t].max() for t in tags])
fixed = np.array([V[t][fixed_arm] for t in tags])
greedy = np.array([recon_greedy(t) for t in tags])
print(f"{'method':28s}{'평균 J':9s}{'vs고정':9s}{'regret':8s}쿼리")
print(f"{'오라클(전 arm)':28s}{oracle.mean():.3f}    {oracle.mean()-fixed.mean():+.3f}   {0.0:.3f}   {len(ARMS)}")
print(f"{'단일 고정':28s}{fixed.mean():.3f}    {0.0:+.3f}   {oracle.mean()-fixed.mean():.3f}   1")
print(f"{'recon-greedy(양만)':28s}{greedy.mean():.3f}    {greedy.mean()-fixed.mean():+.3f}   {oracle.mean()-greedy.mean():.3f}   ~2")
for B in (4, 6):
    g = run(B)
    print(f"{'GP-BAI(예산'+str(B)+')':28s}{g.mean():.3f}    {g.mean()-fixed.mean():+.3f}   {oracle.mean()-g.mean():.3f}   {B}")
print("\n주: GP-BAI는 양+혼란도 전 arm을 탐색(혼란도 축 포함)하므로 정렬-강 모델(14B)의 혼란도 이득도 포착 가능.")
