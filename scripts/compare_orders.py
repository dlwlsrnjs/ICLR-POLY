#!/usr/bin/env python3
"""Compare empirical order (results/sequential_add_20260905) vs resource order
(results/sequential_resource_20260906) for the sequential greedy-add, on the same targets.
Greedy: patience=2, margin=0.05, recon floor 0.4."""
import json, glob, os
from pathlib import Path
import numpy as np
MARGIN=0.05; PAT=2; FLOOR=0.4
def load(P):
    D={}
    for f in glob.glob(str(Path(P)/"*.json")):
        b=os.path.basename(f)
        if b in ("aggregate.json",) or b.startswith("panel_"): continue
        d=json.loads(open(f).read()); D[d["target"]]={s["k"]:s for s in d["steps"]}
    return D
def greedy(R):
    ks=sorted(R); bk=ks[0]; best=R[ks[0]]["gated"]; bad=0; pr=[ks[0]]
    for k in ks[1:]:
        pr.append(k); g=R[k]["gated"]
        if g>best: best=g; bk=k; bad=0
        elif g<best-MARGIN or R[k]["recon"]<FLOOR:
            bad+=1
            if bad>=PAT: break
        else: bad=0
    return R[bk]["gated"], len(pr), bk
E=load("results/sequential_add_20260905"); Rr=load("results/sequential_resource_20260906")
tags=sorted(set(E)&set(Rr))
print(f"공통 대상 {len(tags)}: {tags}\n")
print(f"{'target':24s}{'경험적 greedy':14s}{'자원 greedy':13s}{'경험 oracle':12s}{'자원 oracle'}")
eg,rg,eo,ro=[],[],[],[]
for t in tags:
    egj,eq,ek=greedy(E[t]); rgj,rq,rk=greedy(Rr[t])
    eoj=max(E[t][k]["gated"] for k in E[t]); roj=max(Rr[t][k]["gated"] for k in Rr[t])
    eg.append(egj); rg.append(rgj); eo.append(eoj); ro.append(roj)
    print(f"{t:24s}{egj:.3f}(q{eq})     {rgj:.3f}(q{rq})    {eoj:.3f}       {roj:.3f}")
eg,rg,eo,ro=map(np.array,(eg,rg,eo,ro))
print(f"\n그리디 평균: 경험적 {eg.mean():.3f} | 자원 {rg.mean():.3f} | 차 {eg.mean()-rg.mean():+.3f}")
print(f"오라클 평균: 경험적 {eo.mean():.3f} | 자원 {ro.mean():.3f} | 차 {eo.mean()-ro.mean():+.3f}")
