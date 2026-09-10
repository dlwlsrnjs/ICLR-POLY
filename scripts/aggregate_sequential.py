#!/usr/bin/env python3
"""Aggregate stage-2 sequential-add: universal fixed language order + per-target greedy stop.
Greedy rule: probe EN+top-k for k=1,2,... in the universal order; keep the running best; stop
when joint drops below (running max - margin) for `patience` consecutive steps, or recon<floor.
Report the greedy-chosen config vs (a) oracle best-k, (b) a single fixed config (best-on-avg k)."""
import json, glob, os
from pathlib import Path
import numpy as np
OUT=Path("results/sequential_add_20260905")
MARGIN=0.05; PATIENCE=2; RECON_FLOOR=0.4
D={}
for f in glob.glob(str(OUT/"*.json")):
    b=os.path.basename(f)
    if b in ("aggregate.json",) or b.startswith("panel_"): continue
    d=json.loads(open(f).read()); D[d["target"]]={s["k"]:s for s in d["steps"]}
tags=sorted(D)
order=json.loads((Path("results/lang_rank_20260905/universal_order.json")).read_text())["order"]
print(f"보편 순서: {order}\n대상 {len(tags)}개\n")

def greedy(R):
    ks=sorted(R); best_k=ks[0]; best=R[ks[0]]["gated"]; bad=0; probed=[ks[0]]
    for k in ks[1:]:
        probed.append(k); g=R[k]["gated"]
        if g>best: best=g; best_k=k; bad=0
        elif g < best-MARGIN or R[k]["recon"]<RECON_FLOOR:
            bad+=1
            if bad>=PATIENCE: break
        else:
            bad=0
    return best_k, R[best_k]["gated"], len(probed)

# fixed = single best-on-average k across targets
kavg={k: np.mean([D[t][k]["gated"] for t in tags if k in D[t]]) for k in range(1,10)}
fixed_k=max(kavg,key=kavg.get)
print(f"{'target':24s}{'greedy k(n)':12s}{'greedy J':9s}{'queries':8s}{'oracle k/J':12s}{'fixed J'}")
rows=[]
for t in tags:
    R=D[t]; gk,gj,q=greedy(R)
    ok=max(R,key=lambda k:R[k]["gated"]); oj=R[ok]["gated"]; fj=R[fixed_k]["gated"]
    rows.append(dict(t=t,gk=gk,gj=gj,q=q,ok=ok,oj=oj,fj=fj))
    print(f"{t:24s}k={gk}(n{gk+1}) {gj:.3f}    {q:<8}k={ok}/{oj:.3f}   {fj:.3f}")
gj=np.array([r["gj"] for r in rows]); oj=np.array([r["oj"] for r in rows]); fj=np.array([r["fj"] for r in rows])
q=np.array([r["q"] for r in rows])
print(f"\n고정 최선 k={fixed_k} (n{fixed_k+1}) 평균 J = {fj.mean():.3f}")
print(f"그리디(보편순서+정지) 평균 J = {gj.mean():.3f} | vs고정 {gj.mean()-fj.mean():+.3f} | 이김 {int(np.sum(gj>fj+1e-9))}/{len(rows)}")
print(f"오라클(최선 k) 평균 J = {oj.mean():.3f} | 그리디 regret {oj.mean()-gj.mean():.3f}")
print(f"그리디 평균 쿼리(라이브 유해 배치) = {q.mean():.1f} / 9")
(OUT/"aggregate.json").write_text(json.dumps(dict(order=order,fixed_k=fixed_k,rows=rows,
    greedy_mean=float(gj.mean()),fixed_mean=float(fj.mean()),oracle_mean=float(oj.mean()),
    mean_queries=float(q.mean())),indent=2)+"\n")
print(f"\n저장: {OUT}/aggregate.json")
