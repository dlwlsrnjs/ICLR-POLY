#!/usr/bin/env python3
"""MultiJail-DATASET results summary + cross-dataset (Lingua vs MultiJail) consistency.
Handles partially-missing models gracefully."""
import json, glob, os
from pathlib import Path
import numpy as np
def load(P, key=None):
    D={}
    for f in glob.glob(str(Path(P)/"*.json")):
        b=os.path.basename(f)
        if b.startswith("panel_") or b.startswith("log_"): continue
        try: d=json.loads(open(f).read())
        except: continue
        D[d.get("target",b[:-5])]=d
    return D
MB=load("results/mj_method_20260906")      # methods dict
MJ=load("results/mj_multijail_20260906")   # combination
CB=load("results/mj_combo_20260906")       # variants
SQ={t:{s["k"]:s for s in d["steps"]} for t,d in load("results/mj_sequential_20260906").items() if "steps" in d}
DS={t:{r["delta"]:r for r in d["rows"]} for t,d in load("results/mj_disorder_20260906").items() if "rows" in d}
tags=sorted(set(SQ)|set(MB))
strong=["qwen25_7b","qwen25_14b","llama31_8b_it","gemma2_9b_it"]
print("=== MultiJail 데이터셋: 방법 비교 (gated ASR) ===")
print(f"{'model':16s}{'plain':7s}{'transl':7s}{'MJ-comb':8s}{'aim':6s}{'deepinc':8s}{'ours_combo':11s}{'ours_amt/dis(오라클)'}")
rows=[]
for t in tags:
    m=MB.get(t,{}).get("methods",{})
    def g(k): return m.get(k,{}).get("gated",np.nan)
    mjc=MJ.get(t,{}).get("combination_gated",np.nan)
    cb=CB.get(t,{}).get("variants",{})
    ours_combo=max([cb.get("ours_persona",{}).get("gated",np.nan),cb.get("ours_incept",{}).get("gated",np.nan)]) if cb else np.nan
    amt=max(SQ[t][k]["gated"] for k in SQ[t]) if t in SQ else np.nan
    dis=max(v["gated"] for v in DS[t].values()) if t in DS else np.nan
    oracle=np.nanmax([amt,dis,ours_combo])
    rows.append((t,g("plain"),g("translated"),mjc,g("aim"),g("deepinception"),ours_combo,oracle))
    def s(x): return f"{x:.2f}" if not np.isnan(x) else " - "
    print(f"{t:16s}{s(g('plain')):7s}{s(g('translated')):7s}{s(mjc):8s}{s(g('aim')):6s}{s(g('deepinception')):8s}{s(ours_combo):11s}{s(oracle)}")
print("\n=== 강모델 평균 (gated) ===")
def avg(idx,grp): 
    v=[r[idx] for r in rows if r[0] in grp]; v=[x for x in v if not np.isnan(x)]; return np.mean(v) if v else np.nan
labels=["plain","translated","MJ-comb","aim","deepinception","ours_combo","ours_oracle"]
for gi,gn in [(strong,"강모델"),(tags,"전체")]:
    print(f"  [{gn}] "+" | ".join(f"{labels[i-1]} {avg(i,gi):.2f}" for i in range(1,8)))
