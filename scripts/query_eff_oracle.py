#!/usr/bin/env python3
"""Re-check (reviewer + user methodological point): measure query efficiency by ORACLE ATTAINMENT
(queries to reach >=X% of the per-target oracle), not by the self-referential "reach what ours gets at
3 queries". Separate the two priors: the free probe mu0 (ours vs FLAT, both share the feature kernel)
and the structure (FLAT-GP vs RANDOM/INDEP-UCB, which have no kernel). Panel replay (temp-0 => replay
== live). Read-only; run as jinkwon."""
import sys, json, numpy as np
sys.path.insert(0,"scripts")
from gp_bai import arm_feats, structured_prior_mean, gp_bai
D="results/context_selector_full_train_20260904"
blob=np.load(f"{D}/context_values.npz"); man=json.load(open(f"{D}/manifest.json"))
J=blob["J"]; ctx=blob["context"]; splits=blob["splits"]; arm_names=man["arms"]; tags=man["targets"]
feats=arm_feats(arm_names); A=len(arm_names)
theta=json.load(open("results/structured_policy_ext_20260904/summary.json"))["theta_global"]
Afp=ctx[:,0]; Mfp=ctx[:,1]; Clo=0.2+ctx[:,3]; Chi=0.2+ctx[:,4]
tr=np.where(splits=="train")[0]; ev=np.where(splits=="test")[0]
Jeval=J[:,ev,:].mean(1)   # [T,A] true gated per arm
gmean=float(np.mean([Jeval[t].mean() for t in range(len(tags))]))
REPS=200; BUD=12
rng=np.random.default_rng(0)
def probe(t):
    def f(a):
        items=rng.choice(tr,size=min(25,len(tr)),replace=False)
        return float(np.nanmean(J[t,items,a]))
    return f
def q_to(order, jvals, thr):
    run=-1
    for i,a in enumerate(order,1):
        run=max(run, jvals[a])
        if run>=thr-1e-9: return i
    return None
def indep_ucb_order(pf, budget):
    # no kernel, no prior: forced-explore unseen then UCB per arm
    n=np.zeros(A); s=np.zeros(A); order=[]; r2=np.random.default_rng(123)
    for b in range(budget):
        un=[i for i in range(A) if n[i]==0]
        a=int(r2.choice(un)) if un else int(np.argmax(s/np.maximum(n,1)+1.5*np.sqrt(np.log(b+1)/np.maximum(n,1))))
        y=pf(a); n[a]+=1; s[a]+=y; order.append(a)
    return order
res={lvl:{k:[] for k in ("ours","flat","random","indep")} for lvl in (0.90,0.95)}
solved={lvl:{k:0 for k in ("ours","flat","random","indep")} for lvl in (0.90,0.95)}
for t in range(len(tags)):
    orc=float(Jeval[t].max()); pf=probe(t)
    prm=structured_prior_mean(theta,arm_names,Afp[t],Clo[t],Chi[t],Mfp[t])
    flat=np.full(A,gmean)
    for _ in range(REPS):
        _,o_ours=gp_bai(prm,feats,pf,BUD)
        _,o_flat=gp_bai(flat,feats,pf,BUD,explore=1.5)
        o_rand=list(np.random.default_rng(rng.integers(1<<30)).permutation(A)[:BUD])
        o_ind=indep_ucb_order(pf,BUD)
        for lvl in (0.90,0.95):
            thr=lvl*orc
            for k,o in (("ours",o_ours),("flat",o_flat),("random",o_rand),("indep",o_ind)):
                q=q_to(o,Jeval[t],thr)
                if q is not None: res[lvl][k].append(q); solved[lvl][k]+=1
n=len(tags)*REPS
print("panel targets:",len(tags),"  REPS:",REPS,"  budget:",BUD,"  global arm-mean prior:",round(gmean,3))
for lvl in (0.90,0.95):
    print(f"\n=== reach >= {int(lvl*100)}% of per-target ORACLE ===")
    for k in ("ours","flat","random","indep"):
        qs=res[lvl][k]; mq=np.mean(qs) if qs else float('nan'); solv=solved[lvl][k]/n
        print(f"  {k:8s}  mean-queries {mq:5.2f}  solved-within-12 {solv:5.1%}")
