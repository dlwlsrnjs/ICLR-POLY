#!/usr/bin/env python3
"""Difficulty-stratified query efficiency with FOUR strategies (adds random + structure-free
independent-arm UCB to ours vs flat/uninformed). Stratum = #arms within 90% of the per-target oracle
(narrow=hard). Metric: mean queries to 95% of the per-target oracle + within-budget solve rate.
Replay; run as jinkwon."""
import sys, json, numpy as np
sys.path.insert(0,"scripts")
from gp_bai import arm_feats, structured_prior_mean, gp_bai
D="results/context_selector_full_train_20260904"
blob=np.load(f"{D}/context_values.npz"); man=json.load(open(f"{D}/manifest.json"))
J=blob["J"]; ctx=blob["context"]; splits=blob["splits"]; arm_names=man["arms"]; tags=man["targets"]
feats=arm_feats(arm_names); A=len(arm_names)
theta=json.load(open("results/structured_policy_ext_20260904/summary.json"))["theta_global"]
Afp=ctx[:,0]; Mfp=ctx[:,1]; Clo=0.2+ctx[:,3]; Chi=0.2+ctx[:,4]
tr=np.where(splits=="train")[0]; ev=np.where(splits=="test")[0]; Jeval=J[:,ev,:].mean(1)
gmean=float(np.mean([Jeval[t].mean() for t in range(len(tags))]))
REPS=200; BUD=12; LVL=0.95; rng=np.random.default_rng(0)
STR=("ours","flat","random","indep")
def qto(order,jv,thr):
    r=-1
    for i,a in enumerate(order,1):
        r=max(r,jv[a])
        if r>=thr-1e-9: return i
    return None
def indep_ucb_order(pf,budget):
    n=np.zeros(A); s=np.zeros(A); order=[]; r2=np.random.default_rng(123)
    for b in range(budget):
        un=[i for i in range(A) if n[i]==0]
        a=int(r2.choice(un)) if un else int(np.argmax(s/np.maximum(n,1)+1.5*np.sqrt(np.log(b+1)/np.maximum(n,1))))
        y=pf(a); n[a]+=1; s[a]+=y; order.append(a)
    return order
narrow=np.array([int((Jeval[t]>=0.9*Jeval[t].max()-1e-9).sum()) for t in range(len(tags))])
strata={"narrow":narrow<=3,"mid":(narrow>=4)&(narrow<=7),"wide":narrow>=8}
def eval_stratum(mask):
    out={k:{"q":[],"solv":0,"tot":0} for k in STR}
    for t in np.where(mask)[0]:
        orc=float(Jeval[t].max()); thr=LVL*orc
        def pf(a,t=t): items=rng.choice(tr,size=min(25,len(tr)),replace=False); return float(np.nanmean(J[t,items,a]))
        prm=structured_prior_mean(theta,arm_names,Afp[t],Clo[t],Chi[t],Mfp[t]); flat=np.full(A,gmean)
        for _ in range(REPS):
            orders={"ours":gp_bai(prm,feats,pf,BUD)[1],"flat":gp_bai(flat,feats,pf,BUD,explore=1.5)[1],
                    "random":list(np.random.default_rng(rng.integers(1<<30)).permutation(A)[:BUD]),
                    "indep":indep_ucb_order(pf,BUD)}
            for k,o in orders.items():
                out[k]["tot"]+=1; q=qto(o,Jeval[t],thr)
                if q is not None: out[k]["solv"]+=1; out[k]["q"].append(q)
    return out
print("%-8s %4s | %s"%("stratum","nT"," ".join("%-14s"%s for s in STR)))
agg={k:{"q":[],"solv":0,"tot":0} for k in STR}
for name,mask in strata.items():
    nt=int(mask.sum())
    if nt==0: print("%-8s %4d | empty"%(name,nt)); continue
    o=eval_stratum(mask)
    cells=[]
    for k in STR:
        d=o[k]; agg[k]["q"]+=d["q"]; agg[k]["solv"]+=d["solv"]; agg[k]["tot"]+=d["tot"]
        cells.append("%.2f/%2.0f%%"%(np.mean(d["q"]) if d["q"] else float('nan'),100*d["solv"]/max(d["tot"],1)))
    print("%-8s %4d | %s"%(name,nt," ".join("%-14s"%c for c in cells)))
print("%-8s %4s | %s"%("ALL","", " ".join("%-14s"%("%.2f/%2.0f%%"%(np.mean(agg[k]["q"]),100*agg[k]["solv"]/agg[k]["tot"])) for k in STR)))
