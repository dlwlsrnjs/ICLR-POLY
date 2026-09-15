#!/usr/bin/env python3
"""95%-oracle attainment, EXCLUDING trivial query-1 solves from the mean (per user), and separately
reporting the query-1 solve fraction (the free probe's 'first pick already near-oracle' credit). Panel
replay. ours(kernel+probe) vs flat(kernel only) vs random(no prior) vs indep-UCB(no kernel)."""
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
Jeval=J[:,ev,:].mean(1); gmean=float(np.mean([Jeval[t].mean() for t in range(len(tags))]))
REPS=200; BUD=12; LVL=0.95; rng=np.random.default_rng(0)
def probe(t):
    def f(a):
        items=rng.choice(tr,size=min(25,len(tr)),replace=False); return float(np.nanmean(J[t,items,a]))
    return f
def q_to(order,jv,thr):
    run=-1
    for i,a in enumerate(order,1):
        run=max(run,jv[a])
        if run>=thr-1e-9: return i
    return None
def indep(pf):
    n=np.zeros(A); s=np.zeros(A); order=[]; r2=np.random.default_rng(123)
    for b in range(BUD):
        un=[i for i in range(A) if n[i]==0]
        a=int(r2.choice(un)) if un else int(np.argmax(s/np.maximum(n,1)+1.5*np.sqrt(np.log(b+1)/np.maximum(n,1))))
        y=pf(a); n[a]+=1; s[a]+=y; order.append(a)
    return order
allq={k:[] for k in("ours","flat","random","indep")}
q1={k:0 for k in allq}; solved={k:0 for k in allq}; tot=0
for t in range(len(tags)):
    orc=float(Jeval[t].max()); thr=LVL*orc; pf=probe(t)
    prm=structured_prior_mean(theta,arm_names,Afp[t],Clo[t],Chi[t],Mfp[t]); flat=np.full(A,gmean)
    for _ in range(REPS):
        tot+=1
        orders={"ours":gp_bai(prm,feats,pf,BUD)[1],"flat":gp_bai(flat,feats,pf,BUD,explore=1.5)[1],
                "random":list(np.random.default_rng(rng.integers(1<<30)).permutation(A)[:BUD]),"indep":indep(pf)}
        for k,o in orders.items():
            q=q_to(o,Jeval[t],thr)
            if q is not None:
                solved[k]+=1
                if q==1: q1[k]+=1
                else: allq[k].append(q)   # exclude q==1
print(f"panel {len(tags)} targets x {REPS} reps | reach >= {int(LVL*100)}% oracle | mean EXCLUDES q==1")
print("%-8s %10s %14s %14s"%("strat","mean(q>1)","q1-solve %","solved<=12 %"))
for k in ("ours","flat","random","indep"):
    mq=np.mean(allq[k]) if allq[k] else float('nan')
    print("%-8s %10.2f %13.1f%% %13.1f%%"%(k,mq,100*q1[k]/tot,100*solved[k]/tot))
