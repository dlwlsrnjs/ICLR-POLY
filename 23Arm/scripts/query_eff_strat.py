#!/usr/bin/env python3
"""Where does the free probe's query advantage concentrate? Stratify panel targets by an OBJECTIVE
difficulty = how NARROW the winning region is (number of arms within 90% of the per-target oracle;
few = hard). Compare ours(kernel+probe) vs flat(kernel only, uninformed) by queries-to-95%-oracle and
solve-rate within each stratum. Replay; run as jinkwon."""
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
def qto(order,jv,thr):
    r=-1
    for i,a in enumerate(order,1):
        r=max(r,jv[a])
        if r>=thr-1e-9: return i
    return None
# narrowness per target: #arms >= 90% oracle (few = narrow/hard)
narrow=[]
for t in range(len(tags)):
    orc=Jeval[t].max(); narrow.append(int((Jeval[t]>=0.9*orc-1e-9).sum()))
narrow=np.array(narrow)
# stratify: narrow (hard) = n_near <= 3 ; wide (easy) = n_near >= 8 ; mid otherwise
strata={"narrow (<=3 near-oracle arms)":narrow<=3,"mid (4-7)":(narrow>=4)&(narrow<=7),"wide (>=8)":narrow>=8}
def eval_stratum(mask):
    out={k:{"q":[],"solv":0,"tot":0} for k in("ours","flat")}
    for t in np.where(mask)[0]:
        orc=float(Jeval[t].max()); thr=LVL*orc
        def pf(a,t=t): items=rng.choice(tr,size=min(25,len(tr)),replace=False); return float(np.nanmean(J[t,items,a]))
        prm=structured_prior_mean(theta,arm_names,Afp[t],Clo[t],Chi[t],Mfp[t]); flat=np.full(A,gmean)
        for _ in range(REPS):
            for k,o in (("ours",gp_bai(prm,feats,pf,BUD)[1]),("flat",gp_bai(flat,feats,pf,BUD,explore=1.5)[1])):
                out[k]["tot"]+=1; q=qto(o,Jeval[t],thr)
                if q is not None: out[k]["solv"]+=1; out[k]["q"].append(q)
    return out
print("narrowness distribution (arms within 90%% oracle):",dict(zip(*np.unique(narrow,return_counts=True))))
print("\n%-30s %5s | %-22s %-22s"%("stratum","nT","ours (q / solved%)","flat/uninf (q / solved%)"))
for name,mask in strata.items():
    nt=int(mask.sum())
    if nt==0: print("%-30s %5d | (empty)"%(name,nt)); continue
    o=eval_stratum(mask)
    def fmt(d): return "%.2f / %.0f%%"%(np.mean(d["q"]) if d["q"] else float('nan'),100*d["solv"]/max(d["tot"],1))
    speed=(np.mean(o["flat"]["q"]) if o["flat"]["q"] else float('nan'))/(np.mean(o["ours"]["q"]) if o["ours"]["q"] else float('nan'))
    print("%-30s %5d | %-22s %-22s  (flat/ours q ratio %.2fx)"%(name,nt,fmt(o["ours"]),fmt(o["flat"]),speed))
