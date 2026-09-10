#!/usr/bin/env python3
"""Is GP-UCB the best acquisition for our BAI? Compare acquisitions on the panel (same structured prior
mu0 + same kernel; only the next-arm rule changes), by 95%-oracle attainment. Candidates: UCB(beta=1.0
current), UCB(beta=2.5), pure max-variance (uncertainty), Expected Improvement, Top-Two Thompson (TTTS)."""
import sys, json, numpy as np
sys.path.insert(0,"scripts")
from gp_bai import arm_feats, structured_prior_mean
from scipy.stats import norm
D="results/context_selector_full_train_20260904"
blob=np.load(f"{D}/context_values.npz"); man=json.load(open(f"{D}/manifest.json"))
J=blob["J"]; ctx=blob["context"]; splits=blob["splits"]; arm_names=man["arms"]; tags=man["targets"]
feats=arm_feats(arm_names); A=len(arm_names)
theta=json.load(open("results/structured_policy_ext_20260904/summary.json"))["theta_global"]
Afp=ctx[:,0]; Mfp=ctx[:,1]; Clo=0.2+ctx[:,3]; Chi=0.2+ctx[:,4]
tr=np.where(splits=="train")[0]; ev=np.where(splits=="test")[0]; Jeval=J[:,ev,:].mean(1)
ls,sig,noise=0.35,0.18,0.02; REPS=200; BUD=12; LVL=0.95
K=sig**2*np.exp(-0.5*((feats[:,None,:]-feats[None,:,:])**2).sum(-1)/ls**2)
rng=np.random.default_rng(0)
def post(prm,obs_idx,obs_y):
    if not obs_idx: return prm.copy(), np.sqrt(np.diag(K))
    idx=np.array(obs_idx); y=np.array(obs_y); Koo=K[np.ix_(idx,idx)]+noise*np.eye(len(idx))
    Ki=np.linalg.inv(Koo); Kxi=K[:,idx]; mean=prm+Kxi@(Ki@(y-prm[idx]))
    var=np.clip(np.diag(K)-np.einsum("ai,ij,aj->a",Kxi,Ki,Kxi),1e-9,None); return mean,np.sqrt(var)
def run(acq,prm,pf,r):
    obs_idx,obs_y=[],[]; order=[]
    for b in range(BUD):
        m,sd=post(prm,obs_idx,obs_y)
        best=max(obs_y) if obs_y else 0.0
        if acq=="ucb1": sc=m+1.0*sd
        elif acq=="ucb25": sc=m+2.5*sd
        elif acq=="var": sc=sd.copy()
        elif acq=="ei":
            z=(m-best)/(sd+1e-9); sc=(m-best)*norm.cdf(z)+sd*norm.pdf(z)
        elif acq=="ttts":
            # top-two thompson: sample, take top; with prob 0.5 take the 2nd distinct top of a resample
            s1=m+sd*r.standard_normal(A); s1[obs_idx]=-1e9; a1=int(np.argmax(s1))
            if r.random()<0.5:
                for _ in range(8):
                    s2=m+sd*r.standard_normal(A); s2[obs_idx]=-1e9; a2=int(np.argmax(s2))
                    if a2!=a1: order.append(a2); obs_idx.append(a2); obs_y.append(pf(a2)); break
                else: order.append(a1); obs_idx.append(a1); obs_y.append(pf(a1))
            else:
                order.append(a1); obs_idx.append(a1); obs_y.append(pf(a1))
            continue
        sc=np.asarray(sc,float); sc[obs_idx]=-1e9; a=int(np.argmax(sc))
        order.append(a); obs_idx.append(a); obs_y.append(pf(a))
    return order
def qto(order,jv,thr):
    run=-1
    for i,a in enumerate(order,1):
        run=max(run,jv[a])
        if run>=thr-1e-9: return i
    return None
acqs=["ucb1","ucb25","var","ei","ttts"]; allq={k:[] for k in acqs}; solved={k:0 for k in acqs}; tot=0
for t in range(len(tags)):
    orc=float(Jeval[t].max()); thr=LVL*orc
    def pf(a,t=t): items=rng.choice(tr,size=min(25,len(tr)),replace=False); return float(np.nanmean(J[t,items,a]))
    prm=structured_prior_mean(theta,arm_names,Afp[t],Clo[t],Chi[t],Mfp[t])
    for _ in range(REPS):
        tot+=1; r=np.random.default_rng(rng.integers(1<<30))
        for k in acqs:
            q=qto(run(k,prm,pf,r),Jeval[t],thr)
            if q is not None: solved[k]+=1; allq[k].append(q)
print(f"panel {len(tags)}x{REPS} | reach>= {int(LVL*100)}% oracle | acquisition (same prior+kernel)")
print("%-7s %10s %12s"%("acq","mean-q","solved<=12 %"))
for k in acqs:
    print("%-7s %10.2f %11.1f%%"%(k,np.mean(allq[k]) if allq[k] else float('nan'),100*solved[k]/tot))
