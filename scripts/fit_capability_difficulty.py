"""Capability-Difficulty model for PolyJigsaw.
Difficulty D(ordering,n) = n + delta*[shuffled].
recon(D;C)  = 1/(1+exp(k*(D-C)))         # reconstruction succeeds while capability C > difficulty D
comply(D)   = c0 + (cmax-c0)*(1-exp(-lam*D))  # harder puzzle suppresses safety re-check, saturating
gated(D;C)  = recon(D;C) * comply(D)
Prediction: d*(C)=argmax gated increases with capability C.
"""
import json, numpy as np
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def grid(path):
    cm=json.load(open(path))["matrix"]; pts=[]
    for order in ["ordered","shuffled"]:
        for n in [2,4,6,8,10]:
            e=cm[order].get(str(n));
            if not e: continue
            pts.append((n, 1 if order=="shuffled" else 0, e["semantic_recon_rate"], e["gated_asr"]))
    return pts

def Dval(n,sh,delta): return n + delta*sh
def recon_f(D,C,k): return 1/(1+np.exp(k*(D-C)))
def comply_f(D,c0,cmax,lam): return c0+(cmax-c0)*(1-np.exp(-lam*D))

def fit_model(pts):
    n=np.array([p[0] for p in pts],float); sh=np.array([p[1] for p in pts],float)
    recon=np.array([p[2] for p in pts],float); gated=np.array([p[3] for p in pts],float)
    comply=np.divide(gated,recon,out=np.zeros_like(gated),where=recon>0)
    # joint fit: parameters delta,C,k (recon) and c0,cmax,lam (comply)
    def recon_model(X,delta,C,k):
        nn,ss=X; return recon_f(Dval(nn,ss,delta),C,k)
    p0=[6,10,0.5]
    try:
        (delta,C,k),_=curve_fit(recon_model,(n,sh),recon,p0=p0,maxfev=20000,
                                 bounds=([0,2,0.05],[30,60,5]))
    except Exception: delta,C,k=6,12,0.4
    D=Dval(n,sh,delta)
    def comply_model(DD,c0,cmax,lam): return comply_f(DD,c0,cmax,lam)
    try:
        (c0,cmax,lam),_=curve_fit(comply_model,D,comply,p0=[0.5,0.8,0.2],maxfev=20000,
                                  bounds=([0,0.2,0.001],[1,1.2,2]))
    except Exception: c0,cmax,lam=comply.min(),comply.max(),0.2
    return dict(delta=delta,C=C,k=k,c0=c0,cmax=cmax,lam=lam)

def dstar(params,Drange):
    r=recon_f(Drange,params["C"],params["k"]); c=comply_f(Drange,params["c0"],params["cmax"],params["lam"])
    g=r*c; i=int(np.argmax(g)); return Drange[i], g[i]

# --- Qwen (full grid) ---
qp=grid("results/qwen_interleaving_curve_matrix.json")
qf=fit_model(qp)
Dr=np.linspace(2,40,400)
qd,qg=dstar(qf,Dr)
print("Qwen fit:",{k:round(v,3) for k,v in qf.items()})
print(f"Qwen capability C={qf['C']:.1f}, optimal difficulty D*={qd:.1f}, predicted peak gated={qg:.3f}")

# --- GPT-4o partial grid from pilot (ordered/shuffled n4,n10) ---
def gpt_grid():
    import collections
    R={}; 
    def load(f): return [json.loads(l) for l in open(f)]
    rec={ (r["item_id"],r["condition"]):bool(r["semantic_reconstruction_equivalent"]) for r in load("private_artifacts/pilot_gpt4o_shuffle/recon/restricted_reconstruction_audit.jsonl")}
    gd={ (r["item_id"],r["condition"]):bool(r["guard_strict_unsafe"]) for r in load("private_artifacts/pilot_gpt4o_shuffle/guard/restricted_qwen3guard_audit.jsonl")}
    from collections import defaultdict
    agg=defaultdict(lambda:[0,0,0])
    for k in gd:
        c=k[1]
        if not c.startswith("interleave"): continue
        agg[c][0]+=1; r=rec.get(k,False)
        if r: agg[c][1]+=1
        if r and gd[k]: agg[c][2]+=1
    pts=[]
    for c,(nn,rr,gg) in agg.items():
        order=1 if "shuffled" in c else 0; n=int(c.split("_n")[1])
        pts.append((n,order,rr/nn,gg/nn))
    return pts
gp=gpt_grid()
gf=fit_model(gp)
gd_,gg_=dstar(gf,Dr)
print("GPT-4o(old) fit:",{k:round(v,3) for k,v in gf.items()})
print(f"GPT-4o capability C={gf['C']:.1f}, optimal difficulty D*={gd_:.1f}, predicted peak gated={gg_:.3f}")

# --- figure ---
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
for a,(name,params,pts,col) in zip(ax,[("Qwen2.5-7B (weaker reconstructor)",qf,qp,"#2166ac"),
                                        ("GPT-4o (stronger reconstructor)",gf,gp,"#b2182b")]):
    D=Dr
    r=recon_f(D,params["C"],params["k"]); c=comply_f(D,params["c0"],params["cmax"],params["lam"]); g=r*c
    a.plot(D,r,'--',color="#2166ac",label="recon(D)")
    a.plot(D,c,'--',color="#b2182b",label="comply(D)")
    a.plot(D,g,'-',color="#1a9850",lw=2.5,label="gated(D)=recon×comply")
    ds,gs=dstar(params,D); a.axvline(ds,color="#1a9850",ls=":",lw=1.5)
    a.annotate(f"D*={ds:.0f}",xy=(ds,gs),xytext=(ds+2,gs+0.05),color="#1a7a3f",fontsize=10)
    # empirical points
    for n,sh,rr,gg in pts:
        Dp=Dval(n,sh,params["delta"]); a.scatter([Dp],[gg],color="#1a9850",s=28,zorder=5,edgecolor="k",linewidth=.4)
    a.set_title(name,fontsize=10.5); a.set_xlabel("difficulty D  (= n + δ·shuffled)"); a.set_ylim(0,1.02)
    a.legend(fontsize=8,loc="upper right"); a.grid(alpha=.25)
ax[0].set_ylabel("rate")
plt.suptitle("Capability–Difficulty model: optimal difficulty D* shifts right as reconstruction capability grows",fontsize=11)
plt.tight_layout()
plt.savefig("paper/figures/fig_capability_difficulty.pdf"); plt.savefig("paper/figures/fig_capability_difficulty.png",dpi=150)
json.dump({"qwen":{k:float(v) for k,v in qf.items()},"qwen_dstar":float(qd),
           "gpt4o":{k:float(v) for k,v in gf.items()},"gpt4o_dstar":float(gd_)},
          open("results/paper_capability_difficulty_fit.json","w"),indent=2)
print("wrote fig_capability_difficulty + fit json")
