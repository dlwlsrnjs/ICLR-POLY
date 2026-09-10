"""Fit the capability-difficulty theory across the model panel and validate it.
Panel membership: by default the weakly-aligned control Mistral-7B is EXCLUDED, because both
the attack and the theory presuppose a refusal barrier (paper Sec. Scope). Set EXCLUDE="" to
fit the full six-model panel, and OUT=<path> to choose the output file.
Pipeline per NEW target: benign probe -> capability C; predict compliance response
from a panel-learned b(C); formula gives D*; theory-warm BO refines with few harmful
queries. Leave-one-out validation + query-efficiency vs baselines."""
import json, glob, os, numpy as np
from scipy.optimize import least_squares, minimize_scalar
import sys; sys.path.insert(0,"scripts")
from optimize_difficulty import bo_run, theory_prior

CONDS=[("ordered",0),("shuffled",1)]; NS=[2,4,6,8,10]
def Dval(n,sh,delta): return n+delta*sh
def sig(z): return 1/(1+np.exp(-z))
def recon(D,C,k): return sig(k*(C-D))

def load_model(tag, benign_path, harmful_path):
    b=json.load(open(benign_path)); h=json.load(open(harmful_path))
    br={}; hr={}; hg={}
    for o,sh in CONDS:
        for n in NS:
            key=f"interleave_{o}_n{n}"
            if key in b: br[(n,sh)]=b[key]
            if key in h: hr[(n,sh)]=h[key]["recon"]; hg[(n,sh)]=h[key]["gated"]
    return {"tag":tag,"benign":br,"hrecon":hr,"gated":hg}

# discover panel models
MODELS=[]
# Qwen2.5 full grid
cm=json.load(open("results/qwen_interleaving_curve_matrix.json"))["matrix"]
q25={"tag":"Qwen2.5-7B","benign":json.load(open("results/paper_benign_probe_qwen.json")),
     "hrecon":{},"gated":{}}
q25["benign"]={(int(k.split("_n")[1]),1 if "shuffled" in k else 0):v for k,v in q25["benign"].items()}
for o,sh in CONDS:
    for n in NS:
        e=cm[o][str(n)]; q25["hrecon"][(n,sh)]=e["semantic_recon_rate"]; q25["gated"][(n,sh)]=e["gated_asr"]
MODELS.append(q25)
EXC=set(x for x in os.environ.get("EXCLUDE","Mistral-7B").split(",") if x)
OUT=os.environ.get("OUT","results/paper_panel_theory.json")
for tag,d in [("Qwen3-8B","panel_qwen3"),("Phi-3.5-mini","panel_phi"),("GPT-4o-mini","panel_gpt4omini"),("Mistral-7B","panel_mistral"),("Qwen2.5-32B","panel_qwen32")]:
    if tag in EXC: continue
    bp=f"private_artifacts/paper_main/{d}/benign_recon.json"; hp=f"private_artifacts/paper_main/{d}/harmful_summary.json"
    if os.path.exists(bp) and os.path.exists(hp): MODELS.append(load_model(tag,bp,hp))
# GPT-4o from pilot benign + pilot harmful (partial: n4,n10)
print("panel models available:",[m["tag"] for m in MODELS])
if len(MODELS)<3: print("PANEL INCOMPLETE - waiting for more models"); sys.exit(0)

# --- fit shared k, delta + per-model C to BENIGN recon ---
allpts=[]
for mi,m in enumerate(MODELS):
    for (n,sh),r in m["benign"].items(): allpts.append((mi,n,sh,r))
nM=len(MODELS)
def resid(p):
    k=p[0]; delta=p[1]; Cs=p[2:]; e=[]
    for mi,n,sh,r in allpts: e.append(recon(Dval(n,sh,delta),Cs[mi],k)-r)
    return e
p0=[0.4,8]+[10]*nM
sol=least_squares(resid,p0,bounds=([0.05,0]+[2]*nM,[3,30]+[100]*nM))
k=sol.x[0]; delta=sol.x[1]; Cs={MODELS[i]["tag"]:sol.x[2+i] for i in range(nM)}
print(f"\nshared k={k:.3f} delta={delta:.2f}")
print("capability C:", {t:round(c,1) for t,c in Cs.items()})

# --- per-model comply linear fit; then b(C) relationship ---
comply_fit={}
for m in MODELS:
    D=[]; Cy=[]
    for (n,sh),g in m["gated"].items():
        r=m["hrecon"][(n,sh)]
        if r>0: D.append(Dval(n,sh,delta)); Cy.append(g/r)
    b,a=np.polyfit(D,Cy,1); comply_fit[m["tag"]]=(a,b)
print("comply (a,b):",{t:(round(a,3),round(b,4)) for t,(a,b) in comply_fit.items()})
# b(C): does compliance slope rise with capability?
Carr=np.array([Cs[m["tag"]] for m in MODELS]); barr=np.array([comply_fit[m["tag"]][1] for m in MODELS])
aarr=np.array([comply_fit[m["tag"]][0] for m in MODELS])
bc_slope,bc_int=np.polyfit(Carr,barr,1)
ac_slope,ac_int=np.polyfit(Carr,aarr,1)
corr=np.corrcoef(Carr,barr)[0,1]
print(f"b(C) = {bc_slope:.5f}*C + {bc_int:.4f}   corr(C, comply-slope)={corr:.3f}")

# --- Leave-one-out: predict held-out D* from C + panel b(C); BO efficiency ---
def Dstar(C,a,b,hi=21):
    f=lambda D:-recon(D,C,k)*np.clip(a+b*D,0,1); return minimize_scalar(f,bounds=(2,hi),method="bounded").x
def nearest(Dv):
    best=None
    for o,sh in CONDS:
        for n in NS:
            d=abs(Dval(n,sh,delta)-Dv)
            if best is None or d<best[0]: best=(d,f"{o} n={n}",Dval(n,sh,delta))
    return best[1]
print("\n=== Leave-one-out validation ===")
rows=[]
for mi,m in enumerate(MODELS):
    C=Cs[m["tag"]]
    # refit b(C) and a(C) WITHOUT this model
    idx=[j for j in range(nM) if j!=mi]
    bcs,bci=np.polyfit(Carr[idx],barr[idx],1); acs,aci=np.polyfit(Carr[idx],aarr[idx],1)
    a_pred=aci+acs*C; b_pred=bci+bcs*C
    Ds=Dstar(C,a_pred,b_pred); pred=nearest(Ds)
    pn=int(pred.split("n=")[1]); psh=1 if pred.startswith("shuffled") else 0
    pred_gated=m["gated"][(pn,psh)]
    # oracle
    orc=max(m["gated"].items(),key=lambda kv:kv[1]); orc_cond=f"{'shuffled' if orc[0][1] else 'ordered'} n={orc[0][0]}"
    # BO with theory prior (predicted)
    Dgrid=[Dval(n,sh,delta) for o,sh in CONDS for n in NS]
    gated=[m["gated"][(n,sh)] for o,sh in CONDS for n in NS]
    prior=theory_prior(np.array(Dgrid),C,k,a_pred,b_pred)
    import numpy as _np
    qs=[]
    for s in range(200):
        import optimize_difficulty as OD; OD.rng=_np.random.RandomState(s)
        qs.append(bo_run(Dgrid,gated,prior=prior))
    qv=[]
    for s in range(200):
        import optimize_difficulty as OD; OD.rng=_np.random.RandomState(s)
        qv.append(bo_run(Dgrid,gated,prior=None))
    rows.append((m["tag"],round(C,1),pred,orc_cond,round(orc[1],3),round(float(_np.mean(qs)),2),round(float(_np.mean(qv)),2),round(float(pred_gated),3),round(float(orc[1]-pred_gated),3)))
    print(f"  {m['tag']:>14}: C={C:>5.1f} pred={pred:>13} (gated {pred_gated:.3f}) | oracle={orc_cond:>13} ({orc[1]:.3f}) regret {orc[1]-pred_gated:.3f} | BO-warm {_np.mean(qs):.2f}q vs vanilla {_np.mean(qv):.2f}q")
json.dump({"k":float(k),"delta":float(delta),"C":{t:float(c) for t,c in Cs.items()},
           "comply":{t:[float(a),float(b)] for t,(a,b) in comply_fit.items()},
           "b_of_C_corr":float(corr),"excluded":sorted(EXC),"n_models":nM,
           "loo_columns":["tag","C","predicted","oracle","oracle_gated","warm_bo_q","vanilla_bo_q","predicted_gated","regret"],
           "loo":rows},open(OUT,"w"),indent=2,default=float)
print("\nsaved",OUT)
