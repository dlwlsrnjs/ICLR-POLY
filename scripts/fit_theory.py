"""Parametric capability-difficulty theory (observe -> formula -> generate setting).
recon(D;C) = sigmoid(k(C-D))      capability C from a BENIGN probe (no harmful queries)
comply(D)  = a_m + b_m * D         compliance response, model-specific (few harmful points)
gated(D)   = recon(D;C) * comply(D)
D*(C)      = argmax_D gated(D)      GENERATED in closed form; oracle grid only validates.
"""
import json, numpy as np
from scipy.optimize import least_squares, minimize_scalar
from collections import defaultdict
def load(f): return [json.loads(l) for l in open(f)]
def Dval(n,sh,delta): return n+delta*sh
def sig(z): return 1/(1+np.exp(-z))
def recon(D,C,k): return sig(k*(C-D))

# ---- 1) capability C_m from BENIGN probe (joint recon fit, shared k,delta) ----
def probe(path):
    d=json.load(open(path)); return [(n,sh,d[f"interleave_{o}_n{n}"]) for o,sh in [("ordered",0),("shuffled",1)] for n in [2,4,6,8,10] if f"interleave_{o}_n{n}" in d]
qb=probe("results/paper_benign_probe_qwen.json"); gb=probe("results/paper_benign_probe_gpt4o.json")
def resid(p):
    k,delta,Cq,Cg=p; e=[]
    for n,sh,r in qb: e.append(recon(Dval(n,sh,delta),Cq,k)-r)
    for n,sh,r in gb: e.append(recon(Dval(n,sh,delta),Cg,k)-r)
    return e
k,delta,Cq,Cg=least_squares(resid,[0.5,6,12,40],bounds=([0.05,0,4,20],[3,30,60,120])).x

# ---- 2) comply_m(D) per model from a few harmful points ----
def comply_qwen():
    cm=json.load(open("results/qwen_interleaving_curve_matrix.json"))["matrix"]; P=[]
    for o,sh in [("ordered",0),("shuffled",1)]:
        for n in [2,4,6,8,10]:
            e=cm[o].get(str(n))
            if e and e["semantic_recon_rate"]>0: P.append((Dval(n,sh,delta),e["gated_asr"]/e["semantic_recon_rate"]))
    return P
def comply_gpt():
    R={(r["item_id"],r["condition"]):bool(r["semantic_reconstruction_equivalent"]) for r in load("private_artifacts/pilot_gpt4o_shuffle/recon/restricted_reconstruction_audit.jsonl")}
    G={(r["item_id"],r["condition"]):bool(r["guard_strict_unsafe"]) for r in load("private_artifacts/pilot_gpt4o_shuffle/guard/restricted_qwen3guard_audit.jsonl")}
    a=defaultdict(lambda:[0,0,0])
    for kk in G:
        c=kk[1]
        if not c.startswith("interleave"): continue
        a[c][0]+=1; r=R.get(kk,False)
        if r: a[c][1]+=1
        if r and G[kk]: a[c][2]+=1
    P=[]
    for c,(n_,rec,g) in a.items():
        sh=1 if "shuffled" in c else 0; nn=int(c.split("_n")[1])
        if rec: P.append((Dval(nn,sh,delta),g/rec))
    return P
def fit_lin(P):
    D=np.array([p[0] for p in P]); C=np.array([p[1] for p in P]); b,a=np.polyfit(D,C,1); return a,b
aq,bq=fit_lin(comply_qwen()); ag,bg=fit_lin(comply_gpt())

# ---- 3) closed-form D*(C_m, comply_m) over feasible range (grid spans D~2..21) ----
def Dstar(C,a,b,hi=21):
    f=lambda D: -recon(D,C,k)*np.clip(a+b*D,0,1)
    return minimize_scalar(f,bounds=(2,hi),method="bounded").x
def nearest(Dv):
    best=None
    for o,sh in [("ordered",0),("shuffled",1)]:
        for n in [2,4,6,8,10]:
            d=abs(Dval(n,sh,delta)-Dv)
            if best is None or d<best[0]: best=(d,o,n)
    return f"{best[1]} n={best[2]}"
Dq=Dstar(Cq,aq,bq); Dg=Dstar(Cg,ag,bg)

# ---- 4) oracle ----
cm=json.load(open("results/qwen_interleaving_curve_matrix.json"))["matrix"]
oq=max([(cm[o].get(str(n),{}).get("gated_asr",0),o,n) for o in ["ordered","shuffled"] for n in [2,4,6,8,10]])
out={"k":float(k),"delta":float(delta),
 "Qwen":{"C":float(Cq),"comply_slope":float(bq),"Dstar":float(Dq),"predicted":nearest(Dq),"oracle":f"{oq[1]} n={oq[2]}","oracle_gated":float(oq[0])},
 "GPT-4o":{"C":float(Cg),"comply_slope":float(bg),"Dstar":float(Dg),"predicted":nearest(Dg),"oracle":"shuffled n=10","oracle_gated":0.66}}
print(json.dumps(out,indent=2))
json.dump(out,open("results/paper_theory_fit.json","w"),indent=2)
print("\nSUMMARY: formula predicts Qwen ->",out["Qwen"]["predicted"],"(oracle",out["Qwen"]["oracle"]+"); GPT-4o ->",out["GPT-4o"]["predicted"],"(oracle shuffled n=10)")
