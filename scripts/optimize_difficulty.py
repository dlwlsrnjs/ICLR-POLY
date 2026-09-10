"""Per-target difficulty selection as query-efficient black-box optimization.
Compares: theory-warm-started GP-UCB (Bayesian optimization) vs vanilla GP-UCB vs
random search vs full grid, on the number of (harmful) queries to reach the oracle.
The capability-difficulty formula supplies the GP prior mean (from a benign probe /
public capability), so BO starts near D* and refines with a few queries."""
import json, numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C_
rng=np.random.RandomState(0)

def grid_from_summary(summ, delta):
    # summ: {condition: {recon,gated}} -> list of (D, gated)
    pts=[]
    for cond,v in summ.items():
        sh=1 if "shuffled" in cond else 0; n=int(cond.split("_n")[1])
        pts.append((n+delta*sh, v["gated"], cond))
    pts.sort(); return pts

def theory_prior(D, C, k, a, b):
    r=1/(1+np.exp(-k*(C-D))); c=np.clip(a+b*D,0,1); return r*c

def bo_run(configs, gated, prior=None, iters=6, beta=2.0, noise=0.03):
    """configs: 1D array of D values; gated: true gated per config. Returns queries-to-oracle."""
    X=np.array(configs).reshape(-1,1); ytrue=np.array(gated)
    oracle=ytrue.max(); queried=[]; 
    # start: 1 random (or theory-argmax if prior)
    if prior is not None:
        start=int(np.argmax(prior))
    else:
        start=rng.randint(len(configs))
    order=[start]
    for t in range(1,iters+1):
        Xq=X[order]; yq=ytrue[order]+rng.normal(0,noise,len(order))
        gp=GaussianProcessRegressor(kernel=C_(0.1)*Matern(length_scale=6,nu=2.5),alpha=noise**2,normalize_y=True,n_restarts_optimizer=2)
        if prior is not None:
            gp.fit(Xq,yq-prior[order]); mu,sd=gp.predict(X,return_std=True); mu=mu+prior
        else:
            gp.fit(Xq,yq); mu,sd=gp.predict(X,return_std=True)
        ucb=mu+beta*sd
        for j in np.argsort(-ucb):
            if j not in order: nxt=int(j); break
        order.append(nxt)
        best=ytrue[order].max()
        if best>=oracle-1e-9:
            return t  # queries used to first hit oracle
    return iters+1  # didn't reach in budget

def eval_model(summ, C, k, a, b, delta, trials=200):
    pts=grid_from_summary(summ,delta)
    D=[p[0] for p in pts]; gated=[p[1] for p in pts]
    prior=theory_prior(np.array(D),C,k,a,b)
    res={}
    for name,pr in [("theory-warm BO",prior),("vanilla BO",None)]:
        q=[]; 
        for s in range(trials):
            global rng; rng=np.random.RandomState(s)
            q.append(bo_run(D,gated,prior=pr))
        res[name]=float(np.mean(q))
    # random search: expected queries to hit oracle arm
    n=len(D); best_idx=int(np.argmax(gated)); 
    # number of oracle-optimal arms (ties within eps)
    oracle=max(gated); n_opt=sum(1 for g in gated if g>=oracle-1e-9)
    res["random search"]=float((n+1)/(n_opt+1))  # expected draws to first success
    res["full grid"]=float(n)
    res["oracle_config"]=pts[best_idx][2]; res["oracle_gated"]=float(oracle)
    return res

if __name__=="__main__":
    import sys
    th=json.load(open("results/paper_theory_fit.json"))
    # test on Qwen full grid
    cm=json.load(open("results/qwen_interleaving_curve_matrix.json"))["matrix"]
    summ={f"interleave_{o}_n{n}":{"recon":cm[o][str(n)]["semantic_recon_rate"],"gated":cm[o][str(n)]["gated_asr"]}
          for o in ["ordered","shuffled"] for n in [2,4,6,8,10]}
    delta=th["delta"]; q=th["Qwen"]
    r=eval_model(summ,q["C"],th["k"],0.5,-0.0004,delta)  # qwen comply ~flat
    print("Qwen2.5-7B optimization (avg queries to reach oracle):")
    for k2,v in r.items(): print(f"  {k2}: {v}")
