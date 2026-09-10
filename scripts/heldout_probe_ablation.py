#!/usr/bin/env python3
"""Move-2 ablation (reviewer-requested): on the held-out transfer models, isolate how much of the
frozen selector's gain the FREE harmless probe actually earns, vs an uninformed (flat) prior GP that
spends the same harmful budget. Per collection (esp. Lingua, where the headline held-out win lives):
  uninformed@b : gp_bai with flat prior (0.5), budget b   -- pure structured GP search, no probe
  probe@b      : gp_bai with the frozen benign-probe prior, budget b   -- the deployed selector
  fixed        : panel best-fixed arm applied unchanged
  oracle       : best arm per target
Also reports the probe's STARTING-arm quality (prior-argmax verified) vs random-arm, which explains
the low global rank-correlation: the prior needs only to place the first probe well, not to rank all
arms. Paired bootstrap CI over the held-out targets for probe@3 - uninformed@3. Read-only; run as jinkwon."""
import sys, json, numpy as np
sys.path.insert(0,"scripts")
import heldout_selector as HS

rng = np.random.default_rng(20260909)
def boot(d, reps=5000):
    d=np.asarray(d); m=[np.random.default_rng(s).choice(d,len(d)).mean() for s in range(reps)]
    return float(d.mean()), float(np.percentile(m,2.5)), float(np.percentile(m,97.5))

def probe_fn(G_h, n_items):
    return lambda a: float(np.clip(G_h[a]+rng.normal(0,np.sqrt(max(G_h[a]*(1-G_h[a]),.01)/n_items)),0,1))

out={}
for collection, mod, n_items in HS.DATASETS:
    fixed_i, pub_i, names = HS.panel_refs(mod)
    cand_name = HS.panel_candidate(collection)
    FE = mod.FE
    rows={}
    for t in HS.HELDOUT:
        try: arms = mod.arms_for(t)
        except Exception: continue
        G_h = np.array([a[1] for a in arms]); A=len(G_h)
        cand = HS.candidates(collection, t, names)
        if cand_name not in cand: continue
        pr_probe = cand[cand_name]*0.6
        pr_flat  = np.full(A,0.5)*0.6
        u3,p3,ua,pa,pstart=[],[],[],[],[]
        rand_start=[]
        for _ in range(HS.REPS):
            pf=probe_fn(G_h,n_items)
            ru,_=HS.gp_bai(pr_flat, FE, pf, HS.BUDGET);  u3.append(G_h[ru])
            rp,_=HS.gp_bai(pr_probe,FE, pf, HS.BUDGET);  p3.append(G_h[rp])
            rua,_=HS.gp_bai_adaptive(pr_flat, FE, pf, HS.B_FLOOR,HS.B_MAX,HS.STOP_TAU); ua.append(G_h[rua])
            rpa,_=HS.gp_bai_adaptive(pr_probe,FE, pf, HS.B_FLOOR,HS.B_MAX,HS.STOP_TAU); pa.append(G_h[rpa])
            rand_start.append(G_h[int(rng.integers(A))])
        pstart_arm=int(np.argmax(pr_probe))
        rows[t]=dict(oracle=float(G_h.max()), fixed=float(G_h[fixed_i]),
                     uninformed3=float(np.mean(u3)), probe3=float(np.mean(p3)),
                     uninformed_adapt=float(np.mean(ua)), probe_adapt=float(np.mean(pa)),
                     probe_start=float(G_h[pstart_arm]), random_start=float(np.mean(rand_start)))
    if not rows: continue
    def col(k): return np.array([rows[t][k] for t in rows])
    key = collection
    gap3 = col("probe3")-col("uninformed3")
    gapA = col("probe_adapt")-col("uninformed_adapt")
    m,lo,hi = boot(gap3); ma,loa,hia=boot(gapA)
    out[key]=dict(n=len(rows),
        oracle=round(col("oracle").mean(),3), fixed=round(col("fixed").mean(),3),
        uninformed3=round(col("uninformed3").mean(),3), probe3=round(col("probe3").mean(),3),
        uninformed_adapt=round(col("uninformed_adapt").mean(),3), probe_adapt=round(col("probe_adapt").mean(),3),
        probe_minus_uninformed_3=[round(m,3),round(lo,3),round(hi,3)],
        probe_minus_uninformed_adapt=[round(ma,3),round(loa,3),round(hia,3)],
        probe_start=round(col("probe_start").mean(),3), random_start=round(col("random_start").mean(),3))
    print(f"\n=== {key}  (held-out n={len(rows)}) ===")
    for k in ("oracle","fixed","uninformed3","probe3","uninformed_adapt","probe_adapt","probe_start","random_start"):
        print(f"  {k:18s} {out[key][k]:.3f}")
    print(f"  probe - uninformed @3    {out[key]['probe_minus_uninformed_3']}")
    print(f"  probe - uninformed adapt {out[key]['probe_minus_uninformed_adapt']}")
json.dump(out, open("results/heldout_probe_ablation.json","w"), indent=2)
print("\nwrote results/heldout_probe_ablation.json")
