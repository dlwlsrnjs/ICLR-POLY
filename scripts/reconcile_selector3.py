#!/usr/bin/env python3
"""Read-only reconciliation diagnostic for the LOTO benign selector@3 gap (Table 1 reps=200: LG 0.727
/ MJ 0.644; Table 8 REPS=60: LG 0.702 / MJ 0.646). Recompute the DEPLOYED protocol (prior_for LOTO
benign prior, same probe-noise model, gp_bai budget 3) at increasing REPS to see if they converge =
pure Monte-Carlo. Writes results/reconcile_selector3.json; touches NO paper table/macro. CPU replay."""
import sys, json, io, contextlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
with contextlib.redirect_stdout(io.StringIO()):
    import mj_bandit_full as MJ
    import lingua_bandit_full as LG
from benign_prior import prior_for   # noqa: E402
from gp_bai import gp_bai            # noqa: E402

def models_of(mod): return getattr(mod, "MODELS", None) or getattr(mod, "STRONG")
DS = {"MultiJail": (MJ, 64), "Lingua-SafetyBench": (LG, 40)}

def selector3(mod, coll, n_items, reps, seed):
    tags = models_of(mod); G = mod.G; FE = mod.FE; names = mod.names
    priors = {t: prior_for(coll, t, names)[0] * 0.6 for t in tags}   # LOTO benign prior, cached (rep-independent)
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(reps):
        vs = []
        for t in tags:
            probe = lambda a, t=t: float(np.clip(
                G[t][a] + rng.normal(0, np.sqrt(max(G[t][a]*(1-G[t][a]), .01)/n_items)), 0, 1))
            rec, _ = gp_bai(priors[t], FE, probe, 3)
            vs.append(float(G[t][rec]))
        accs.append(np.mean(vs))
    return round(float(np.mean(accs)), 4), round(float(np.std(accs)/np.sqrt(reps)), 4)

out = {}
for coll, (mod, n) in DS.items():
    out[coll] = {}
    for reps in (60, 200, 500, 2000):
        m, se = selector3(mod, coll, n, reps, seed=0)
        out[coll][reps] = {"selector3": m, "mc_se": se}
        print(f"{coll:20s} REPS={reps:5d} -> selector@3 = {m:.4f}  (MC se {se:.4f})", flush=True)
Path("results").mkdir(exist_ok=True)
Path("results/reconcile_selector3.json").write_text(json.dumps(out, indent=2))
print("\nSaved results/reconcile_selector3.json")
