#!/usr/bin/env python3
"""Lingua-SafetyBench adaptive selector over the SAME superset arm space as the MJ bandit, so the two
datasets' tables are directly comparable. Arms per model: multilingual amount(n2..10) + disorder(delta)
+ combo(ours/persona/incept) + single-vector transforms as n=1 arms (aim=persona, deepinception=fiction,
pap=persuasion). (No low-resource weak-lang answer arm here: Lingua's languages are high/mid-resource,
so that MJ-specific arm does not apply.) Pure offline computation over existing results. Reports
per-model oracle, best fixed arm, and the budget curve for flat / benign-recon / leave-one-out priors."""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gp_bai import gp_bai
from arm_scoring import arm_score, arm_recon
from benign_prior import prior_for   # harmless probe only; see docs/DATA_AUDIT_2026-09-07.md   # one consistent gate rule (docs/DATA_AUDIT_2026-09-07.md)

R = "results"
MODELS = ["qwen25_3b","qwen25_7b","qwen25_14b","qwen25_32b","llama32_3b_it","llama31_8b_it","gemma2_2b_it","gemma2_9b_it","gemma2_27b"]


def jload(p):
    return json.load(open(p))


def _row(name, rec, feat):
    """(name, score, recon, features) under the consistent gate rule."""
    return (name, arm_score(name, rec["gated"], rec["unsafe"]), arm_recon(name, rec["recon"]), feat)


def arms_for(t):
    out = []
    for s in jload(f"{R}/sequential_resource_20260906/{t}.json")["steps"]:
        out.append(_row(f"amt_n{s['n']}", s, [s["n"] / 10, 0, 0, 0]))
    dis = jload(f"{R}/disorder_sweep_20260906/{t}.json")
    for r in (dis.get("rows") or dis.get("steps")):
        if r["delta"] == 0.0:
            continue
        out.append(_row(f"dis_{r['delta']}", r, [0.4, r["delta"], 0, 0]))
    cmb = jload(f"{R}/combo_20260906/{t}.json")["variants"]
    fmap = {"ours": [0.4, 0, 0, 1], "ours_persona": [0.4, 0, 1, 1],
            "ours_incept": [0.4, 0, 0, 1], "incept_only": [0.1, 0, 0, 1]}
    for k, v in cmb.items():
        out.append(_row(f"combo_{k}", v, fmap.get(k, [0.4, 0, 0, 1])))
    # role separation, English answer only (see Section on judge accuracy).
    tri = jload(f"{R}/lingua_triple_20260907/{t}.json")["variants"]
    hien = jload(f"{R}/lingua_hien_20260908/{t}.json")["variants"]
    out.append(_row("tri_hi_en", hien["hi_en"], [0.4, 0, 0, 0]))
    out.append(_row("tri_triple_en", tri["triple_en"], [0.4, 0, 1, 0]))
    meth = jload(f"{R}/method_baselines_v2_20260906/{t}.json")["methods"]
    mf = {"aim": [0.1, 0, 1, 0], "deepinception": [0.1, 0, 0, 1], "pap": [0.1, 0, 0, 0]}
    for k in ("aim", "deepinception", "pap"):
        v = meth[k]
        out.append(_row(f"m_{k}", v, mf[k]))
    # low-resource single-language translation: the canonical multilingual baseline, given the
    # request whole in another language, so clear-text and scored ungated (arm_scoring).
    tr = jload(f"{R}/extra_arms_lg_20260908/{t}.json")["configs"]["translated"]
    out.append(_row("m_translated", tr, [0.1, 0, 0, 0]))
    # See mj_bandit_full: two indicator dimensions separate arms that shared a feature vector (the
    # clear-text translation attack vs PAP; role separation vs plain amount) so the GP cannot conflate
    # them. Fixed by arm identity, not tuned per target.
    out = [(nm, sc, rc, ft + [1.0 if nm == "m_translated" else 0.0,
                              1.0 if nm.startswith("tri_") else 0.0]) for (nm, sc, rc, ft) in out]
    return out


names = [a[0] for a in arms_for(MODELS[0])]
for t in MODELS:
    assert [a[0] for a in arms_for(t)] == names, f"arm mismatch {t}: {[a[0] for a in arms_for(t)]}"
FE = np.array([a[3] for a in arms_for(MODELS[0])], float)
G = {t: np.array([a[1] for a in arms_for(t)]) for t in MODELS}
RC = {t: np.array([a[2] for a in arms_for(t)]) for t in MODELS}

oracle = np.mean([G[t].max() for t in MODELS])
fixed_arm = int(np.argmax(np.mean([G[t] for t in MODELS], axis=0)))
fixed = np.mean([G[t][fixed_arm] for t in MODELS])
rng = np.random.default_rng(0)


def run(prior_kind, B, reps=200):
    accs = []
    for _ in range(reps):
        vs = []
        for t in MODELS:
            if prior_kind == "benign-recon":
                # harmless probe, with the candidate chosen leave-one-target-out. NEVER RC[t]:
                # that column comes from the harmful runs and shares a factor with the target.
                pr = prior_for("Lingua-SafetyBench", t, names)[0] * 0.6
            elif prior_kind == "flat":
                pr = np.full(len(names), float(np.mean([G[x].mean() for x in MODELS])))
            else:
                pr = np.mean([G[x] for x in MODELS if x != t], axis=0)
            # binomial standard error at this collection's item count, the same probe-noise
            # model bandit_bootstrap_ci.py and query_efficiency.py use, so every table agrees.
            probe = lambda a, t=t: float(np.clip(G[t][a] + rng.normal(
                0, np.sqrt(max(G[t][a] * (1 - G[t][a]), .01) / 40)), 0, 1))
            rec, _ = gp_bai(pr, FE, probe, B)
            vs.append(G[t][rec])
        accs.append(np.mean(vs))
    return float(np.mean(accs))


print(f"Lingua {len(MODELS)} | arm {len(names)} (양+혼란도+결합+단일벡터aim/incept/pap)")
print(f"오라클(전-arm) {oracle:.3f} | 고정최선({names[fixed_arm]}) {fixed:.3f}\n")
print("모델별 전-arm 오라클 (단일벡터 포함):")
for t in MODELS:
    i = int(np.argmax(G[t]))
    top3 = sorted(range(len(names)), key=lambda j: -G[t][j])[:3]
    print(f"  {t:16s} best={names[i]:14s} {G[t].max():.3f}   top3: " +
          ", ".join(f"{names[j]}={G[t][j]:.2f}" for j in top3))
# compute the curve ONCE, then both print and save it: printing and saving used to call
# run() twice off the same RNG, so the logged table and the saved JSON disagreed by ~0.01.
CURVE = {B: {k: round(run(k, B), 3) for k in ("flat", "benign-recon", "loo-data")} for B in (1, 2, 3, 4, 6)}
print(f"\n{'예산':6s}{'flat':10s}{'benign-recon':14s}{'loo-data'}")
for B, row in CURVE.items():
    print(f"{B:<6}{row['flat']:.3f}     {row['benign-recon']:.3f}         {row['loo-data']:.3f}")

summ = {"models": MODELS, "arms": names, "oracle": round(oracle, 3),
        "fixed_arm": names[fixed_arm], "fixed": round(fixed, 3),
        "per_model_best": {t: {"arm": names[int(np.argmax(G[t]))], "gated": round(float(G[t].max()), 3)} for t in MODELS},
        "budget": CURVE, "scoring_rule": "arm_scoring.arm_score: clear-text baseline arms are ungated (unsafe), hidden arms strict (gated)"}
op = Path(R) / "lingua_bandit_full_20260906.json"
fd = os.open(op, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as h:
    json.dump(summ, h, indent=2)
print("saved", op)
