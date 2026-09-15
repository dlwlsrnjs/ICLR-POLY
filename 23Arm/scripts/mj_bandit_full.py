#!/usr/bin/env python3
"""MJ adaptive selector over a SUPERSET arm space that subsumes single-vector baselines as arms.
Arms per (strong) model: multilingual amount(n2..10) + disorder(delta) + combo(ours/persona/incept)
+ our strengthened triple(hi_wl / triple / triple_en) + single-vector transforms as n=1 arms
(aim=persona, deepinception=fiction, pap=persuasion). Point: the model that resists our multilingual
mechanism but falls to a single-vector attack (gemma-2-9b -> AIM) is covered because AIM is just the
(n=1 + persona) special case of the arm space; the benign-recon warm-start still finds it. Pure
offline computation over existing results (no GPU). Reports per-model oracle, best fixed arm, and the
budget curve for flat / benign-recon / leave-one-out priors."""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gp_bai import gp_bai
from arm_scoring import arm_score, arm_recon
from benign_prior import prior_for   # harmless probe only; see docs/DATA_AUDIT_2026-09-07.md   # one consistent gate rule (docs/DATA_AUDIT_2026-09-07.md)

STRONG = ["qwen25_3b","qwen25_7b","qwen25_14b","qwen25_32b","llama32_3b_it","llama31_8b_it","gemma2_2b_it","gemma2_9b_it","gemma2_27b"]
R = "results"


def jload(p):
    return json.load(open(p))


def _row(name, rec, feat):
    """(name, score, recon, features) under the consistent gate rule."""
    return (name, arm_score(name, rec["gated"], rec["unsafe"]), arm_recon(name, rec["recon"]), feat)


def arms_for(t):
    """Return list of (name, gated, recon, feature-vector) for model t."""
    out = []
    seq = jload(f"{R}/mj_sequential_20260906/{t}.json")["steps"]
    for s in seq:                                   # multilingual amount arms
        out.append(_row(f"amt_n{s['n']}", s, [s["n"] / 10, 0, 0, 0, 0]))
    dis = jload(f"{R}/mj_disorder_20260906/{t}.json")
    for r in (dis.get("rows") or dis.get("steps")):
        if r["delta"] == 0.0:
            continue
        out.append(_row(f"dis_{r['delta']}", r, [0.4, r["delta"], 0, 0, 0]))
    cmb = jload(f"{R}/mj_combo_20260906/{t}.json")["variants"]
    fmap = {"ours": [0.4, 0, 0, 0, 1], "ours_persona": [0.4, 0, 1, 0, 1],
            "ours_incept": [0.4, 0, 0, 1, 1], "incept_only": [0.1, 0, 0, 1, 0]}
    for k, v in cmb.items():
        out.append(_row(f"combo_{k}", v, fmap.get(k, [0.4, 0, 0, 0, 1])))
    # role separation, English answer only. Low-resource answers are dropped: the safety judge
    # scores an English answer reliably but over-counts a low-resource one (Section on judge
    # accuracy). hi_en is the persona-free arm, triple_en the persona arm; both answer in English.
    tri = jload(f"{R}/mj_triple_20260906/{t}.json")["variants"]
    hien = jload(f"{R}/mj_hien_20260908/{t}.json")["variants"]
    out.append(_row("tri_hi_en", hien["hi_en"], [0.4, 0, 0, 0, 0]))
    out.append(_row("tri_triple_en", tri["triple_en"], [0.4, 0, 1, 0, 0]))
    meth = jload(f"{R}/mj_method_20260906/{t}.json")["methods"]     # single-vector transforms as n=1 arms
    mf = {"aim": [0.1, 0, 1, 0, 0], "deepinception": [0.1, 0, 0, 1, 0], "pap": [0.1, 0, 0, 0, 0]}
    for k in ("aim", "deepinception", "pap"):
        v = meth[k]
        out.append(_row(f"m_{k}", v, mf[k]))
    # low-resource single-language translation: the canonical multilingual baseline, given the
    # request whole in another language, so clear-text and scored ungated (arm_scoring).
    tr = jload(f"{R}/extra_arms_mj_20260908/{t}.json")["configs"]["translated"]
    out.append(_row("m_translated", tr, [0.1, 0, 0, 0, 0]))
    # Separate arms that shared a feature vector so the GP surrogate cannot conflate them: the
    # clear-text translation attack collided with PAP (best vs worst on several targets), and role
    # separation collided with plain amount. Two indicator dimensions fix both. This is a
    # representation correction, not a per-target tuning: the flags are fixed by arm identity.
    out = [(nm, sc, rc, ft + [1.0 if nm == "m_translated" else 0.0,
                              1.0 if nm.startswith("tri_") else 0.0]) for (nm, sc, rc, ft) in out]
    return out


# consistent arm ordering across models (names identical across the 3 strong models)
names = [a[0] for a in arms_for(STRONG[0])]
for t in STRONG:
    assert [a[0] for a in arms_for(t)] == names, f"arm mismatch {t}"
FE = np.array([a[3] for a in arms_for(STRONG[0])], float)
G = {t: np.array([a[1] for a in arms_for(t)]) for t in STRONG}
RC = {t: np.array([a[2] for a in arms_for(t)]) for t in STRONG}

oracle = np.mean([G[t].max() for t in STRONG])
fixed_arm = int(np.argmax(np.mean([G[t] for t in STRONG], axis=0)))
fixed = np.mean([G[t][fixed_arm] for t in STRONG])
rng = np.random.default_rng(0)


def run(prior_kind, B, reps=200):
    accs = []
    for _ in range(reps):
        vs = []
        for t in STRONG:
            if prior_kind == "benign-recon":
                # harmless probe, with the candidate chosen leave-one-target-out. NEVER RC[t]:
                # that column comes from the harmful runs and shares a factor with the target.
                pr = prior_for("MultiJail", t, names)[0] * 0.6
            elif prior_kind == "flat":
                pr = np.full(len(names), float(np.mean([G[x].mean() for x in STRONG])))
            else:
                pr = np.mean([G[x] for x in STRONG if x != t], axis=0)
            # binomial standard error at this collection's item count, the same probe-noise
            # model bandit_bootstrap_ci.py and query_efficiency.py use, so every table agrees.
            probe = lambda a, t=t: float(np.clip(G[t][a] + rng.normal(
                0, np.sqrt(max(G[t][a] * (1 - G[t][a]), .01) / 64)), 0, 1))
            rec, _ = gp_bai(pr, FE, probe, B)
            vs.append(G[t][rec])
        accs.append(np.mean(vs))
    return float(np.mean(accs))


print(f"MJ 강모델 {len(STRONG)} | arm {len(names)} (양+혼란도+결합+triple+단일벡터aim/incept/pap)")
print(f"오라클(전-arm) {oracle:.3f} | 고정최선({names[fixed_arm]}) {fixed:.3f}\n")
print("모델별 전-arm 오라클 (단일벡터 포함):")
for t in STRONG:
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
print(f"\n오라클={oracle:.3f}. 단일벡터(aim 등)를 arm으로 편입 → gemma가 arm공간에 포섭됨.")

# save summary
summ = {"models": STRONG, "arms": names, "oracle": round(oracle, 3),
        "fixed_arm": names[fixed_arm], "fixed": round(fixed, 3),
        "per_model_best": {t: {"arm": names[int(np.argmax(G[t]))], "gated": round(float(G[t].max()), 3)} for t in STRONG},
        "budget": CURVE, "scoring_rule": "arm_scoring.arm_score: clear-text baseline arms are ungated (unsafe), hidden arms strict (gated)"}
op = Path(R) / "mj_bandit_full_20260906.json"
fd = os.open(op, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as h:
    json.dump(summ, h, indent=2)
print("saved", op)
