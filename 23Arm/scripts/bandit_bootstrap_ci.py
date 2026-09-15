#!/usr/bin/env python3
"""Bootstrap 95% CIs (resampling the model panel) for the headline bandit numbers on each dataset:
oracle, best fixed arm, and the adaptive selector at its operating budget. Reuses the arm/gated
matrices built by mj_bandit_full and lingua_bandit_full. Pure offline (no GPU). Model-level
bootstrap (n = size of the panel, currently 9) — reported as such; item-level CIs would need
per-item judgments (not stored).

Priors: the headline tables (mj_bandit_full / lingua_bandit_full) select with the BENIGN-RECON prior
(prior_mean = recon * 0.6, no harmful supervision), so that is the primary estimate here. The
leave-one-out cross-target prior (mean gated of the other models, i.e. harmful data from other
targets) is reported alongside as a secondary row. On the 7-model panel the two nearly coincided;
on 9 models they diverge (MultiJail@2: 0.61 benign vs 0.54 loo), so a table must state which it uses.
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benign_prior import prior_for   # harmless probe only

PRIORS = ("benign-recon", "loo-data")
# difficulty-adaptive budget (the deployed selector, paper-wide): floor = the reported budget, up to
# B_MAX queries while no arm has been observed at >= STOP_TAU. All fixed a priori. The fixed-budget
# curve stays the query-efficiency analysis in the bandit summaries; this is the operating point.
B_MAX, STOP_TAU = 8, 0.5


def load_module(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def analyze(mod, tag, budget):
    models = mod.STRONG if hasattr(mod, "STRONG") else mod.MODELS
    G = mod.G; RC = mod.RC; FE = mod.FE; names = mod.names
    from gp_bai import gp_bai, gp_bai_adaptive
    rng = np.random.default_rng(0)

    allm = models
    n_items = 64 if tag.startswith("Multi") else 40

    def _probe(t):
        return lambda a, t=t: float(np.clip(G[t][a] + rng.normal(0, np.sqrt(max(G[t][a] * (1 - G[t][a]), .01) / n_items)), 0, 1))

    coll = "MultiJail" if tag.startswith("Multi") else "Lingua-SafetyBench"

    def prior_of(t, kind):
        if kind == "benign-recon":
            return prior_for(coll, t, names)[0] * 0.6              # harmless probe, never RC[t]
        return np.mean([G[x] for x in allm if x != t], axis=0)     # cross-target harmful prior

    def adaptive_on(sample_models, B, kind, reps=60):
        accs = []
        for _ in range(reps):
            vs = []
            for t in sample_models:
                rec, _ = gp_bai_adaptive(prior_of(t, kind), FE, _probe(t), B, B_MAX, STOP_TAU)
                vs.append(G[t][rec])
            accs.append(np.mean(vs))
        return float(np.mean(accs))

    # fixed best arm chosen on the FULL panel (as in the paper), evaluated per bootstrap sample
    fixed_arm = int(np.argmax(np.mean([G[t] for t in models], axis=0)))
    # oracle/fixed: vectorized full bootstrap (cheap). adaptive: fewer resamples (gp_bai per model).
    Gmax = np.array([G[t].max() for t in models]); Gfix = np.array([G[t][fixed_arm] for t in models])
    idx = rng.integers(0, len(models), size=(2000, len(models)))
    boot = {"oracle": list(Gmax[idx].mean(1)), "fixed": list(Gfix[idx].mean(1))}
    # Per-model adaptive value at high reps, computed once and reused for the headline CI and the
    # paired deltas. The panel-mean CI is a bootstrap over the nine per-model values; this replaces a
    # nested model-bootstrap with too few inner reps, which under-converged on bimodal targets.
    REPS_PT = 1500
    adapt_pm, qbar = {}, {}
    for kind in PRIORS:
        adapt_pm[kind] = {}
        for t in models:
            picks, qs = [], []
            for _ in range(REPS_PT):
                rec, obs = gp_bai_adaptive(prior_of(t, kind), FE, _probe(t), budget, B_MAX, STOP_TAU)
                picks.append(G[t][rec]); qs.append(len(obs))
            adapt_pm[kind][t] = float(np.mean(picks))
            if kind == "benign-recon":
                qbar[t] = float(np.mean(qs))
        vals = np.array([adapt_pm[kind][t] for t in models])
        bi = rng.integers(0, len(models), size=(5000, len(models)))
        boot[f"adaptive[{kind}]"] = list(vals[bi].mean(1))

    def ci(x):
        a = np.array(x)
        return round(float(a.mean()), 3), round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)

    print(f"\n=== {tag} (n={len(models)} 모델, 적응예산 바닥{budget}·상한{B_MAX}·τ{STOP_TAU}, per-model {REPS_PT}rep, 모델 부트스트랩) ===")
    out = {}
    labels = {"oracle": "오라클", "fixed": f"최선고정({names[fixed_arm]})",
              "adaptive[benign-recon]": f"적응예산(benign-recon, 주)",
              "adaptive[loo-data]": f"적응예산(loo-data, 부)"}
    for k in ("oracle", "fixed", *[f"adaptive[{p}]" for p in PRIORS]):
        m, lo, hi = ci(boot[k])
        out[k] = {"mean": m, "ci95": [lo, hi]}
        print(f"  {labels[k]:32s} {m:.3f}  95%CI[{lo:.3f}, {hi:.3f}]")
    out["adaptive"] = out["adaptive[benign-recon]"]   # primary, for consumers reading the old key

    # PAIRED: per-model adaptive - fixed, from the same high-rep per-model estimates
    for kind in PRIORS:
        adapt_t = adapt_pm[kind]
        deltas = np.array([adapt_t[t] - G[t][fixed_arm] for t in models])
        di = rng.integers(0, len(models), size=(5000, len(models)))
        bd = deltas[di].mean(1)
        dm, dlo, dhi = round(float(deltas.mean()), 3), round(float(np.percentile(bd, 2.5)), 3), round(float(np.percentile(bd, 97.5)), 3)
        n_pos = int((deltas >= -1e-9).sum())
        print(f"  paired Δ(적응−고정) [{kind:12s}] {dm:+.3f}  95%CI[{dlo:+.3f}, {dhi:+.3f}]  (개선 모델 {n_pos}/{len(models)})")
        out[f"paired_delta[{kind}]"] = {"mean": dm, "ci95": [dlo, dhi], "n_improved": n_pos, "n": len(models),
                                        "per_model": {t: round(adapt_t[t] - G[t][fixed_arm], 3) for t in models}}
    out["paired_delta"] = out["paired_delta[benign-recon]"]   # primary
    out["mean_queries"] = round(float(np.mean(list(qbar.values()))), 2)
    out["queries_per_model"] = {t: round(q, 2) for t, q in qbar.items()}
    print(f"  적응형 평균 질의수(benign-recon): {out['mean_queries']:.2f}  (바닥 {budget}, 상한 {B_MAX}, 정지 τ={STOP_TAU})")

    # paired difference against the cheap adaptive baseline: trying each published single-vector
    # attack once, which costs the same three harmful queries and needs no configuration space.
    SINGLE = ("m_aim", "m_deepinception", "m_pap")
    si = [names.index(a) for a in SINGLE if a in names]
    if si:
        base_t = {}
        for t in models:
            true = np.array([G[t][i] for i in si])
            picks = []
            for _ in range(400):
                obs = np.clip(true + rng.normal(0, np.sqrt(np.maximum(true * (1 - true), .01) / n_items)), 0, 1)
                picks.append(true[int(np.argmax(obs))])
            base_t[t] = float(np.mean(picks))
        for kind in PRIORS:
            adapt_t = adapt_pm[kind]   # reuse the high-rep per-model estimates
            d = np.array([adapt_t[t] - base_t[t] for t in models])
            di = rng.integers(0, len(models), size=(5000, len(models)))
            bd = d[di].mean(1)
            dm = round(float(d.mean()), 3)
            lo, hi = round(float(np.percentile(bd, 2.5)), 3), round(float(np.percentile(bd, 97.5)), 3)
            print(f"  paired vs published-3 [{kind:12s}] {dm:+.3f}  95%CI[{lo:+.3f}, {hi:+.3f}]  "
                  f"(improved {int((d >= -1e-9).sum())}/{len(models)})")
            out[f"paired_vs_published[{kind}]"] = {
                "mean": dm, "ci95": [lo, hi], "n_improved": int((d >= -1e-9).sum()), "n": len(models),
                "baseline_mean": round(float(np.mean(list(base_t.values()))), 3)}
    return out


if __name__ == "__main__":
    mj = load_module("mjb", "scripts/mj_bandit_full.py")
    lg = load_module("lgb", "scripts/lingua_bandit_full.py")
    # one difficulty-adaptive operating point for BOTH datasets: floor 3, cap 8, stop at 0.5. The
    # floor/cap/threshold are fixed a priori for both datasets; picking a different budget per dataset
    # would be selecting on the result. The fixed-budget curve stays in the bandit summaries.
    res = {"multijail": analyze(mj, "MultiJail", 3),
           "lingua": analyze(lg, "Lingua-SafetyBench", 3),
           "note": "adaptive = difficulty-adaptive budget (floor 3, cap 8, stop 0.5), benign-recon prior "
                   "(matches the headline tables); the [loo-data] keys are the cross-target harmful prior."}
    Path("results/bandit_ci_20260907.json").write_text(json.dumps(res, indent=2))
    print("\nsaved results/bandit_ci_20260907.json")
