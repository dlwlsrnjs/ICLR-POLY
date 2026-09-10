#!/usr/bin/env python3
"""Two-axis capability-alignment law for the optimal reconstruction difficulty.

The capability-only law fit gated(D) = recon(D;C) * (a + b*D) and read the whole
compliance response off capability. On four similar models that looked clean
(corr(C,b)=0.92); widening the panel broke it, because compliance is set by how strong
the target's refusal barrier is, and that is not a function of capability.

This script tests whether a cheap probe of the barrier restores prediction. Probes:
  C_en     one-language reassembly rate: puzzle ability with no other language involved
  M_mean   mean rate at which the target renders a single non-English sentence into English
  M_min    the same for its weakest language, which bottlenecks a multilingual puzzle
  cap_gap  C_en - C_multi, the multilingual capability deficit
  A_en_panel   compliance on the english_direct cell of the panel run itself, so the English
               barrier measured on exactly the items and judge the attack is scored with
  A_fi_panel   the same for the official Finnish translation cell, the multilingual barrier
  gap_panel    A_fi_panel - A_en_panel
  A_over   over-refusal on 250 SAFE XSTest prompts (benign, no harmful queries)
  A_direct compliance on ~50 direct English harmful requests
  A_multi  the same requests in four non-English languages (official translations)
  A_xstest compliance on the 200 blunt UNSAFE XSTest prompts (public benchmark)
  gap      A_multi - A_direct, the English/multilingual alignment gap

Every predictor set is scored the same way: leave one target out, fit the compliance
coefficients on the rest, generate the held-out target's difficulty in closed form, and
report REGRET, the gated ASR given up against that target's own oracle grid cell.
Regret is what an attacker loses by using the law instead of brute force, and unlike a
"did the label match" score it is comparable across predictor sets.
"""
from __future__ import annotations
import argparse, glob, json, os
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.stats import pearsonr, spearmanr

ROOT = "/home/ubuntu/342/jinkwon/poly/PolyJigsaw"
CONDS = [("ordered", 0), ("shuffled", 1)]
NS = [2, 4, 6, 8, 10]

def sig(z): return 1 / (1 + np.exp(-z))
def Dval(n, sh, delta): return n + delta * sh
def recon(D, C, k): return sig(k * (C - D))

def load_panel(base: str, labels: dict) -> list[dict]:
    models = []
    for d in sorted(glob.glob(f"{base}/*/")):
        tag = Path(d).name
        bp, hp = f"{d}benign_recon.json", f"{d}harmful_summary.json"
        if not (os.path.exists(bp) and os.path.exists(hp)):
            continue
        b, h = json.load(open(bp)), json.load(open(hp))
        keys = [f"interleave_{o}_n{n}" for o, sh in CONDS for n in NS]
        if not all(k in b and k in h for k in keys):
            continue
        extra = {}
        for key, name in (("english_direct", "A_en_panel"), ("translated_direct_Finnish", "A_fi_panel")):
            if key in h:
                extra[name] = h[key]["gated"]
        if "A_en_panel" in extra and "A_fi_panel" in extra:
            extra["gap_panel"] = extra["A_fi_panel"] - extra["A_en_panel"]
        models.append({"tag": tag, "label": labels.get(tag, tag), "panel_alignment": extra,
                       "benign": {(n, sh): b[f"interleave_{o}_n{n}"] for o, sh in CONDS for n in NS},
                       "hrecon": {(n, sh): h[f"interleave_{o}_n{n}"]["recon"] for o, sh in CONDS for n in NS},
                       "gated": {(n, sh): h[f"interleave_{o}_n{n}"]["gated"] for o, sh in CONDS for n in NS}})
    return models

def fit_capability(models):
    pts = [(i, n, sh, r) for i, m in enumerate(models) for (n, sh), r in m["benign"].items()]
    nM = len(models)
    def resid(p):
        k, delta, Cs = p[0], p[1], p[2:]
        return [recon(Dval(n, sh, delta), Cs[i], k) - r for i, n, sh, r in pts]
    sol = least_squares(resid, [0.4, 8] + [10] * nM,
                        bounds=([0.05, 0] + [2] * nM, [3, 30] + [100] * nM))
    return float(sol.x[0]), float(sol.x[1]), {models[i]["tag"]: float(sol.x[2 + i]) for i in range(nM)}

def comply_fit(m, delta, n_items=250, min_recon=0.05):
    """Compliance given reconstruction, regressed on difficulty.

    Cells are weighted by how many items actually reconstructed there, because
    gated/recon is estimated from only those items and is very noisy where
    reconstruction nearly collapses. Cells below min_recon carry no usable estimate.
    """
    D, y, w = [], [], []
    for (n, sh), g in m["gated"].items():
        r = m["hrecon"][(n, sh)]
        if r >= min_recon:
            D.append(Dval(n, sh, delta)); y.append(g / r); w.append(r * n_items)
    b, a = np.polyfit(np.array(D), np.array(y), 1, w=np.sqrt(np.array(w)))
    return float(a), float(b)

def Dstar(C, k, a, b, hi=25):
    f = lambda D: -recon(D, C, k) * np.clip(a + b * D, 0, 1)
    return float(minimize_scalar(f, bounds=(2, hi), method="bounded").x)

def nearest_cell(Dv, delta):
    best = None
    for o, sh in CONDS:
        for n in NS:
            d = abs(Dval(n, sh, delta) - Dv)
            if best is None or d < best[0]:
                best = (d, n, sh, f"{o} n={n}")
    return best[1], best[2], best[3]

def loo(models, feats, C, k, delta, ab):
    tags = [m["tag"] for m in models]
    X = np.array([[1.0] + list(feats[t]) for t in tags])
    A_ = np.array([ab[t][0] for t in tags]); B_ = np.array([ab[t][1] for t in tags])
    if len(tags) - 1 <= X.shape[1]:
        return None
    rows = []
    for i, m in enumerate(models):
        idx = [j for j in range(len(tags)) if j != i]
        wa = np.linalg.lstsq(X[idx], A_[idx], rcond=None)[0]
        wb = np.linalg.lstsq(X[idx], B_[idx], rcond=None)[0]
        a_p, b_p = float(X[i] @ wa), float(X[i] @ wb)
        n_, sh_, label = nearest_cell(Dstar(C[m["tag"]], k, a_p, b_p), delta)
        got = m["gated"][(n_, sh_)]; orc = max(m["gated"].values())
        rows.append({"tag": m["tag"], "label": m["label"], "predicted": label,
                     "predicted_gated": round(got, 3), "oracle_gated": round(orc, 3),
                     "regret": round(orc - got, 3)})
    return rows

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", default=f"{ROOT}/private_artifacts/panel_v2")
    ap.add_argument("--alignment", default=f"{ROOT}/results/paper_alignment_probe.json")
    ap.add_argument("--capability", default=f"{ROOT}/results/paper_capability_probe_decomp.json")
    ap.add_argument("--out", default=f"{ROOT}/results/paper_two_axis_law.json")
    a = ap.parse_args()
    labels = json.load(open(f"{ROOT}/results/panel_labels.json"))
    models = load_panel(a.panel, labels)
    if len(models) < 5:
        raise SystemExit(f"only {len(models)} complete panel models found in {a.panel}")
    k, delta, C = fit_capability(models)
    ab = {m["tag"]: comply_fit(m, delta) for m in models}
    align = json.load(open(a.alignment))
    cap = json.load(open(a.capability)) if os.path.exists(a.capability) else {}

    print(f"panel n={len(models)}  shared k={k:.3f} delta={delta:.2f}")
    print("%-14s %6s %7s %8s %8s %8s %9s %9s %8s" % (
        "model", "C", "a", "b", "A_direct", "A_multi", "A_en_pan", "A_fi_pan", "oracle"))
    for m in models:
        t = m["tag"]; e = align.get(t, {}); q = m["panel_alignment"]
        def f(x): return "-" if x is None else ("%.3f" % x)
        print("%-14s %6.1f %7.3f %8.4f %8s %8s %9s %9s %8.3f" % (
            m["label"], C[t], ab[t][0], ab[t][1], f(e.get("A_direct")), f(e.get("A_multi")),
            f(q.get("A_en_panel")), f(q.get("A_fi_panel")), max(m["gated"].values())))

    have = [m["tag"] for m in models if m["tag"] in align]
    preds = {"C_capability": {t: C[t] for t in have}}
    for key in ("A_over_judge", "A_direct", "A_multi", "A_xstest", "gap"):
        if all(key in align[t] and align[t][key] is not None for t in have):
            preds[key] = {t: float(align[t][key]) for t in have}
    for key in ("C_en", "M_mean", "M_min", "cap_gap", "C_multi"):
        if cap and all(key in cap.get(t, {}) for t in have):
            preds[key] = {t: float(cap[t][key]) for t in have}
    pa = {m["tag"]: m["panel_alignment"] for m in models}
    for key in ("A_en_panel", "A_fi_panel", "gap_panel"):
        if all(key in pa.get(t, {}) for t in have):
            preds[key] = {t: float(pa[t][key]) for t in have}

    # What is the fitted capability made of? Multilingual puzzle ability should need both
    # monolingual reassembly and per-language reading, so report each part and the pair.
    decomp = {}
    if cap and all("C_en" in cap.get(t, {}) and "M_mean" in cap.get(t, {}) for t in have):
        y = np.array([C[t] for t in have])
        xe = np.array([cap[t]["C_en"] for t in have]); xm = np.array([cap[t]["M_mean"] for t in have])
        Xd = np.column_stack([np.ones(len(have)), xe, xm])
        w, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        pred = Xd @ w
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        decomp = {"corr_C_vs_C_en": round(float(pearsonr(xe, y)[0]), 3),
                  "corr_C_vs_M_mean": round(float(pearsonr(xm, y)[0]), 3),
                  "corr_C_en_vs_M_mean": round(float(pearsonr(xe, xm)[0]), 3),
                  "R2_C_from_both": round(float(r2), 3)}
        print("\n=== what the fitted capability is made of ===")
        print("  corr(C, C_en)=%+.3f  corr(C, M_mean)=%+.3f  corr(C_en, M_mean)=%+.3f  R2(C ~ C_en + M_mean)=%.3f"
              % (decomp["corr_C_vs_C_en"], decomp["corr_C_vs_M_mean"],
                 decomp["corr_C_en_vs_M_mean"], decomp["R2_C_from_both"]))

    print("\n=== which predictor explains the compliance coefficients? (n=%d) ===" % len(have))
    cors = {}
    for pname, pv in preds.items():
        x = np.array([pv[t] for t in have])
        for tname, arr in (("intercept_a", np.array([ab[t][0] for t in have])),
                           ("slope_b", np.array([ab[t][1] for t in have]))):
            r, p = pearsonr(x, arr); rho = spearmanr(x, arr)[0]
            cors[f"{pname}->{tname}"] = {"pearson": round(float(r), 3), "p": round(float(p), 4),
                                         "spearman": round(float(rho), 3)}
            print("  %-14s -> %-12s r=%+.3f (p=%.3f) rho=%+.3f" % (pname, tname, r, p, rho))

    print("\n=== leave-one-out difficulty generation (regret = gated ASR lost vs oracle) ===")
    sub = [m for m in models if m["tag"] in have]
    variants = {}
    cands = [p for p in preds if p != "C_capability"]
    sets = [("capability only", ["C_capability"])]
    sets += [(f"alignment only: {c}", [c]) for c in cands]
    sets += [(f"capability + {c}", ["C_capability", c]) for c in cands]
    for name, keys in sets:
        feats = {t: [preds[kk][t] for kk in keys] for t in have}
        rows = loo(sub, feats, C, k, delta, ab)
        if rows is None:
            print("  %-34s skipped (too few models)" % name); continue
        mr = float(np.mean([r["regret"] for r in rows])); wr = float(np.max([r["regret"] for r in rows]))
        variants[name] = {"mean_regret": round(mr, 4), "max_regret": round(wr, 4), "rows": rows}
        print("  %-34s mean regret %.3f   worst %.3f" % (name, mr, wr))
    orc = {m["tag"]: max(m["gated"].values()) for m in sub}
    fixed = {f"{o} n={n}": float(np.mean([orc[m["tag"]] - m["gated"][(n, sh)] for m in sub]))
             for o, sh in CONDS for n in NS}
    bf = min(fixed.items(), key=lambda kv: kv[1])
    print("  %-34s mean regret %.3f   (%s)" % ("best single fixed setting", bf[1], bf[0]))

    out = {"panel": [m["label"] for m in models], "k": k, "delta": delta,
           "C": {m["label"]: C[m["tag"]] for m in models},
           "comply": {m["label"]: ab[m["tag"]] for m in models},
           "oracle": {m["label"]: max(m["gated"].values()) for m in models},
           "alignment": {t: align[t] for t in have},
           "capability_decomposition": {t: cap[t] for t in have if t in cap},
           "capability_decomposition_fit": decomp,
           "panel_alignment": {m["label"]: m["panel_alignment"] for m in models},
           "correlations": cors, "loo": variants,
           "best_fixed": {"setting": bf[0], "mean_regret": round(bf[1], 4)},
           "fixed_setting_regret": {kk: round(v, 4) for kk, v in sorted(fixed.items(), key=lambda kv: kv[1])}}
    json.dump(out, open(a.out, "w"), indent=2)
    print("\nsaved", a.out)

if __name__ == "__main__":
    main()
