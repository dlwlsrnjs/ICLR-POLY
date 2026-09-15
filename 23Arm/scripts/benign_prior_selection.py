#!/usr/bin/env python3
"""Which harmless measurement should warm-start the selector?

The paper claims the prior costs no harmful supervision. That is only true if the prior is computed
from harmless interactions. This script compares every candidate prior on both collections:

  harmful-recon      the reconstruction column of the HARMFUL runs. This is what the earlier code
                     used. It leaks, because verified ASR = recon AND unsafe, so the prior already
                     contains a factor of the target. Reported as a reference, never as a result.
  benign-recon       reconstruction rate on harmless FLORES puzzles (benign_arms_probe)
  benign-signals     the harmless answer-frame signals (benign_signals_probe): non-refusal,
                     answer-language fidelity, persona hold, fiction hold, detail
  borderline-*       the same signals measured on safety-adjacent but still harmless requests
                     (forensics, locksmithing, penetration testing), where a strongly aligned model
                     hedges even though nothing harmful is asked
  products           benign-recon x each signal, i.e. comprehension x willingness
  flat               a constant (no information)

Each candidate is scored two ways: (a) Spearman correlation with verified ASR across the 23
configurations, per target; (b) the verified ASR the selector actually attains at a three-query
budget. To avoid choosing a prior on the same panel it is evaluated on, the headline choice is made
leave-one-target-out: for each target the best candidate is picked on the other eight and applied to
the held-out one.

Writes results/benign_prior_selection_20260907.json and paper/tab_sel_prior.tex.
"""
from __future__ import annotations
import io, json, contextlib, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
with contextlib.redirect_stdout(io.StringIO()):
    import mj_bandit_full as MJ
    import lingua_bandit_full as LG
from gp_bai import gp_bai  # noqa: E402

BUDGET = 3
REPS = 60
SIGNALS = ["nonrefusal", "lang_fidelity", "persona_hold", "fiction_hold", "detail"]
DATASETS = {
    "MultiJail": dict(mod=MJ, n_items=64, recon="results/benign_arms_mj_20260907",
                      sig="results/benign_signals_mj_20260907",
                      bord="results/benign_borderline_mj_20260907"),
    "Lingua-SafetyBench": dict(mod=LG, n_items=40, recon="results/benign_arms_20260907",
                               sig="results/benign_signals_20260907",
                               bord="results/benign_borderline_20260907"),
}


def panel(mod):
    return mod.STRONG if hasattr(mod, "STRONG") else mod.MODELS


# How many harmless prompts the attacker must actually run. The probe measures one reconstruction
# rate per distinct benign construction; a coverage variant measures only a subset and lets the GP
# fill the rest from the feature space, which is what makes a cheap probe possible.
COVERAGE = {
    "full (15 prompts)": None,
    "amount only (9 prompts)": lambda k: k.startswith("amt_"),
    "three prompts (n=4, role-split, plain)": lambda k: k in ("amt_n4", "triple_hi", "plain"),
    "one prompt (n=4)": lambda k: k == "amt_n4",
}


def load_benign(cfg, mod):
    """{target: {candidate: vector over arms}} from harmless measurements only, plus the leaky
    reference, which is labelled as such and never used as a result."""
    import benign_arms_probe as BA
    models = panel(mod); names = mod.names
    out = {}
    for t in models:
        cand = {"harmful-recon (leaks)": mod.RC[t].copy(),
                "flat": np.full(len(names), 0.5)}
        p = Path(cfg["recon"]) / f"{t}.json"
        if p.exists():
            blob = json.loads(p.read_text())
            ar = blob["arm_recon"]; cr = blob.get("cfg_recon", {})
            # the English-answer role-split arms and the translated arm reuse probed constructions
            # (see benign_prior.py); map the new names onto what the harmless probe measured.
            ralias = {"tri_hi_en": "tri_hi_wl", "tri_triple_en": "tri_triple_en"}
            cand["benign-recon"] = np.array(
                [1.0 if a == "m_translated" else ar[ralias.get(a, a)] for a in names], float)
            for lbl, keep in COVERAGE.items():
                if keep is None:
                    continue
                seen = {k: v for k, v in cr.items() if keep(k)}
                if not seen:
                    continue
                fill = float(np.mean(list(seen.values())))
                cand[f"benign-recon, {lbl}"] = np.array(
                    [seen.get(BA.arm_to_cfg(a), fill) for a in names], float)
        for tag, key in (("", "sig"), ("borderline-", "bord")):
            q = Path(cfg[key]) / f"{t}.json"
            if not q.exists():
                continue
            sg = json.loads(q.read_text())["arm_signals"]
            salias = {"tri_hi_en": "tri_hi_wl", "tri_triple_en": "tri_triple_en", "m_translated": "m_pap"}
            for s in SIGNALS:
                cand[f"benign-{tag}{s}"] = np.array([sg[salias.get(a, a)][s] for a in names], float)
            if "benign-recon" in cand:
                for s in SIGNALS:
                    v = np.array([sg[salias.get(a, a)][s] for a in names], float)
                    cand[f"benign-recon x {tag}{s}"] = cand["benign-recon"] * v
        out[t] = cand
    return out


def evaluate(cfg):
    mod = cfg["mod"]; models = panel(mod); G = mod.G; FE = mod.FE
    B = load_benign(cfg, mod)
    # A candidate is evaluated on the targets where its probe exists. Requiring full coverage would
    # drop a whole candidate because one target's probe is missing, so we keep it and report how many
    # targets it covers; only fully covered candidates are eligible for the headline choice.
    cands = sorted({c for t in models for c in B[t]})
    rng = np.random.default_rng(0)

    def probe(t):
        return lambda a, t=t: float(np.clip(
            G[t][a] + rng.normal(0, np.sqrt(max(G[t][a] * (1 - G[t][a]), .01) / cfg["n_items"])), 0, 1))

    res = {}
    for c in cands:
        rho, sel = {}, {}
        for t in models:
            if c not in B[t]:
                continue
            v = B[t][c]
            rho[t] = 0.0 if np.allclose(v, v[0]) else float(spearmanr(v, G[t]).statistic)
            picks = []
            for _ in range(REPS):
                rec, _ = gp_bai(v * 0.6, FE, probe(t), BUDGET)
                picks.append(G[t][rec])
            sel[t] = float(np.mean(picks))
        res[c] = {"n_targets": len(sel), "full_coverage": len(sel) == len(models),
                  "rho_mean": round(float(np.mean(list(rho.values()))), 3),
                  "rho_per_target": {t: round(v, 3) for t, v in rho.items()},
                  "selector_at_3": round(float(np.mean(list(sel.values()))), 3),
                  "selector_per_target": {t: round(v, 3) for t, v in sel.items()}}
    # leave-one-target-out choice among harmless candidates only
    # restrict LOTO to DEPLOYABLE candidates: benign_prior.candidates() cannot produce the
    # probe-count-ablation variants ("benign-recon, <coverage>"), so picking one there makes prior_for
    # silently fall back to the default at deploy time (was the Table 1 vs Table 8 gemma2_2b_it gap).
    harmless = [c for c in cands if not c.startswith("harmful") and c != "flat"
                and not c.startswith("benign-recon, ")
                and res[c]["full_coverage"]]
    loo_vals, loo_pick = [], {}
    for t in models:
        best = max(harmless, key=lambda c: np.mean([res[c]["selector_per_target"][x]
                                                    for x in models if x != t]))
        loo_pick[t] = best
        loo_vals.append(res[best]["selector_per_target"][t])
    res["__loo__"] = {"selector_at_3": round(float(np.mean(loo_vals)), 3), "picked_per_target": loo_pick,
                      "oracle": round(float(np.mean([G[t].max() for t in models])), 3)}
    return res


def main():
    out = {}
    for name, cfg in DATASETS.items():
        print(f"=== {name} ===", flush=True)
        out[name] = evaluate(cfg)
        rows = [(c, v["rho_mean"], v["selector_at_3"]) for c, v in out[name].items() if c != "__loo__"]
        for c, r, s in sorted(rows, key=lambda x: -x[2]):
            n = out[name][c]["n_targets"]
            mark = "" if out[name][c]["full_coverage"] else f"  (targets {n})"
            print(f"  {c:42s} rho {r:+.3f}   selector@3 {s:.3f}{mark}")
        print(f"  leave-one-target-out choice -> {out[name]['__loo__']['selector_at_3']:.3f} "
              f"(oracle {out[name]['__loo__']['oracle']:.3f})")
        print(f"  picked: {out[name]['__loo__']['picked_per_target']}")
    Path("results/benign_prior_selection_20260907.json").write_text(json.dumps(out, indent=2))

    keep = ["flat", "one prompt (n=4)", "three prompts (n=4, role-split, plain)",
            "amount only (9 prompts)", "benign-recon", "benign-lang_fidelity", "benign-persona_hold",
            "benign-borderline-nonrefusal", "benign-recon x lang_fidelity",
            "benign-recon x borderline-nonrefusal", "harmful-recon (leaks)"]
    keep = [("benign-recon, " + k) if k in COVERAGE and k != "full (15 prompts)" else k for k in keep]
    L = ["\\begin{table}[t]", "\\centering", "\\small",
         "\\caption{Which harmless measurement should warm-start the search. Each row is a prior "
         "computed \\emph{without any harmful query}, except the last, which is the reconstruction "
         "column of the harmful runs and is listed only to show what leaked supervision would buy. "
         "$\\rho$ is the mean Spearman correlation with verified ASR over the 23 configurations; the "
         "selector column is verified ASR at a three-query budget.}",
         "\\label{tab:sel_prior}", "\\begin{tabular}{lcccc}", "\\toprule",
         "& \\multicolumn{2}{c}{MultiJail} & \\multicolumn{2}{c}{Lingua-SafetyBench} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
         "Warm-start prior & $\\rho$ & selector@3 & $\\rho$ & selector@3 \\\\", "\\midrule"]
    pretty = {"flat": "none (constant prior)",
              "benign-recon": "benign reconstruction, full probe (15 prompts)",
              "benign-recon, one prompt (n=4)": "benign reconstruction, 1 probe prompt",
              "benign-recon, three prompts (n=4, role-split, plain)": "benign reconstruction, 3 probe prompts",
              "benign-recon, amount only (9 prompts)": "benign reconstruction, 9 probe prompts",
              "benign-lang_fidelity": "benign answer-language fidelity",
              "benign-persona_hold": "benign persona adherence",
              "benign-recon x lang_fidelity": "benign reconstruction $\\times$ language fidelity",
              "benign-recon x persona_hold": "benign reconstruction $\\times$ persona adherence",
              "benign-borderline-nonrefusal": "non-refusal on safety-adjacent harmless requests",
              "benign-recon x borderline-nonrefusal":
                  "benign reconstruction $\\times$ safety-adjacent non-refusal",
              "harmful-recon (leaks)": "harmful-run reconstruction (leaks; reference)"}
    for c in keep:
        if not all(c in out[d] and out[d][c]["full_coverage"] for d in out):
            continue
        cells = []
        for d in out:
            cells += [f"{out[d][c]['rho_mean']:+.2f}", f"{out[d][c]['selector_at_3']:.3f}"]
        L.append(f"{pretty.get(c, c)} & " + " & ".join(cells) + " \\\\")
    L.append("\\midrule")
    cells = []
    for d in out:
        cells += ["--", f"\\textbf{{{out[d]['__loo__']['selector_at_3']:.3f}}}"]
    L.append("chosen leave-one-target-out & " + " & ".join(cells) + " \\\\")
    cells = []
    for d in out:
        cells += ["--", f"{out[d]['__loo__']['oracle']:.3f}"]
    L.append("oracle & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    Path("paper/tab_sel_prior.tex").write_text("\n".join(L))
    print("wrote paper/tab_sel_prior.tex")


if __name__ == "__main__":
    raise SystemExit(main())
