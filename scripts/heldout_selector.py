#!/usr/bin/env python3
"""Held-out generalization of the benign-warm-started selector.

The nine panel models built the configuration space, the warm-start prior's leave-one-target-out
probe choice, and the winner's-curse calibration. These seven models touched none of that:

  phi35_mini    microsoft/Phi-3.5-mini-instruct        (Microsoft, 3.8B)
  mistral7b     mistralai/Mistral-7B-Instruct-v0.3     (Mistral, 7B)
  falcon3_7b    tiiuae/Falcon3-7B-Instruct             (TII, 7B)
  glm4_9b       THUDM/glm-4-9b-chat-hf                 (Zhipu, 9B)
  mistral24b    mistralai/Mistral-Small-24B-Instruct   (Mistral, 24B)

For each held-out target we build the same arm profile and run the same selector. The prior is the
harmless probe on that target only; the candidate is NOT re-selected on the held-out model (it is not
in the selection file, so `benign_prior.chosen_candidate` returns the default probe), so nothing about
the selector is tuned to these models. Two references are frozen on the PANEL, not the held-out set:

  fixed(panel)      the single configuration with the highest mean verified ASR across the nine panel
                    models, applied unchanged to each held-out model. This is 'pick one attack on your
                    development models and ship it'.
  published(panel)  the published attack (translated / AIM / DeepInception) with the highest mean
                    across the panel, applied to each held-out model.

We report, per held-out model and in aggregate: oracle (best arm on that model), fixed(panel),
published(panel), and the selector at a 3-query budget. The claim is that the selector transfers:
its verified ASR tracks each unseen model's oracle and beats the attack you would have shipped.

Offline: reads existing results only, no GPU. Writes results/heldout_selector_20260908.json,
paper/tab_heldout.tex, paper/heldout_numbers.tex.
"""
from __future__ import annotations
import io, json, contextlib, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
with contextlib.redirect_stdout(io.StringIO()):   # importing runs their headline print
    import mj_bandit_full as MJ
    import lingua_bandit_full as LG
from gp_bai import gp_bai, gp_bai_adaptive
from benign_prior import prior_for, candidates, SELECTION
from collections import Counter

# difficulty-adaptive budget: never fewer than the fixed baseline, up to B_MAX extra queries for a
# target that has not yet yielded an arm observed at >= STOP_TAU. Fixed a priori (not tuned per model).
B_FLOOR, B_MAX, STOP_TAU = 3, 8, 0.5

HELDOUT = ["phi35_mini", "mistral7b", "falcon3_7b", "glm4_9b", "mistral24b",
           "olmo2_7b", "zephyr7b"]
PRETTY = {"phi35_mini": "Phi-3.5-mini (3.8B)", "mistral7b": "Mistral-7B",
          "falcon3_7b": "Falcon3-7B", "glm4_9b": "GLM-4-9B", "mistral24b": "Mistral-Small-24B",
          "olmo2_7b": "OLMo-2-7B",
          "zephyr7b": "Zephyr-7B"}
VENDOR = {"phi35_mini": "Microsoft", "mistral7b": "Mistral", "falcon3_7b": "TII",
          "glm4_9b": "Zhipu", "mistral24b": "Mistral",
          "olmo2_7b": "AllenAI", "zephyr7b": "HuggingFace"}
PUBLISHED = ("m_translated", "m_aim", "m_deepinception")
DATASETS = (("MultiJail", MJ, 64), ("Lingua-SafetyBench", LG, 40))
REPS = 2000
BUDGET = 3
rng = np.random.default_rng(0)


def _panel(mod):
    return getattr(mod, "STRONG", None) or mod.MODELS


def panel_refs(mod):
    """fixed(panel) arm index and published(panel) arm index, frozen on the nine panel models."""
    G = np.array([[a[1] for a in mod.arms_for(t)] for t in _panel(mod)])   # (9, n_arms)
    names = mod.names
    fixed_i = int(np.argmax(G.mean(0)))
    pub_idx = [names.index(p) for p in PUBLISHED if p in names]
    pub_i = pub_idx[int(np.argmax(G.mean(0)[pub_idx]))]
    return fixed_i, pub_i, names


def panel_candidate(collection):
    """The prior candidate the nine-panel leave-one-target-out selection settled on (its modal pick).
    We apply THIS to every held-out model, so the held-out prior uses the same recipe as the main
    experiment rather than being re-tuned per unseen model."""
    sel = json.loads(SELECTION.read_text())
    picks = sel[collection]["__loo__"]["picked_per_target"].values()
    return Counter(picks).most_common(1)[0][0]


def selector_value(collection, mod, t, G_h, names, n_items, cand_name):
    """Mean verified ASR of the arm the selector recommends, over REPS noisy probe realisations.
    The prior is benign_prior's `cand_name` recipe computed on THIS target's harmless probes only."""
    FE = mod.FE
    cand = candidates(collection, t, names)
    if cand_name not in cand:
        raise FileNotFoundError(f"held-out probe for {cand_name} missing on {collection}/{t}")
    pr = cand[cand_name] * 0.6
    fixed_vals, adapt_vals, adapt_q = [], [], []
    for _ in range(REPS):
        probe = lambda a: float(np.clip(G_h[a] + rng.normal(
            0, np.sqrt(max(G_h[a] * (1 - G_h[a]), .01) / n_items)), 0, 1))
        rec, _ = gp_bai(pr, FE, probe, BUDGET)
        fixed_vals.append(G_h[rec])
        reca, obs = gp_bai_adaptive(pr, FE, probe, B_FLOOR, B_MAX, STOP_TAU)
        adapt_vals.append(G_h[reca]); adapt_q.append(len(obs))
    return float(np.mean(fixed_vals)), float(np.mean(adapt_vals)), float(np.mean(adapt_q))


def main():
    out = {}
    macros = []
    for collection, mod, n_items in DATASETS:
        fixed_i, pub_i, names = panel_refs(mod)
        cand_name = panel_candidate(collection)   # frozen probe recipe from the 9-panel selection
        rows, agg = {}, {"oracle": [], "fixed": [], "published": [], "selector": []}
        missing = []
        for t in HELDOUT:
            try:
                arms = mod.arms_for(t)
            except (FileNotFoundError, KeyError) as e:
                missing.append(t); continue
            G_h = np.array([a[1] for a in arms])
            names_t = [a[0] for a in arms]
            assert names_t == names, f"arm mismatch {collection}/{t}"
            oracle = float(G_h.max())
            fixed = float(G_h[fixed_i])
            published = float(G_h[pub_i])
            try:
                sel, sel_a, q_a = selector_value(collection, mod, t, G_h, names, n_items, cand_name)
            except FileNotFoundError:
                missing.append(t + "(probe)"); continue
            rows[t] = {"oracle": round(oracle, 3), "fixed_panel": round(fixed, 3),
                       "published_panel": round(published, 3), "selector3": round(sel, 3),
                       "selector_adapt": round(sel_a, 3), "adapt_queries": round(q_a, 2),
                       "best_arm": names[int(np.argmax(G_h))],
                       "sel_minus_fixed": round(sel - fixed, 3),
                       "adapt_minus_fixed": round(sel_a - fixed, 3),
                       "sel_minus_pub": round(sel - published, 3),
                       "oracle_gap": round(oracle - sel, 3)}
            agg["oracle"].append(oracle); agg["fixed"].append(fixed)
            agg["published"].append(published); agg["selector"].append(sel)
            agg.setdefault("selector_adapt", []).append(sel_a)
            agg.setdefault("adapt_queries", []).append(q_a)
        A = {k: float(np.mean(v)) for k, v in agg.items()}
        # paired bootstrap over the held-out models for selector - fixed and selector - published
        sv = np.array(agg["selector"]); fv = np.array(agg["fixed"]); pv = np.array(agg["published"])
        def boot(delta):
            bs = [np.mean(delta[rng.integers(0, len(delta), len(delta))]) for _ in range(10000)]
            return round(float(np.mean(delta)), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)
        av = np.array(agg.get("selector_adapt", []))
        d_fixed = boot(sv - fv); d_pub = boot(sv - pv)
        d_adapt = boot(av - fv) if len(av) else (0.0, 0.0, 0.0)
        d_adapt_pub = boot(av - pv) if len(av) else (0.0, 0.0, 0.0)
        out[collection] = {"fixed_panel_arm": names[fixed_i], "published_panel_arm": names[pub_i],
                           "prior_candidate": cand_name,
                           "per_model": rows, "aggregate": {k: round(v, 3) for k, v in A.items()},
                           "selector_minus_fixed": d_fixed, "selector_minus_published": d_pub,
                           "adapt_minus_fixed": d_adapt, "adapt_minus_published": d_adapt_pub,
                           "wins_vs_fixed": int(sum(1 for t in rows if rows[t]["sel_minus_fixed"] > 0)),
                           "adapt_wins_vs_fixed": int(sum(1 for t in rows if rows[t]["adapt_minus_fixed"] > 0)),
                           "adapt_wins_vs_pub": int(sum(1 for t in rows if rows[t]["selector_adapt"] - rows[t]["published_panel"] > 0)),
                           "wins_vs_pub": int(sum(1 for t in rows if rows[t]["sel_minus_pub"] > 0)),
                           "n": len(rows), "missing": missing}
        key = "MJ" if collection == "MultiJail" else "LG"
        macros += [f"\\newcommand{{\\pjho{key}oracle}}{{{A['oracle']:.3f}}}",
                   f"\\newcommand{{\\pjho{key}fixed}}{{{A['fixed']:.3f}}}",
                   f"\\newcommand{{\\pjho{key}pub}}{{{A['published']:.3f}}}",
                   f"\\newcommand{{\\pjho{key}sel}}{{{A['selector']:.3f}}}",
                   f"\\newcommand{{\\pjho{key}adapt}}{{{A.get('selector_adapt', 0):.3f}}}",
                   f"\\newcommand{{\\pjho{key}adaptq}}{{{A.get('adapt_queries', 0):.2f}}}",
                   f"\\newcommand{{\\pjho{key}dfixed}}{{{d_fixed[0]:+.3f}}}",
                   f"\\newcommand{{\\pjho{key}dfixedlo}}{{{d_fixed[1]:+.3f}}}",
                   f"\\newcommand{{\\pjho{key}dfixedhi}}{{{d_fixed[2]:+.3f}}}",
                   f"\\newcommand{{\\pjho{key}dadapt}}{{{d_adapt[0]:+.3f}}}",
                   f"\\newcommand{{\\pjho{key}dadaptlo}}{{{d_adapt[1]:+.3f}}}",
                   f"\\newcommand{{\\pjho{key}dadapthi}}{{{d_adapt[2]:+.3f}}}",
                   f"\\newcommand{{\\pjho{key}dpub}}{{{d_adapt_pub[0]:+.3f}}}",
                   f"\\newcommand{{\\pjho{key}dpublo}}{{{d_adapt_pub[1]:+.3f}}}",
                   f"\\newcommand{{\\pjho{key}dpubhi}}{{{d_adapt_pub[2]:+.3f}}}",
                   f"\\newcommand{{\\pjho{key}winsfixed}}{{{out[collection]['wins_vs_fixed']}}}",
                   f"\\newcommand{{\\pjho{key}adaptwins}}{{{out[collection]['adapt_wins_vs_fixed']}}}",
                   f"\\newcommand{{\\pjho{key}n}}{{{len(rows)}}}"]

    Path("results/heldout_selector_20260908.json").write_text(json.dumps(out, indent=2))
    Path("paper/heldout_numbers.tex").write_text(
        "% auto-generated by scripts/heldout_selector.py -- do not edit\n" + "\n".join(macros) + "\n")

    # LaTeX table: per held-out model, both datasets stacked
    L = ["\\begin{table}[t]", "\\centering", "\\small",
         "\\caption{Held-out generalization. Seven models used in neither the arm space, the warm-start "
         "probe selection, nor the winner's-curse calibration, spanning six vendors and 3.8--24B. "
         "\\emph{fixed} is the single configuration with the highest mean verified ASR on the nine panel "
         "models, applied unchanged; \\emph{published} is the strongest published attack "
         "(translation / AIM / DeepInception) chosen the same way. \\emph{sel@3} is the selector at the "
         "fixed three-query budget; \\emph{sel-adapt} adds the difficulty-adaptive budget (floor 3, cap "
         "8, stop once an arm is observed at $\\geq 0.5$), with $\\bar{q}$ the mean queries it spends. "
         "Nothing is re-tuned on these models: the arm space, the probe recipe, the GP, and the budget "
         "rule are all frozen on the panel.}",
         "\\label{tab:heldout}", "\\begin{tabular}{llcccccc}", "\\toprule",
         "Dataset & Held-out model & oracle & fixed & pub. & sel@3 & sel-adapt & $\\bar{q}$ \\\\",
         "\\midrule"]
    for collection, mod, _ in DATASETS:
        o = out[collection]; rows = o["per_model"]
        first = True
        for t in HELDOUT:
            if t not in rows:
                continue
            r = rows[t]
            ds = collection if first else ""
            first = False
            L.append(f"{ds} & {PRETTY[t]} & {r['oracle']:.3f} & {r['fixed_panel']:.3f} & "
                     f"{r['published_panel']:.3f} & {r['selector3']:.3f} & "
                     f"\\textbf{{{r['selector_adapt']:.3f}}} & {r['adapt_queries']:.1f} \\\\")
        A = o["aggregate"]; df = o["selector_minus_fixed"]; da = o["adapt_minus_fixed"]
        L.append(f"\\cmidrule(l){{2-8}}")
        L.append(f" & mean & {A['oracle']:.3f} & {A['fixed']:.3f} & {A['published']:.3f} & "
                 f"{A['selector']:.3f} & \\textbf{{{A.get('selector_adapt', 0):.3f}}} & "
                 f"{A.get('adapt_queries', 0):.1f} \\\\")
        L.append(f" & \\multicolumn{{7}}{{l}}{{\\footnotesize sel-adapt$-$fixed $= {da[0]:+.3f}$ "
                 f"[{da[1]:+.3f}, {da[2]:+.3f}], wins {o['adapt_wins_vs_fixed']}/{o['n']}}} \\\\")
        L.append("\\midrule")
    L[-1] = "\\bottomrule"
    L += ["\\end{tabular}", "\\end{table}", ""]
    Path("paper/tab_heldout.tex").write_text("\n".join(L))

    for collection in out:
        o = out[collection]; A = o["aggregate"]
        print(f"\n=== {collection} (held-out {o['n']}/{len(HELDOUT)}) ===")
        print(f"  fixed(panel)={o['fixed_panel_arm']}  published(panel)={o['published_panel_arm']}  prior={o['prior_candidate']}")
        print(f"  {'model':22s}{'oracle':>8}{'fixed':>8}{'pub':>8}{'sel@3':>8}{'adapt':>8}{'q̄':>6}{'ad-fix':>8}")
        for t in HELDOUT:
            if t not in o["per_model"]:
                print(f"  {PRETTY[t]:22s}  (missing: no results yet)"); continue
            r = o["per_model"][t]
            print(f"  {PRETTY[t]:22s}{r['oracle']:8.3f}{r['fixed_panel']:8.3f}{r['published_panel']:8.3f}"
                  f"{r['selector3']:8.3f}{r['selector_adapt']:8.3f}{r['adapt_queries']:6.1f}{r['adapt_minus_fixed']:+8.3f}")
        print(f"  {'MEAN':22s}{A['oracle']:8.3f}{A['fixed']:8.3f}{A['published']:8.3f}{A['selector']:8.3f}"
              f"{A.get('selector_adapt',0):8.3f}{A.get('adapt_queries',0):6.1f}")
        print(f"  selector@3-fixed {o['selector_minus_fixed']}  wins {o['wins_vs_fixed']}/{o['n']}")
        print(f"  adaptive-fixed   {o['adapt_minus_fixed']}  wins {o['adapt_wins_vs_fixed']}/{o['n']}")
        print(f"  adaptive-published {o['adapt_minus_published']}  wins {o['adapt_wins_vs_pub']}/{o['n']}")
        if o["missing"]:
            print(f"  MISSING: {o['missing']}")
    print("\nwrote paper/tab_heldout.tex, paper/heldout_numbers.tex")


if __name__ == "__main__":
    raise SystemExit(main())
