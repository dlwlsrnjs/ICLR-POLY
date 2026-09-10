#!/usr/bin/env python3
"""Compact per-MODEL body tables (held-in and held-out, kept separate). One row per model; MultiJail
and Lingua-SafetyBench as side-by-side column groups showing best-fixed / OURS(sel-adapt) / oracle.
Mean is the bottom row. Reuses the exact replay from heldout_selector.py. Offline; run as jinkwon.
Writes paper/tab_heldin.tex, paper/tab_heldout.tex, paper/heldin_numbers.tex."""
import io, contextlib, sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, "scripts")
with contextlib.redirect_stdout(io.StringIO()):
    import mj_bandit_full as MJ, lingua_bandit_full as LG
from heldout_selector import panel_refs, panel_candidate, selector_value
from benign_prior import SELECTION

DATASETS = (("MultiJail", MJ, 64), ("Lingua-SafetyBench", LG, 40))
HELDIN = ["qwen25_3b","qwen25_7b","qwen25_14b","qwen25_32b","llama32_3b_it","llama31_8b_it",
          "gemma2_2b_it","gemma2_9b_it","gemma2_27b"]
HELDOUT = ["phi35_mini","mistral7b","falcon3_7b","glm4_9b","mistral24b","olmo2_7b","zephyr7b"]
PRETTY = {"qwen25_3b":"Qwen2.5-3B","qwen25_7b":"Qwen2.5-7B","qwen25_14b":"Qwen2.5-14B",
    "qwen25_32b":"Qwen2.5-32B","llama32_3b_it":"Llama-3.2-3B","llama31_8b_it":"Llama-3.1-8B",
    "gemma2_2b_it":"Gemma-2-2B","gemma2_9b_it":"Gemma-2-9B","gemma2_27b":"Gemma-2-27B",
    "phi35_mini":"Phi-3.5-mini","mistral7b":"Mistral-7B","falcon3_7b":"Falcon3-7B","glm4_9b":"GLM-4-9B",
    "mistral24b":"Mistral-Small-24B","olmo2_7b":"OLMo-2-7B","zephyr7b":"Zephyr-7B"}

def compute(models, heldin):
    """returns per-model {model: {ds: (fixed, ours, oracle)}}, means, and mean qbar per ds."""
    per = {m: {} for m in models}; means = {}; q = {}
    for collection, mod, n_items in DATASETS:
        fixed_i, pub_i, names = panel_refs(mod)
        if heldin:
            ppt = json.loads(SELECTION.read_text())[collection]["__loo__"]["picked_per_target"]
        else:
            cand = panel_candidate(collection)
        agg = {"fixed": [], "ours": [], "oracle": []}; qs = []
        for t in models:
            G = np.array([a[1] for a in mod.arms_for(t)])
            oracle = float(G.max()); fixed = float(G[fixed_i])
            _, ours, qa = selector_value(collection, mod, t, G, names, n_items,
                                         ppt[t] if heldin else cand)
            per[t][collection] = (fixed, ours, oracle)
            agg["fixed"].append(fixed); agg["ours"].append(ours); agg["oracle"].append(oracle); qs.append(qa)
        means[collection] = tuple(float(np.mean(agg[k])) for k in ("fixed","ours","oracle"))
        q[collection] = float(np.mean(qs))
    return per, means, q

def emit(models, per, means, q, label, cap, path):
    L = [r"\begin{table}[t]\centering\small", r"\setlength{\tabcolsep}{4.5pt}", cap, label,
         r"\begin{tabular}{l ccc ccc}", r"\toprule",
         r" & \multicolumn{3}{c}{MultiJail} & \multicolumn{3}{c}{Lingua-SafetyBench} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"Model & fixed & \textbf{ours} & oracle & fixed & \textbf{ours} & oracle \\", r"\midrule"]
    for t in models:
        mj = per[t]["MultiJail"]; lg = per[t]["Lingua-SafetyBench"]
        L.append(f"{PRETTY[t]} & {mj[0]:.3f} & \\textbf{{{mj[1]:.3f}}} & {mj[2]:.3f} & "
                 f"{lg[0]:.3f} & \\textbf{{{lg[1]:.3f}}} & {lg[2]:.3f} \\\\")
    mmj = means["MultiJail"]; mlg = means["Lingua-SafetyBench"]
    L += [r"\midrule",
          f"\\emph{{mean}} & {mmj[0]:.3f} & \\textbf{{{mmj[1]:.3f}}} & {mmj[2]:.3f} & "
          f"{mlg[0]:.3f} & \\textbf{{{mlg[1]:.3f}}} & {mlg[2]:.3f} \\\\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(path).write_text("\n".join(L) + "\n")
    print(f"{path}: MJ mean fixed/ours/oracle {mmj[0]:.3f}/{mmj[1]:.3f}/{mmj[2]:.3f}  q {q['MultiJail']:.2f} | "
          f"LG {mlg[0]:.3f}/{mlg[1]:.3f}/{mlg[2]:.3f}  q {q['Lingua-SafetyBench']:.2f}")
    return mmj, mlg, q

def main():
    hi_per, hi_m, hi_q = compute(HELDIN, True)
    ho_per, ho_m, ho_q = compute(HELDOUT, False)
    emit(HELDIN, hi_per, hi_m, hi_q, r"\label{tab:heldin}",
         r"\caption{\textbf{Per-model results on the nine held-in panel models} (primary; not averaged). "
         r"\emph{fixed} = single panel-best configuration applied unchanged; \textbf{ours} = "
         r"plaintext-warm-started selector ($\bar{q}\approx3$ harmful queries, no harmful supervision); "
         r"\emph{oracle} = best arm on that model. The winning attack varies sharply by model (Gemma falls "
         r"to a persona frame the fixed translation baseline misses); the selector recovers most of each "
         r"model's oracle. Bottom row = model-averaged mean.}",
         "paper/tab_heldin.tex")
    emit(HELDOUT, ho_per, ho_m, ho_q, r"\label{tab:heldout}",
         r"\caption{\textbf{Per-model results on seven held-out models} used in neither the arm space, the "
         r"probe selection, nor the winner's-curse calibration (six vendors, 3.8--24B; primary, not "
         r"averaged). Columns as in Table~\ref{tab:heldin}; the frozen selector is applied with no "
         r"re-tuning. Bottom row = model-averaged mean.}",
         "paper/tab_heldout.tex")
    # macros for the held-in means (cross-check vs headline)
    macros = [f"\\newcommand{{\\pjhiMJfixed}}{{{hi_m['MultiJail'][0]:.3f}}}",
              f"\\newcommand{{\\pjhiMJours}}{{{hi_m['MultiJail'][1]:.3f}}}",
              f"\\newcommand{{\\pjhiMJoracle}}{{{hi_m['MultiJail'][2]:.3f}}}",
              f"\\newcommand{{\\pjhiLGfixed}}{{{hi_m['Lingua-SafetyBench'][0]:.3f}}}",
              f"\\newcommand{{\\pjhiLGours}}{{{hi_m['Lingua-SafetyBench'][1]:.3f}}}",
              f"\\newcommand{{\\pjhiLGoracle}}{{{hi_m['Lingua-SafetyBench'][2]:.3f}}}"]
    Path("paper/heldin_numbers.tex").write_text("% per-model held-in means (scripts/permodel_body_tables.py)\n"
        + "\n".join(macros) + "\n")
    print("wrote tab_heldin.tex, tab_heldout.tex, heldin_numbers.tex")

if __name__ == "__main__":
    main()
