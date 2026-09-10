#!/usr/bin/env python3
"""Per-model technique comparison: MODELS as columns, TECHNIQUES as rows (verified ASR). Each attack
is an arm in the panel matrix; OURS is the plaintext-warm-started selector's per-model recommendation
(sel-adapt), oracle is the best arm. Held-in and held-out kept separate; MultiJail and Lingua blocks
stacked. Reuses heldout_selector's replay. Offline; run as jinkwon. Writes paper/tab_heldin.tex,
paper/tab_heldout.tex, paper/heldin_numbers.tex."""
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
ABBR = {"qwen25_3b":"Q3B","qwen25_7b":"Q7B","qwen25_14b":"Q14B","qwen25_32b":"Q32B",
        "llama32_3b_it":"L3B","llama31_8b_it":"L8B","gemma2_2b_it":"G2B","gemma2_9b_it":"G9B",
        "gemma2_27b":"G27B","phi35_mini":"Phi","mistral7b":"Mis7","falcon3_7b":"Fal7","glm4_9b":"GLM9",
        "mistral24b":"Mis24","olmo2_7b":"OLM7","zephyr7b":"Zep7"}
# technique arm rows (label, arm-name); ours + oracle handled specially
TECHS = [("low-resource translation","m_translated"), ("AIM persona","m_aim"),
         ("DeepInception","m_deepinception"), ("PAP persuasion","m_pap")]

def build(models, heldin, label, caption, path):
    cols = len(models)
    L = [r"\begin{table}[t]\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}", caption, label,
         r"\begin{tabular}{l" + "c"*cols + "c}", r"\toprule",
         "Attack (verified ASR) & " + " & ".join(ABBR[m] for m in models) + r" & mean \\"]
    hi_means = {}
    for collection, mod, n_items in DATASETS:
        fixed_i, pub_i, names = panel_refs(mod)
        idx = {n:i for i,n in enumerate(names)}
        if heldin:
            ppt = json.loads(SELECTION.read_text())[collection]["__loo__"]["picked_per_target"]
        else:
            cand = panel_candidate(collection)
        # per-model arm vectors
        G = {m: np.array([a[1] for a in mod.arms_for(m)]) for m in models}
        L.append(r"\midrule")
        L.append(rf"\multicolumn{{{cols+2}}}{{l}}{{\emph{{{collection}}}}} \\")
        # published-attack rows
        for lab, arm in TECHS:
            vals = [G[m][idx[arm]] for m in models]
            L.append(f"\\quad {lab} & " + " & ".join(f"{v:.2f}" for v in vals) +
                     f" & {np.mean(vals):.2f} \\\\")
        # ours (selector) row
        ours = []
        for m in models:
            _, sa, _ = selector_value(collection, mod, m, G[m], names, n_items,
                                      ppt[m] if heldin else cand)
            ours.append(sa)
        L.append(r"\quad \textbf{ours (selector)} & " + " & ".join(f"\\textbf{{{v:.2f}}}" for v in ours) +
                 f" & \\textbf{{{np.mean(ours):.2f}}} \\\\")
        # oracle row
        orac = [float(G[m].max()) for m in models]
        L.append(r"\quad oracle & " + " & ".join(f"{v:.2f}" for v in orac) +
                 f" & {np.mean(orac):.2f} \\\\")
        tag = "MJ" if collection == "MultiJail" else "LG"
        hi_means[tag] = (float(np.mean([G[m][fixed_i] for m in models])), float(np.mean(ours)), float(np.mean(orac)))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(path).write_text("\n".join(L) + "\n")
    print(path, {k: tuple(round(x,3) for x in v) for k,v in hi_means.items()})
    return hi_means

def main():
    hi = build(HELDIN, True, r"\label{tab:heldin}",
        r"\caption{\textbf{Per-model results on the nine held-in panel models} (primary; not averaged). "
        r"Verified ASR of each attack \emph{technique} (rows) on each model (columns; Q=Qwen2.5, L=Llama-3, "
        r"G=Gemma-2, with size); \textbf{ours} is the plaintext-warm-started selector's per-model pick "
        r"($\bar{q}\approx3$ harmful queries, no harmful supervision), \emph{oracle} the best arm. No single "
        r"fixed technique wins on every model---translation dominates Qwen/Llama, a persona frame dominates "
        r"Gemma---while the selector tracks each model's best. Last column is the model-averaged mean.}",
        "paper/tab_heldin.tex")
    build(HELDOUT, False, r"\label{tab:heldout}",
        r"\caption{\textbf{Per-model results on seven held-out models} (six vendors, 3.8--24B; primary, not "
        r"averaged), used in neither the arm space, probe selection, nor winner's-curse calibration. Rows/"
        r"columns as in Table~\ref{tab:heldin} (Phi-3.5-mini, Mistral-7B, Falcon3-7B, GLM-4-9B, "
        r"Mistral-Small-24B, OLMo-2-7B, Zephyr-7B); the frozen selector is applied with no re-tuning.}",
        "paper/tab_heldout.tex")
    macros = [f"\\newcommand{{\\pjhi{t}{k}}}{{{v:.3f}}}"
              for t in ("MJ","LG") for k,v in zip(("fixed","ours","oracle"), hi[t])]
    Path("paper/heldin_numbers.tex").write_text("% per-model held-in means (permodel_technique_tables.py)\n"
        + "\n".join(macros) + "\n")
    print("wrote tab_heldin.tex, tab_heldout.tex, heldin_numbers.tex")

if __name__ == "__main__":
    main()
