#!/usr/bin/env python3
"""Per-MODEL held-in results for the nine panel models (not averaged): oracle, fixed(panel),
selector@3 and selector-adaptive verified ASR, plus mean adaptive queries. Reuses the exact replay
from heldout_selector.py (same GP-BAI, same benign-recon LOO probe recipe, same budget rule), just
iterated over the nine panel models instead of the held-out set. Offline; run as jinkwon.
Writes paper/tab_heldin.tex + paper/heldin_numbers.tex and prints the aggregate for cross-check
against the headline (MJ adaptive 0.664 / Lingua 0.726)."""
import io, contextlib, sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, "scripts")
with contextlib.redirect_stdout(io.StringIO()):
    import mj_bandit_full as MJ, lingua_bandit_full as LG
from heldout_selector import panel_refs, selector_value
from benign_prior import SELECTION

PRETTY = {"qwen25_3b":"Qwen2.5-3B","qwen25_7b":"Qwen2.5-7B","qwen25_14b":"Qwen2.5-14B",
          "qwen25_32b":"Qwen2.5-32B","llama32_3b_it":"Llama-3.2-3B","llama31_8b_it":"Llama-3.1-8B",
          "gemma2_2b_it":"Gemma-2-2B","gemma2_9b_it":"Gemma-2-9B","gemma2_27b":"Gemma-2-27B"}
ORDER = ["qwen25_3b","qwen25_7b","qwen25_14b","qwen25_32b","llama32_3b_it","llama31_8b_it",
         "gemma2_2b_it","gemma2_9b_it","gemma2_27b"]
DATASETS = (("MultiJail", MJ, 64), ("Lingua-SafetyBench", LG, 40))

def main():
    rows_tex = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\caption{\textbf{Per-model held-in jailbreak detail.} Verified ASR for each of the nine "
        r"panel models (not averaged): \emph{oracle} (best arm on that model), \emph{fixed} (the single "
        r"panel-best configuration applied unchanged), \emph{sel@3} (selector at a fixed three-query "
        r"budget) and \emph{sel-adapt} (difficulty-adaptive budget, floor 3 / cap 8 / stop 0.5) with "
        r"$\bar{q}$ its mean queries. The selector uses only the free benign-recon warm start "
        r"(leave-one-target-out); no harmful supervision.}",
        r"\label{tab:heldin}", r"\begin{tabular}{llccccc}", r"\toprule",
        r"Dataset & Panel model & oracle & fixed & sel@3 & sel-adapt & $\bar{q}$ \\", r"\midrule"]
    macros = []
    for collection, mod, n_items in DATASETS:
        fixed_i, pub_i, names = panel_refs(mod)
        ppt = json.loads(SELECTION.read_text())[collection]["__loo__"]["picked_per_target"]  # each target's OWN LOO pick (honest held-in selector)
        agg = {k: [] for k in ("oracle","fixed","sel3","adapt","q")}
        first = True
        tag = "MJ" if collection == "MultiJail" else "LG"
        for t in ORDER:
            arms = mod.arms_for(t); G = np.array([a[1] for a in arms])
            oracle = float(G.max()); fixed = float(G[fixed_i])
            sel3, seladapt, q = selector_value(collection, mod, t, G, names, n_items, ppt[t])
            agg["oracle"].append(oracle); agg["fixed"].append(fixed)
            agg["sel3"].append(sel3); agg["adapt"].append(seladapt); agg["q"].append(q)
            ds = collection if first else ""
            rows_tex.append(f"{ds} & {PRETTY[t]} & {oracle:.3f} & {fixed:.3f} & {sel3:.3f} & "
                            f"\\textbf{{{seladapt:.3f}}} & {q:.1f} \\\\")
            first = False
        m = {k: float(np.mean(v)) for k, v in agg.items()}
        rows_tex.append(r"\cmidrule(l){2-7}")
        rows_tex.append(f" & mean & {m['oracle']:.3f} & {m['fixed']:.3f} & {m['sel3']:.3f} & "
                        f"\\textbf{{{m['adapt']:.3f}}} & {m['q']:.1f} \\\\")
        rows_tex.append(r"\midrule" if collection == "MultiJail" else r"\bottomrule")
        for k, mk in (("oracle","oracle"),("fixed","fixed"),("sel3","selthree"),("adapt","adapt")):
            macros.append(f"\\newcommand{{\\pjhi{tag}{mk}}}{{{m[k]:.3f}}}")
        print(f"{collection}: oracle {m['oracle']:.3f} fixed {m['fixed']:.3f} sel@3 {m['sel3']:.3f} "
              f"sel-adapt {m['adapt']:.3f} q {m['q']:.2f}")
    rows_tex += [r"\end{tabular}", r"\end{table}"]
    Path("paper/tab_heldin.tex").write_text("\n".join(rows_tex) + "\n")
    Path("paper/heldin_numbers.tex").write_text("% per-model held-in aggregate (scripts/heldin_permodel.py)\n"
        + "\n".join(macros) + "\n")
    print("wrote paper/tab_heldin.tex + paper/heldin_numbers.tex")

if __name__ == "__main__":
    main()
