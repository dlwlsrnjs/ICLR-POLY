#!/usr/bin/env python3
"""Turn the arm-space ablation JSON (results/space_ab_compare.json) into a PAPER-READY LaTeX table
+ macro file, in the repo's booktabs/macro convention. Rows = spaces A/B/C/D; columns = held-in and
held-out means of oracle / benign-only / adaptive@8. C is emphasised as the operating point.
Writes paper/tab_arm_space_ablation.tex and paper/arm_space_numbers.tex."""
import json, sys
from pathlib import Path
import numpy as np

ROOT = sys.argv[sys.argv.index("--root")+1] if "--root" in sys.argv else "experiments_suite/exp02_panel_collect/results"
rows = json.loads((Path(ROOT)/"space_ab_compare.json").read_text())
SP = [("A_comp_lang_order","A: comp$\\times$lang$\\times$order"),
      ("B_weak_will","B: $+$weak willingness"),
      ("C_medium_will","\\textbf{C: $+$medium willingness}"),
      ("D_full_stacks","D: $+$full stacks")]
def agg(held, space, metric):
    vals=[r[space][metric] for r in rows if r["held"]==held and space in r]
    return np.mean(vals) if vals else None
def n(held): return len({r["tag"][:-3] for r in rows if r["held"]==held})

macros=[]
def mac(name,val): macros.append(f"\\newcommand{{\\{name}}}{{{'--' if val is None else f'{val:.3f}'}}}")

lines=[r"\begin{table}[t]\centering\small",
 r"\caption{\textbf{Arm-space ablation.} Verified ASR (model-averaged) as the configuration space grows "
 r"from comprehension$\times$language$\times$order (A) by adding willingness. \emph{oracle} = best arm; "
 r"\emph{free} = benign-prior pick (no harmful target query); \emph{adapt} = C-space selector "
 r"(\texttt{ucb\_add\_pibo}) at 8 harmful pulls. Willingness helps up to C; beyond it (D) achievable "
 r"ASR is unchanged while the free prior degrades and cost doubles, so C is the operating point.}",
 r"\label{tab:armspace}",
 r"\begin{tabular}{l ccc ccc}",
 r"\toprule",
 r"& \multicolumn{3}{c}{Held-in} & \multicolumn{3}{c}{Held-out} \\",
 r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
 r"Space (\#arms) & oracle & free & adapt & oracle & free & adapt \\",
 r"\midrule"]
NARMS={"A_comp_lang_order":32,"B_weak_will":96,"C_medium_will":160,"D_full_stacks":288}
for key,label in SP:
    cells=[]
    for held in ("held_in","held_out"):
        for metric,mk in (("oracle","or"),("benign_only","fr"),("adaptive8","ad")):
            v=agg(held,key,metric); cells.append('--' if v is None else f'{v:.3f}')
            mac(f"pjas{key.split('_')[0]}{held[5:7]}{mk}", v)
    lines.append(f"{label} ({NARMS[key]}) & "+" & ".join(cells)+r" \\")
lines+=[r"\bottomrule",r"\end{tabular}\end{table}"]
Path("paper/tab_arm_space_ablation.tex").write_text("\n".join(lines)+"\n")
Path("paper/arm_space_numbers.tex").write_text("\n".join(macros)+"\n")
print("wrote paper/tab_arm_space_ablation.tex + paper/arm_space_numbers.tex")
print(f"(held-in n={n('held_in')}, held-out n={n('held_out')} models so far)")
print("\n=== tab_arm_space_ablation.tex ===")
print(Path("paper/tab_arm_space_ablation.tex").read_text())
