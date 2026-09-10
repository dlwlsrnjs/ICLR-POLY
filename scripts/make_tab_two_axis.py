#!/usr/bin/env python3
"""Emit the LaTeX tables and figure for the capability-alignment law."""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/ubuntu/342/jinkwon/poly/PolyJigsaw"
d = json.load(open(f"{ROOT}/results/paper_two_axis_law.json"))
C, comply, align, oracle = d["C"], d["comply"], d["alignment"], d["oracle"]
labels = json.load(open(f"{ROOT}/results/panel_labels.json"))
tag2label = labels
label2tag = {v: k for k, v in labels.items()}

# ---------- Table 1: the panel score card ----------
rows = sorted(d["panel"], key=lambda t: C[t])
lines = [r"\begin{table}[t]", r"\centering",
         r"\caption{The model panel. Capability $C$ comes from the benign FLORES probe and "
         r"costs no harmful queries. The alignment probes measure the refusal barrier: "
         r"$A_{\mathrm{direct}}$ is compliance on direct English harmful requests, "
         r"$A_{\mathrm{multi}}$ the same requests in four non-English languages, and the gap "
         r"between them is the English/multilingual alignment gap. $a$ and $b$ are the fitted "
         r"compliance intercept and slope, and the last column is the best gated ASR any grid "
         r"cell reaches on that target.}",
         r"\label{tab:panel_scorecard}", r"\small",
         r"\begin{tabular}{l c cc c cc c}\toprule",
         r"Target & $C$ & $A_{\mathrm{direct}}$ & $A_{\mathrm{multi}}$ & gap & $a$ & $b$ & oracle gated \\\midrule"]
for lab in rows:
    t = label2tag.get(lab, lab); e = align.get(t, {})
    a_, b_ = comply[lab]
    lines.append(f"{lab} & {C[lab]:.0f} & {e.get('A_direct', float('nan')):.2f} & "
                 f"{e.get('A_multi', float('nan')):.2f} & {e.get('gap', float('nan')):+.2f} & "
                 f"{a_:.2f} & {b_:+.4f} & {oracle[lab]:.3f} \\\\")
lines += [r"\bottomrule\end{tabular}\end{table}"]
open(f"{ROOT}/paper/tab_panel_scorecard.tex", "w").write("\n".join(lines) + "\n")

# ---------- Table 2: leave-one-out regret by predictor set ----------
loo = d["loo"]
order = sorted(loo, key=lambda k: loo[k]["mean_regret"])
lines = [r"\begin{table}[t]", r"\centering",
         r"\caption{Generating the attack difficulty from a probe, leave-one-out across the "
         r"panel. Each row fits the compliance coefficients on the other targets from the listed "
         r"predictors, generates the held-out target's difficulty in closed form, and reports the "
         r"gated ASR given up against that target's own oracle cell. Capability alone is not "
         r"enough; adding a measurement of the refusal barrier is what makes the difficulty "
         r"predictable.}",
         r"\label{tab:two_axis}", r"\small",
         r"\begin{tabular}{l cc}\toprule",
         r"Predictors & mean regret & worst regret \\\midrule"]
for name in order:
    v = loo[name]
    lines.append(f"{name.replace('_', chr(92) + '_')} & {v['mean_regret']:.3f} & {v['max_regret']:.3f} \\\\")
bf = d["best_fixed"]
lines.append(f"best single fixed setting ({bf['setting']}) & {bf['mean_regret']:.3f} & -- \\\\")
lines += [r"\bottomrule\end{tabular}\end{table}"]
open(f"{ROOT}/paper/tab_two_axis.tex", "w").write("\n".join(lines) + "\n")

# ---------- Figure: capability and alignment are separate axes ----------
tags = [label2tag.get(l, l) for l in d["panel"]]
cs = np.array([C[l] for l in d["panel"]])
a_s = np.array([comply[l][0] for l in d["panel"]])
am = np.array([align[t].get("A_multi", np.nan) if t in align else np.nan for t in tags])
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4))
def scat(axis, x, y, xl, yl, title):
    ok = ~(np.isnan(x) | np.isnan(y))
    axis.scatter(x[ok], y[ok], s=64, color="#b2182b", zorder=5)
    for xi, yi, l in zip(x, y, d["panel"]):
        if not (np.isnan(xi) or np.isnan(yi)):
            axis.annotate(l, (xi, yi), fontsize=7.5, xytext=(4, 4), textcoords="offset points")
    if ok.sum() > 2:
        z = np.polyfit(x[ok], y[ok], 1); xs = np.linspace(x[ok].min(), x[ok].max(), 50)
        r = np.corrcoef(x[ok], y[ok])[0, 1]
        axis.plot(xs, z[0] * xs + z[1], "--", color="#888", label=f"r={r:.2f}")
        axis.legend(fontsize=9)
    axis.set_xlabel(xl); axis.set_ylabel(yl); axis.set_title(title, fontsize=10.5); axis.grid(alpha=.25)
scat(ax[0], cs, a_s, "capability $C$ (benign probe)", "compliance intercept $a$",
     "Capability does not set compliance")
scat(ax[1], am, a_s, "$A_{\\mathrm{multi}}$ (multilingual barrier)", "compliance intercept $a$",
     "The refusal barrier does")
scat(ax[2], cs, am, "capability $C$", "$A_{\\mathrm{multi}}$", "The two axes are distinct")
plt.tight_layout()
plt.savefig(f"{ROOT}/paper/figures/fig_two_axis.pdf")
plt.savefig(f"{ROOT}/paper/figures/fig_two_axis.png", dpi=150)
print("wrote paper/tab_panel_scorecard.tex, paper/tab_two_axis.tex, paper/figures/fig_two_axis.{pdf,png}")
