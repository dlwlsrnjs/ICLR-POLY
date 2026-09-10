#!/usr/bin/env python3
"""Intro teaser (single column): across models, both multilingual comprehension (how well the target
reconstructs the puzzle) and the reachable jailbreak level (oracle verified ASR) vary, and the winning
lever is consistent within a model family. One point per panel model, coloured by family, sized by
parameters, annotated with the lever it falls to. Saves paper/figures/fig_intro_teaser.{pdf,png}."""
from pathlib import Path
import io, contextlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
with contextlib.redirect_stdout(io.StringIO()):
    import mj_bandit_full as MJ
    import lingua_bandit_full as LG

OUT = Path("paper/figures"); OUT.mkdir(parents=True, exist_ok=True)
FAM = {"qwen25_3b": "Qwen", "qwen25_7b": "Qwen", "qwen25_14b": "Qwen", "qwen25_32b": "Qwen",
       "llama32_3b_it": "Llama", "llama31_8b_it": "Llama",
       "gemma2_2b_it": "Gemma", "gemma2_9b_it": "Gemma", "gemma2_27b": "Gemma"}
SIZE = {"qwen25_3b": 3, "qwen25_7b": 7, "qwen25_14b": 14, "qwen25_32b": 32, "llama32_3b_it": 3,
        "llama31_8b_it": 8, "gemma2_2b_it": 2, "gemma2_9b_it": 9, "gemma2_27b": 27}
FCOL = {"Qwen": "#1b6ca8", "Llama": "#ef8a3f", "Gemma": "#2e8b57"}
PRETTY = {"qwen25_3b": "Qwen-3B", "qwen25_7b": "Qwen-7B", "qwen25_14b": "Qwen-14B", "qwen25_32b": "Qwen-32B",
          "llama32_3b_it": "Llama-3.2-3B", "llama31_8b_it": "Llama-3.1-8B",
          "gemma2_2b_it": "Gemma-2B", "gemma2_9b_it": "Gemma-9B", "gemma2_27b": "Gemma-27B"}


def lever(nm):
    if nm.startswith("amt") or nm.startswith("dis"):
        return "amount"
    if nm.startswith("tri"):
        return "role sep."
    if nm == "m_aim" or "persona" in nm:
        return "persona"
    if nm == "m_deepinception" or "incept" in nm:
        return "fiction"
    if nm == "m_translated":
        return "translation"
    if nm == "m_pap":
        return "persuasion"
    return "combination"


def collect():
    panel = getattr(MJ, "STRONG", None) or MJ.MODELS
    d = {}
    for t in panel:
        gm = np.array([a[1] for a in MJ.arms_for(t)]); rm = np.array([a[2] for a in MJ.arms_for(t)])
        gl = np.array([a[1] for a in LG.arms_for(t)]); rl = np.array([a[2] for a in LG.arms_for(t)])
        comp = float((rm.mean() + rl.mean()) / 2)
        orc = float((gm.max() + gl.max()) / 2)
        win = lever(MJ.names[int(np.argmax(gm))])
        d[t] = dict(comp=comp, oracle=orc, win=win, fam=FAM[t], sz=SIZE[t])
    return d


def main():
    d = collect()
    for t, v in d.items():
        print(f"{t:15s} {v['fam']:5s} comp={v['comp']:.3f} oracle={v['oracle']:.3f} win={v['win']}")
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    for t, v in d.items():
        ax.scatter(v["comp"], v["oracle"], s=40 + 15 * v["sz"], color=FCOL[v["fam"]],
                   alpha=0.85, edgecolor="white", lw=1.2, zorder=3)
        ax.annotate(f"{PRETTY[t]}\n$\\rightarrow$ {v['win']}", (v["comp"], v["oracle"]),
                    textcoords="offset points", xytext=(0, 10 + 0.2 * v["sz"]), ha="center",
                    fontsize=6.0, color="#333")
    # family legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", color=FCOL[f], ms=8, label=f) for f in FCOL]
    ax.legend(handles=handles, fontsize=7.5, loc="lower right", title="model family", title_fontsize=7.5)
    ax.set_xlabel("multilingual comprehension  (mean puzzle reconstruction)", fontsize=8.5)
    ax.set_ylabel("reachable jailbreak\n(oracle verified ASR)", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.grid(alpha=0.25)
    ax.margins(0.16)
    fig.tight_layout()
    fig.savefig(OUT / "fig_intro_teaser.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_intro_teaser.png", dpi=200, bbox_inches="tight")
    print("wrote", OUT / "fig_intro_teaser.pdf")


if __name__ == "__main__":
    raise SystemExit(main())
