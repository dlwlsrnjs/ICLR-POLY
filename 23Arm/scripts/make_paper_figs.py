#!/usr/bin/env python3
"""Publication figures + two offline experiments (arm-space ablation, recon-gated correlation).
Reuses the arm/gated matrices from mj_bandit_full and lingua_bandit_full. No GPU."""
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "scripts")

OUT = Path("results/figs"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130})

# quiet import of the two bandit modules (they print on import)
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import mj_bandit_full as MJ
    import lingua_bandit_full as LG

DS = [("MultiJail", MJ), ("Lingua-SafetyBench", LG)]
SHORT = {"qwen25_3b": "Qwen-3B", "qwen25_7b": "Qwen-7B", "qwen25_14b": "Qwen-14B",
         "llama32_3b_it": "Llama-3B", "llama31_8b_it": "Llama-8B",
         "gemma2_2b_it": "Gemma-2B", "gemma2_9b_it": "Gemma-9B",
         "qwen25_32b": "Qwen-32B", "gemma2_27b": "Gemma-27B"}
FAMS = ["amount", "disorder", "combo", "triple", "single", "translated"]


def fam(n):
    if n == "m_translated":
        return "translated"
    for f, p in [("amount", "amt_"), ("disorder", "dis_"), ("combo", "combo_"), ("triple", "tri_"), ("single", "m_")]:
        if n.startswith(p):
            return f
    return "?"


def models(mod):
    return mod.STRONG if hasattr(mod, "STRONG") else mod.MODELS


# ---------- Fig 1: budget-accuracy curve ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, (name, mod) in zip(axes, DS):
    b = json.load(open(f"results/{'mj' if mod is MJ else 'lingua'}_bandit_full_20260906.json"))
    Bs = [1, 2, 3, 4, 6]
    br = [b["budget"][str(x)]["benign-recon"] for x in Bs]
    fl = [b["budget"][str(x)]["flat"] for x in Bs]
    ax.plot(Bs, br, "o-", lw=2.2, color="#1f77b4", label="benign-recon warm-start")
    ax.plot(Bs, fl, "s--", lw=1.6, color="#888", label="flat (no benign prior)")
    ax.axhline(b["oracle"], color="#2ca02c", ls=":", lw=1.8, label=f"oracle {b['oracle']:.3f}")
    ax.axhline(b["fixed"], color="#d62728", ls="-.", lw=1.5, label=f"best fixed arm {b['fixed']:.3f}")
    ax.set_title(name); ax.set_xlabel("query budget"); ax.set_ylabel("gated ASR"); ax.set_ylim(0.28, 0.78)
    ax.legend(fontsize=8, loc="lower right")
fig.suptitle("Budget–accuracy: benign warm-start GP-BAI reaches near-oracle", y=1.02, fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "fig1_budget_curve.png", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 2: per-model arm-family heatmap (best gated per family), MJ ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
for ax, (name, mod) in zip(axes, DS):
    M = models(mod); idxf = {f: [i for i, n in enumerate(mod.names) if fam(n) == f] for f in FAMS}
    H = np.array([[max(mod.G[t][i] for i in idxf[f]) if idxf[f] else np.nan for f in FAMS] for t in M])
    im = ax.imshow(H, cmap="YlOrRd", vmin=0, vmax=0.95, aspect="auto")
    ax.set_xticks(range(len(FAMS))); ax.set_xticklabels(FAMS, rotation=30, ha="right")
    ax.set_yticks(range(len(M))); ax.set_yticklabels([SHORT[t] for t in M])
    for i in range(len(M)):
        jbest = int(np.nanargmax(H[i]))
        for j in range(len(FAMS)):
            v = H[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.55 else "black",
                        fontweight="bold" if j == jbest else "normal")
        ax.add_patch(plt.Rectangle((jbest - .5, i - .5), 1, 1, fill=False, ec="#1f77b4", lw=2.5))
    ax.set_title(f"{name}: best gated per arm-family (box = winner)")
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout(); fig.savefig(OUT / "fig2_arm_heatmap.png", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 3: gate cost (raw unsafe vs recon vs gated) for each model's best arm ----------
def best_arm_stats(mod):
    M = models(mod); b = json.load(open(f"results/{'mj' if mod is MJ else 'lingua'}_bandit_full_20260906.json"))
    out = {}
    for t in M:
        i = int(np.argmax(mod.G[t]))
        out[t] = dict(gated=float(mod.G[t][i]), recon=float(mod.RC[t][i]))
    return out, b


fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, (name, mod) in zip(axes, DS):
    M = models(mod)
    # need raw unsafe per best arm: recompute from source files via the same loaders in the module
    arms = {t: mod.names[int(np.argmax(mod.G[t]))] for t in M}
    # pull unsafe from the per-family source using arms_for (has (name,gated,recon,feat)) -> lacks unsafe;
    # read unsafe directly from result files by arm name
    def unsafe_of(t, arm):
        pre = "mj" if mod is MJ else None
        if arm.startswith("amt_n"):
            f = f"results/{'mj_sequential_20260906' if mod is MJ else 'sequential_resource_20260906'}/{t}.json"
            n = int(arm[5:]); return {s['n']: s for s in json.load(open(f))['steps']}[n]['unsafe']
        if arm.startswith("dis_"):
            f = f"results/{'mj_disorder_20260906' if mod is MJ else 'disorder_sweep_20260906'}/{t}.json"
            d = json.load(open(f)); dv = float(arm[4:])
            return {r['delta']: r for r in (d.get('rows') or d.get('steps'))}[dv]['unsafe']
        if arm.startswith("combo_"):
            f = f"results/{'mj_combo_20260906' if mod is MJ else 'combo_20260906'}/{t}.json"
            return json.load(open(f))['variants'][arm[6:]]['unsafe']
        if arm.startswith("tri_"):
            return json.load(open(f"results/mj_triple_20260906/{t}.json"))['variants'][arm[4:]]['unsafe']
        if arm.startswith("m_"):
            f = f"results/{'mj_method_20260906' if mod is MJ else 'method_baselines_v2_20260906'}/{t}.json"
            return json.load(open(f))['methods'][arm[2:]]['unsafe']
    raw = [unsafe_of(t, arms[t]) for t in M]
    rec = [float(mod.RC[t][int(np.argmax(mod.G[t]))]) for t in M]
    gat = [float(mod.G[t].max()) for t in M]
    x = np.arange(len(M)); w = 0.27
    ax.bar(x - w, raw, w, label="raw unsafe", color="#ff7f0e")
    ax.bar(x, rec, w, label="recon", color="#9467bd")
    ax.bar(x + w, gat, w, label="gated (reported)", color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[t] for t in M], rotation=30, ha="right")
    ax.set_ylabel("rate"); ax.set_ylim(0, 1.02); ax.set_title(f"{name}: gate cost (gated = recon AND unsafe)")
    ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig3_gate_cost.png", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 4 + experiment: arm-space ablation (cumulative oracle) ----------
abl = {}
fig, ax = plt.subplots(figsize=(7.5, 4.4))
for name, mod in DS:
    M = models(mod); idxf = {f: [i for i, n in enumerate(mod.names) if fam(n) == f] for f in FAMS}
    cum = []
    for k in range(1, len(FAMS) + 1):
        use = sum([idxf[f] for f in FAMS[:k]], [])
        cum.append(float(np.mean([max(mod.G[t][i] for i in use) for t in M])))
    abl[name] = cum
    ax.plot(range(1, len(FAMS) + 1), cum, "o-", lw=2.2, label=name)
ax.set_xticks(range(1, len(FAMS) + 1))
ax.set_xticklabels(["+".join(FAMS[:k]) for k in range(1, len(FAMS) + 1)], rotation=18, ha="right", fontsize=8)
ax.set_ylabel("oracle gated ASR (arm-space ceiling)"); ax.set_title("Arm-space ablation: each family raises the ceiling")
ax.legend(); fig.tight_layout(); fig.savefig(OUT / "fig4_ablation.png", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 5 + experiment: recon->gated Spearman per model ----------
corr = {}
fig, ax = plt.subplots(figsize=(8.5, 4.4))
allm = models(MJ)
xw = np.arange(len(allm)); w = 0.38
for off, (name, mod) in zip((-w / 2, w / 2), DS):
    rs = [float(spearmanr(mod.RC[t], mod.G[t]).correlation) for t in allm]
    corr[name] = {t: rs[i] for i, t in enumerate(allm)}
    ax.bar(xw + off, rs, w, label=f"{name} (mean {np.nanmean(rs):+.2f})")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(xw); ax.set_xticklabels([SHORT[t] for t in allm], rotation=30, ha="right")
ax.set_ylabel("Spearman ρ (recon vs gated)")
ax.set_title("Does benign recon predict gated? decouples on strongly-aligned models")
ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(OUT / "fig5_recon_corr.png", bbox_inches="tight"); plt.close(fig)

# save experiment numbers
exp = {"ablation_cumulative_oracle": {"families_order": FAMS, **abl},
       "recon_gated_spearman": corr,
       "recon_gated_spearman_mean": {name: float(np.nanmean(list(corr[name].values()))) for name, _ in DS}}
Path("results/ablation_corr_20260907.json").write_text(json.dumps(exp, indent=2))
print("figs:", sorted(p.name for p in OUT.glob("*.png")))
print("ablation MJ:", [round(x, 3) for x in abl["MultiJail"]])
print("ablation Lingua:", [round(x, 3) for x in abl["Lingua-SafetyBench"]])
print("recon-gated mean rho:", exp["recon_gated_spearman_mean"])
