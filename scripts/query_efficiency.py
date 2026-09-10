#!/usr/bin/env python3
"""Query efficiency: how many harmful queries does each search strategy need to reach the level the
benign warm start reaches at three?

All strategies are evaluated on the SAME stored arm matrix and the SAME probe-noise model, so the
comparison is about search efficiency and nothing else. Strategies:

  repeat-fixed      submit the best fixed configuration again and again. Decoding is greedy, so
                    repetition returns the same response; the curve is flat. This is the "one
                    template, many queries" practice the paper argues against.
  random-search     probe uniformly random distinct configurations, recommend the best observed.
                    The uninformed iterative attacker.
  uninformed GP     GP-BAI with a constant prior (no information at all).
  cross-target GP   GP-BAI whose prior is the mean verified ASR of the other eight targets, i.e.
                    harmful supervision from other models that a real attacker would not hold.
  benign GP (ours)  GP-BAI whose prior is the benign reconstruction rate of the target itself.

Writes results/query_efficiency_20260907.json, results/figs/fig6_query_efficiency.png and
paper/tab_sel_queryeff.tex. Offline, no GPU.
"""
from __future__ import annotations
import io, json, contextlib, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
with contextlib.redirect_stdout(io.StringIO()):     # the bandit modules print on import
    import mj_bandit_full as MJ
    import lingua_bandit_full as LG
from gp_bai import gp_bai  # noqa: E402
from benign_prior import prior_for  # noqa: E402  harmless probe only

REPS = 400
BUDGETS = list(range(1, 13))
OUT = Path("results/figs"); OUT.mkdir(parents=True, exist_ok=True)


def panel(mod):
    return mod.STRONG if hasattr(mod, "STRONG") else mod.MODELS


def curves(mod, n_items):
    models = panel(mod); G = mod.G; RC = mod.RC; FE = mod.FE
    A = len(mod.names)
    rng = np.random.default_rng(0)
    fixed_arm = int(np.argmax(np.mean([G[t] for t in models], axis=0)))

    def probe(t):
        # binomial standard error at the collection's item count, as in bandit_bootstrap_ci
        return lambda a, t=t: float(np.clip(
            G[t][a] + rng.normal(0, np.sqrt(max(G[t][a] * (1 - G[t][a]), .01) / n_items)), 0, 1))

    coll = "MultiJail" if n_items == 64 else "Lingua-SafetyBench"

    def gp(kind, B):
        vals = []
        for _ in range(REPS // 10):
            per = []
            for t in models:
                if kind == "benign":
                    pr = prior_for(coll, t, mod.names)[0] * 0.6      # harmless probe, never RC[t]
                elif kind == "flat":
                    pr = np.full(A, float(np.mean([G[x].mean() for x in models])))
                else:
                    pr = np.mean([G[x] for x in models if x != t], axis=0)
                rec, _ = gp_bai(pr, FE, probe(t), B)
                per.append(G[t][rec])
            vals.append(np.mean(per))
        return float(np.mean(vals))

    def random_search(B):
        vals = []
        for _ in range(REPS):
            per = []
            for t in models:
                idx = rng.choice(A, size=B, replace=False)
                obs = [probe(t)(int(a)) for a in idx]
                per.append(G[t][int(idx[int(np.argmax(obs))])])   # recommend the best observed
            vals.append(np.mean(per))
        return float(np.mean(vals))

    fixed = float(np.mean([G[t][fixed_arm] for t in models]))
    oracle = float(np.mean([G[t].max() for t in models]))
    out = {"oracle": round(oracle, 3), "fixed_arm": mod.names[fixed_arm], "fixed": round(fixed, 3),
           "budgets": BUDGETS, "repeat-fixed": [round(fixed, 3)] * len(BUDGETS),
           "random-search": [], "uninformed": [], "cross-target": [], "benign": []}
    for B in BUDGETS:
        out["random-search"].append(round(random_search(B), 3))
        out["uninformed"].append(round(gp("flat", B), 3))
        out["cross-target"].append(round(gp("loo", B), 3))
        out["benign"].append(round(gp("benign", B), 3))
        print(f"  budget {B:2d}  benign {out['benign'][-1]:.3f}  cross {out['cross-target'][-1]:.3f}"
              f"  uninformed {out['uninformed'][-1]:.3f}  random {out['random-search'][-1]:.3f}", flush=True)
    # No-structure control (reviewer Q3): independent-arm UCB---forced exploration of unseen arms then
    # UCB, no prior and no cross-arm kernel---to show the uninformed budget is not an artifact of the
    # GP acquisition. Uses its own RNG so the curves above are bit-identical to before.
    rng2 = np.random.default_rng(12345)
    def indep_ucb(B):
        vals = []
        for _ in range(REPS):
            per = []
            for t in models:
                n = np.zeros(A); s = np.zeros(A); order = rng2.permutation(A)
                for step in range(B):
                    unseen = [int(a) for a in order if n[a] == 0]
                    if unseen:
                        a = unseen[0]
                    else:
                        mean = s / np.maximum(n, 1)
                        a = int(np.argmax(mean + np.sqrt(2 * np.log(step + 1) / np.maximum(n, 1))))
                    y = float(np.clip(G[t][a] + rng2.normal(0, np.sqrt(max(G[t][a] * (1 - G[t][a]), .01) / n_items)), 0, 1))
                    n[a] += 1; s[a] += y
                per.append(G[t][int(np.argmax(s / np.maximum(n, 1)))])
            vals.append(np.mean(per))
        return float(np.mean(vals))
    out["indep-ucb"] = [round(indep_ucb(B), 3) for B in BUDGETS]
    print("  indep-ucb:", out["indep-ucb"], flush=True)
    return out


def first_reach(curve, target):
    """smallest budget at which the curve reaches target, else None"""
    for b, v in zip(BUDGETS, curve):
        if v >= target - 1e-9:
            return b
    return None


def main():
    cache = Path("results/query_efficiency_20260907.json")
    if "--replot" in sys.argv and cache.exists():
        res = json.loads(cache.read_text())
        print("replotting from", cache)
        return finish(res)
    res = {}
    for name, mod, n_items in (("MultiJail", MJ, 64), ("Lingua-SafetyBench", LG, 40)):
        print(f"=== {name} ===")
        res[name] = curves(mod, n_items)
        ref = res[name]["benign"][BUDGETS.index(3)]
        res[name]["benign_at_3"] = ref
        res[name]["queries_to_match"] = {k: first_reach(res[name][k], ref)
                                         for k in ("random-search", "uninformed", "cross-target", "repeat-fixed", "indep-ucb")}
        print(f"  benign@3 = {ref:.3f}; queries needed to match:", res[name]["queries_to_match"])
    cache.write_text(json.dumps(res, indent=2))
    return finish(res)


def finish(res):

    # ---- figure ----
    style = {"benign": ("#1b6ca8", "o", "benign warm start (ours)"),
             "cross-target": ("#7a5195", "s", "cross-target prior (uses others' harmful data)"),
             "uninformed": ("#bc5090", "^", "uninformed GP search"),
             "random-search": ("#ef8a3f", "v", "random search"),
             "indep-ucb": ("#2e8b57", "D", "independent-arm UCB (no structure)"),
             "repeat-fixed": ("#8c8c8c", None, "repeat best fixed configuration")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for ax, name in zip(axes, res):
        r = res[name]
        for k, (c, m, lbl) in style.items():
            ax.plot(BUDGETS, r[k], color=c, marker=m, ms=4.5, lw=2 if k == "benign" else 1.5,
                    ls="--" if k == "repeat-fixed" else "-", label=lbl if ax is axes[0] else None)
        ax.axhline(r["oracle"], color="k", lw=1, ls=":")
        ax.text(BUDGETS[-1], r["oracle"] + .004, "oracle", ha="right", va="bottom", fontsize=8)
        b3 = r["benign_at_3"]
        ax.plot([3], [b3], marker="*", ms=15, color="#1b6ca8", zorder=5)
        need = r["queries_to_match"]["random-search"]
        if need:
            ax.annotate("", xy=(need, b3), xytext=(3, b3),
                        arrowprops=dict(arrowstyle="<->", color="#444", lw=1))
            ax.text((3 + need) / 2, b3, f"{need - 3} more harmful queries\nto reach the same point",
                    ha="center", va="center", fontsize=8, color="#333",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbb", lw=.6))
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("harmful queries to the target")
        ax.set_xticks(BUDGETS[::2])
        ax.grid(alpha=.25)
    axes[0].set_ylabel("verified ASR (mean over 9 targets)")
    axes[0].legend(fontsize=8, loc="lower right", framealpha=.95)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_query_efficiency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "fig6_query_efficiency.png")

    # ---- table ----
    # Only the "queries to match" content is unique to this table; the per-strategy verified ASR at a
    # three-query budget is the canonical budget curve reported in Table~\ref{tab:sel_main} /
    # Table~\ref{tab:sel_budget}, so we do not repeat it here (it would also disagree with the canonical
    # curve, which uses more replications than this figure's simulation).
    L = ["\\begin{table}[t]", "\\centering", "\\small",
         "\\caption{Query efficiency of the search itself: the smallest budget at which each strategy "
         "reaches what the benign warm start reaches at three queries (a dash means it does not within "
         "twelve). Every strategy is scored on the same arm matrix with the same probe noise, so the "
         "only difference is how the queries are spent; the verified ASR each reaches at three queries "
         "is the budget curve in Table~\\ref{tab:sel_budget}.}",
         "\\label{tab:sel_queryeff}", "\\begin{tabular}{lcc}", "\\toprule",
         "Strategy & MultiJail & Lingua-SafetyBench \\\\", "\\midrule"]
    order = [("repeat-fixed", "repeat best fixed configuration"),
             ("random-search", "random search"),
             ("uninformed", "uninformed GP search"),
             ("indep-ucb", "independent-arm UCB (no structure)"),
             ("cross-target", "cross-target prior (others' harmful data)"),
             ("benign", "benign warm start (ours)")]
    for k, lbl in order:
        cells = []
        for name in res:
            r = res[name]
            need = "3" if k == "benign" else (str(r["queries_to_match"][k]) if r["queries_to_match"].get(k) else "--")
            cells.append(need)
        L.append(f"{lbl} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    Path("paper/tab_sel_queryeff.tex").write_text("\n".join(L))
    print("wrote paper/tab_sel_queryeff.tex")

    # ---- prose macros ----
    DIG = str.maketrans({"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
                         "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"})
    lines = ["% auto-generated by scripts/query_efficiency.py -- do not edit"]
    for name, key in (("MultiJail", "MJ"), ("Lingua-SafetyBench", "LG")):
        r = res[name]
        for k, short in (("random-search", "rand"), ("uninformed", "unin"), ("cross-target", "cross"), ("indep-ucb", "iucb")):
            n = r["queries_to_match"][k]
            lines.append(f"\\newcommand{{\\pjq{short}{key}}}{{{n if n else '>12'}}}")
            # the uninformed-at-3 value is the same quantity as the canonical \pjflatThree* from the
            # main budget curve (selector_numbers.tex); emit only that one to avoid two names for it.
            if short != "unin":
                lines.append(f"\\newcommand{{\\pjq{short}at3{key}}}".translate(DIG) +
                             f"{{{r[k][BUDGETS.index(3)]:.3f}}}")
        lines.append(f"\\newcommand{{\\pjqrepeat{key}}}{{{r['fixed']:.3f}}}")
    Path("paper/query_numbers.tex").write_text("\n".join(lines) + "\n")
    print("wrote paper/query_numbers.tex")


if __name__ == "__main__":
    raise SystemExit(main())
