#!/usr/bin/env python3
"""How small can a probe be?

Everywhere else a probe of a configuration observes a full evaluation batch (64 items on MultiJail,
40 on Lingua-SafetyBench), so a three-query budget means three configurations tried, not three
prompts sent. A defender monitoring individual prompts sees the prompt count, so the honest question
is whether the selector still works when a probe sees only a handful of items.

We rerun the selector with the probe's observation noise set by the number of items it sees,
m in {1, 5, 10, 20, full}, holding everything else fixed. The underlying per-configuration values are
the stored batch means; only the noise the selector has to see through changes. This is a
sensitivity analysis of the selection rule, not a new evaluation.

Writes results/probe_batch_size_20260908.json and paper/tab_sel_batch.tex.
"""
from __future__ import annotations
import io, json, contextlib, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
with contextlib.redirect_stdout(io.StringIO()):
    import mj_bandit_full as MJ
    import lingua_bandit_full as LG
from gp_bai import gp_bai            # noqa: E402
from benign_prior import prior_for   # noqa: E402

REPS = 120
BUDGETS = (3, 6, 10)
SIZES = (1, 5, 10, 20, None)         # None = the full evaluation batch
DS = (("MultiJail", MJ, 64), ("Lingua-SafetyBench", LG, 40))


def main():
    out = {}
    for name, mod, full in DS:
        models = mod.STRONG if hasattr(mod, "STRONG") else mod.MODELS
        G, FE, names = mod.G, mod.FE, mod.names
        rng = np.random.default_rng(0)
        priors = {}
        for t in models:
            try:
                priors[t] = prior_for(name, t, names)[0] * 0.6
            except FileNotFoundError:
                priors[t] = None
        usable = [t for t in models if priors[t] is not None]
        out[name] = {"targets": usable, "oracle": round(float(np.mean([G[t].max() for t in usable])), 3),
                     "curves": {}}
        for m in SIZES:
            k = "full" if m is None else str(m)
            n_items = full if m is None else m
            out[name]["curves"][k] = {}
            for B in BUDGETS:
                acc = []
                for _ in range(REPS):
                    per = []
                    for t in usable:
                        def probe(a, t=t):
                            p = G[t][a]
                            sd = np.sqrt(max(p * (1 - p), .01) / n_items)
                            return float(np.clip(p + rng.normal(0, sd), 0, 1))
                        rec, _ = gp_bai(priors[t], FE, probe, B)
                        per.append(G[t][rec])
                    acc.append(np.mean(per))
                out[name]["curves"][k][B] = round(float(np.mean(acc)), 3)
            print(f"  {name}: items/probe {k:>4s} -> " +
                  "  ".join(f"B={B} {out[name]['curves'][k][B]:.3f}" for B in BUDGETS), flush=True)
    Path("results/probe_batch_size_20260908.json").write_text(json.dumps(out, indent=2))

    L = ["\\begin{table}[t]", "\\centering", "\\small",
         "\\caption{How much the probe has to see. A probe normally observes a full evaluation batch; "
         "here it observes only $m$ items, so its observation noise grows as $1/\\sqrt{m}$. Verified "
         "ASR of the recommended configuration, averaged over the targets with a harmless probe.}",
         "\\label{tab:sel_batch}", "\\begin{tabular}{lccc|ccc}", "\\toprule"]
    names = [n for n, _, _ in DS if n in out]
    L.append("& " + " & ".join("\\multicolumn{3}{c%s}{%s}" % ("|" if i == 0 and len(names) > 1 else "", n)
                               for i, n in enumerate(names)) + " \\\\")
    L.append("\\cmidrule(lr){2-4}" + ("\\cmidrule(lr){5-7}" if len(names) > 1 else ""))
    L.append("Items per probe & " + " & ".join([" & ".join(f"$B{{=}}{B}$" for B in BUDGETS)] * len(names)) + " \\\\")
    L.append("\\midrule")
    for m in SIZES:
        k = "full" if m is None else str(m)
        cells = []
        for n in names:
            cells += [f"{out[n]['curves'][k][B]:.3f}" for B in BUDGETS]
        label = "full batch" if m is None else str(m)
        L.append(f"{label} & " + " & ".join(cells) + " \\\\")
    L += ["\\midrule",
          "oracle & " + " & ".join(f"\\multicolumn{{3}}{{c{'|' if i == 0 and len(names) > 1 else ''}}}{{{out[n]['oracle']:.3f}}}"
                                   for i, n in enumerate(names)) + " \\\\",
          "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    Path("paper/tab_sel_batch.tex").write_text("\n".join(L))
    print("wrote paper/tab_sel_batch.tex")


if __name__ == "__main__":
    raise SystemExit(main())
