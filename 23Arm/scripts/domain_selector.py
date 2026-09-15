#!/usr/bin/env python3
"""Is adapting per harm domain worth the extra queries?

domain_analysis.py shows the best configuration moves between domains inside a target. That alone
does not justify a per-domain selector: adapting per domain multiplies the query budget by the number
of domains, so it has to beat spending the same total number of queries on a single per-target
choice. This script runs that comparison on the per-item judgments from domain_breakdown.py.

Three strategies, all scored on the same per-item outcomes with the same probe-noise model:

  fixed            the configuration with the best panel-average, applied everywhere
  per target       a fixed-budget search that spends B harmful queries and commits to one
                   configuration for the whole target
  per domain       the same search run inside each domain with b queries each, so the total spend is
                   b x (number of domains)

Reported against total harmful queries, which is the quantity that has to be paid either way.
Writes results/domain_selector_20260908.json, paper/tab_sel_domainsel.tex, and appends macros to
paper/domain_numbers.tex.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
from statistics import mean
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from domain_analysis import ORDER, PRETTY, CFG, COLLECTIONS, domain_of, MIN_ITEMS  # noqa: E402

REPS = 300
rng = np.random.default_rng(0)


def load(path):
    """{target: {domain: {config: [per-item outcomes]}}}"""
    out = {}
    for t in ORDER:
        f = Path(path) / f"{t}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        cfgs = [c for c in CFG if c in d["configs"]]
        by = {}
        for c in cfgs:
            for it in d["configs"][c]["items"]:
                by.setdefault(domain_of(it["scenario"]), {}).setdefault(c, []).append(it["gated"])
        by = {k: v for k, v in by.items() if len(v[cfgs[0]]) >= MIN_ITEMS}
        if by:
            out[t] = (cfgs, by)
    return out


def search(values, n_items, budget):
    """Probe `budget` distinct configurations with binomial noise, return the true value of the one
    that looked best. This is the same selection rule the paper's selector uses, without the GP: with
    five configurations the feature-space smoothing has nothing to interpolate."""
    keys = list(values)
    if budget >= len(keys):
        obs = {k: values[k] + rng.normal(0, np.sqrt(max(values[k] * (1 - values[k]), .01) / n_items))
               for k in keys}
        return values[max(obs, key=obs.get)]
    idx = rng.choice(len(keys), size=budget, replace=False)
    obs = {}
    for i in idx:
        k = keys[int(i)]
        obs[k] = values[k] + rng.normal(0, np.sqrt(max(values[k] * (1 - values[k]), .01) / n_items))
    return values[max(obs, key=obs.get)]


def evaluate(data, fixed_cfg):
    """For each total-query budget, the mean verified ASR of each strategy over the targets."""
    res = {}
    tot_budgets = [1, 2, 3, 4, 5, 8, 10, 16, 24, 40]
    for name in ("fixed", "per target", "per domain"):
        res[name] = {}
    for t, (cfgs, by) in data.items():
        doms = sorted(by)
        n_dom = len(doms)
        # domain-weighted target mean, so both strategies are scored on the same population
        w = {d: len(by[d][cfgs[0]]) for d in doms}
        tot = sum(w.values())
        dom_mean = {c: sum(mean(by[d][c]) * w[d] for d in doms) / tot for c in cfgs}
        n_items_dom = mean(w.values())
        for B in tot_budgets:
            res["fixed"].setdefault(B, []).append(dom_mean[fixed_cfg] if fixed_cfg in dom_mean
                                                  else max(dom_mean.values()))
            # per target: all B queries buy one choice for the whole target
            vals = [search(dom_mean, tot, B) for _ in range(REPS)]
            res["per target"].setdefault(B, []).append(float(np.mean(vals)))
            # per domain: B queries split evenly across domains (at least one each if affordable)
            b = B // n_dom
            if b < 1:
                res["per domain"].setdefault(B, []).append(float("nan"))
                continue
            acc = []
            for _ in range(REPS):
                per = []
                for d in doms:
                    dv = {c: mean(by[d][c]) for c in cfgs}
                    per.append(search(dv, n_items_dom, b) * w[d])
                acc.append(sum(per) / tot)
            res["per domain"].setdefault(B, []).append(float(np.mean(acc)))
    return {k: {B: (None if any(np.isnan(x) for x in v) else round(float(np.mean(v)), 3))
                for B, v in d.items()} for k, d in res.items()}


def _write_block(tag, lines):
    """Replace this script's block in paper/domain_numbers.tex instead of appending a duplicate."""
    from pathlib import Path as _P
    p = _P("paper/domain_numbers.tex")
    start, end = f"% begin {tag}", f"% end {tag}"
    body = "\n".join([start] + lines + [end])
    old = p.read_text() if p.exists() else ""
    if start in old:
        old = re.sub(re.escape(start) + r".*?" + re.escape(end), body, old, flags=re.S)
    else:
        old = old.rstrip("\n") + "\n" + body + "\n"
    p.write_text(old.rstrip("\n") + "\n")


def main():
    out = {}
    for name, path in COLLECTIONS.items():
        data = load(path)
        if not data:
            continue
        cfgs = next(iter(data.values()))[0]
        # panel-average best configuration among those evaluated here (the "fixed" strategy)
        panel = {c: mean(mean(by[d][c]) for _, (_, by) in data.items() for d in by) for c in cfgs}
        fixed_cfg = max(panel, key=panel.get)
        res = evaluate(data, fixed_cfg)
        out[name] = {"targets": sorted(data), "fixed_cfg": fixed_cfg,
                     "panel_means": {c: round(v, 3) for c, v in panel.items()}, "curves": res}
        print(f"=== {name} ({len(data)} targets, fixed = {fixed_cfg}) ===")
        print(f"  {'total queries':>14s} " + "".join(f"{k:>13s}" for k in res))
        for B in sorted(res["fixed"]):
            cells = "".join(f"{(res[k][B] if res[k][B] is not None else float('nan')):>13.3f}" for k in res)
            print(f"  {B:>14d} {cells}")
    Path("results/domain_selector_20260908.json").write_text(json.dumps(out, indent=2))

    L = ["\\begin{table}[t]", "\\centering", "\\small",
         "\\caption{Adapting per harm domain against adapting per target, at equal total harmful "
         "queries. A per-domain search must pay for every domain, so it only pays off once the total "
         "budget covers them; below that a single per-target choice is better. Verified ASR, "
         "domain-weighted, over the targets with per-item judgments.}",
         "\\label{tab:sel_domainsel}", "\\begin{tabular}{lccc|ccc}", "\\toprule"]
    names = [n for n in out]
    L.append("& " + " & ".join("\\multicolumn{3}{c%s}{%s}" % ("|" if i == 0 and len(names) > 1 else "", n)
                               for i, n in enumerate(names)) + " \\\\")
    L.append("\\cmidrule(lr){2-4}" + ("\\cmidrule(lr){5-7}" if len(names) > 1 else ""))
    L.append("Total harmful queries & " + " & ".join(["fixed & per target & per domain"] * len(names)) + " \\\\")
    L.append("\\midrule")
    for B in (3, 5, 8, 16, 24, 40):
        cells = []
        for n in names:
            r = out[n]["curves"]
            for k in ("fixed", "per target", "per domain"):
                v = r[k].get(B)
                cells.append("--" if v is None else f"{v:.3f}")
        L.append(f"{B} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    Path("paper/tab_sel_domainsel.tex").write_text("\n".join(L))

    DIG = str.maketrans({"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
                         "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"})
    lines = []
    for n in names:
        key = "MJ" if n == "MultiJail" else "LG"
        r = out[n]["curves"]
        cross = next((B for B in sorted(r["fixed"]) if r["per domain"].get(B) is not None
                      and r["per domain"][B] > r["per target"][B]), None)
        lines.append(f"\\newcommand{{\\pjdomsel{key}cross}}{{{cross if cross else '>40'}}}")
        for B in (8, 24, 40):
            for k, tag in (("per target", "tgt"), ("per domain", "dom")):
                v = r[k].get(B)
                if v is not None:
                    lines.append(f"\\newcommand{{\\pjdomsel{key}{tag}{str(B).translate(DIG)}}}{{{v:.3f}}}")
    _write_block("domain_selector", lines)
    print("\nwrote paper/tab_sel_domainsel.tex and appended macros to paper/domain_numbers.tex")


if __name__ == "__main__":
    raise SystemExit(main())
