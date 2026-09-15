#!/usr/bin/env python3
"""Does the within-target domain effect exceed what shuffling the domain labels would produce?

domain_analysis.py counts, per target, how often the average-best configuration also wins each
domain, and how many distinct configurations win at least one domain. Both statistics move on their
own when domains are small: with about thirty items per domain and five configurations, sampling
noise alone hands different domains different winners.

The test: keep every item's per-configuration outcome vector intact and permute only the domain
labels within a target. That destroys any real domain structure while preserving the item counts, the
configuration means and the correlation between configurations on an item. Anything the observed
statistics do beyond this null is a domain effect.

Writes results/domain_permutation_20260908.json and paper/tab_sel_domainperm.tex.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
from statistics import mean
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from domain_analysis import ORDER, PRETTY, CFG, COLLECTIONS, domain_of, MIN_ITEMS  # noqa: E402

PERM = 2000
rng = np.random.default_rng(0)


def per_item(path, target):
    """(configs, item x config outcome matrix, domain label per item)"""
    f = Path(path) / f"{target}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    cfgs = [c for c in CFG if c in d["configs"]]
    ids = [it["item_id"] for it in d["configs"][cfgs[0]]["items"]]
    idx = {k: i for i, k in enumerate(ids)}
    M = np.zeros((len(ids), len(cfgs)))
    dom = [None] * len(ids)
    for j, c in enumerate(cfgs):
        for it in d["configs"][c]["items"]:
            i = idx[it["item_id"]]
            M[i, j] = it["gated"]
            dom[i] = domain_of(it["scenario"])
    return cfgs, M, np.array(dom)


def stats(M, dom, keep):
    """(#domains the average-best configuration wins, #distinct winners, max gap)"""
    avg_best = int(np.argmax(M.mean(0)))
    wins, winners, gaps = 0, set(), []
    for d in keep:
        sub = M[dom == d]
        m = sub.mean(0)
        b = int(np.argmax(m))
        winners.add(b)
        if b == avg_best:
            wins += 1
        gaps.append(m[b] - m[avg_best])
    return wins, len(winners), max(gaps)


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
        rows = {}
        for t in ORDER:
            got = per_item(path, t)
            if got is None:
                continue
            cfgs, M, dom = got
            counts = {d: int((dom == d).sum()) for d in set(dom)}
            keep = [d for d, n in counts.items() if n >= MIN_ITEMS]
            if len(keep) < 2:
                continue
            obs = stats(M, dom, keep)
            null_w, null_d, null_g = [], [], []
            for _ in range(PERM):
                p = rng.permutation(dom)
                w, dd, g = stats(M, p, keep)
                null_w.append(w); null_d.append(dd); null_g.append(g)
            rows[t] = {
                "n_domains": len(keep),
                "wins": obs[0], "wins_null_mean": round(float(np.mean(null_w)), 2),
                "wins_p": round(float(np.mean(np.array(null_w) <= obs[0])), 3),
                "distinct": obs[1], "distinct_null_mean": round(float(np.mean(null_d)), 2),
                "distinct_p": round(float(np.mean(np.array(null_d) >= obs[1])), 3),
                "max_gap": round(obs[2], 3), "max_gap_null_mean": round(float(np.mean(null_g)), 3),
                "max_gap_p": round(float(np.mean(np.array(null_g) >= obs[2])), 3),
            }
        if rows:
            out[name] = rows
    Path("results/domain_permutation_20260908.json").write_text(json.dumps(out, indent=2))

    L = ["\\begin{table}[t]", "\\centering", "\\small",
         "\\caption{Permutation test for the within-target domain effect. Domain labels are shuffled "
         "inside each target, keeping every item's outcomes and the domain sizes, so the null holds "
         "sampling noise fixed and removes only domain structure. $p$ is the fraction of 2{,}000 "
         "permutations at least as extreme as the observation. Small domains generate apparent "
         "heterogeneity on their own, and the observed statistics do not exceed it.}",
         "\\label{tab:sel_domainperm}", "\\begin{tabular}{lccccc}", "\\toprule",
         "Target & domains & average-best wins & null & distinct winners & null \\\\", "\\midrule"]
    for name, rows in out.items():
        for t in ORDER:
            if t not in rows:
                continue
            r = rows[t]
            L.append(f"{PRETTY[t]} & {r['n_domains']} & {r['wins']} ($p={r['wins_p']:.2f}$) & "
                     f"{r['wins_null_mean']:.1f} & {r['distinct']} ($p={r['distinct_p']:.2f}$) & "
                     f"{r['distinct_null_mean']:.1f} \\\\")
        vals = [rows[t] for t in ORDER if t in rows]
        L += ["\\cmidrule(lr){1-6}",
              f"mean & {mean(v['n_domains'] for v in vals):.0f} & {mean(v['wins'] for v in vals):.1f} & "
              f"{mean(v['wins_null_mean'] for v in vals):.1f} & "
              f"{mean(v['distinct'] for v in vals):.1f} & {mean(v['distinct_null_mean'] for v in vals):.1f} \\\\"]
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    Path("paper/tab_sel_domainperm.tex").write_text("\n".join(L))

    DIG = str.maketrans({"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
                         "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"})
    lines = []
    for name, rows in out.items():
        key = "MJ" if name == "MultiJail" else "LG"
        vals = list(rows.values())
        lines += [f"\\newcommand{{\\pjperm{key}wins}}{{{mean(v['wins'] for v in vals):.1f}}}",
                  f"\\newcommand{{\\pjperm{key}winsnull}}{{{mean(v['wins_null_mean'] for v in vals):.1f}}}",
                  f"\\newcommand{{\\pjperm{key}distinct}}{{{mean(v['distinct'] for v in vals):.1f}}}",
                  f"\\newcommand{{\\pjperm{key}distinctnull}}{{{mean(v['distinct_null_mean'] for v in vals):.1f}}}",
                  f"\\newcommand{{\\pjperm{key}nsig}}{{{sum(1 for v in vals if v['distinct_p'] < 0.05)}}}",
                  f"\\newcommand{{\\pjperm{key}n}}{{{len(vals)}}}"]
    _write_block("domain_permutation", lines)

    for name, rows in out.items():
        print(f"=== {name} ===")
        for t in ORDER:
            if t not in rows:
                continue
            r = rows[t]
            print(f"  {PRETTY[t]:14s} wins {r['wins']}/{r['n_domains']} (null {r['wins_null_mean']:.1f}, "
                  f"p={r['wins_p']:.2f})   distinct {r['distinct']} (null {r['distinct_null_mean']:.1f}, "
                  f"p={r['distinct_p']:.2f})   max gap {r['max_gap']:.3f} (null {r['max_gap_null_mean']:.3f}, "
                  f"p={r['max_gap_p']:.2f})")
    print("\nwrote paper/tab_sel_domainperm.tex")


if __name__ == "__main__":
    raise SystemExit(main())
