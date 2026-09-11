#!/usr/bin/env python3
"""EXP04d strategy bake-off across every collected model x collection: which acquisition is
CONSISTENTLY best for our data, and how does that track a data characteristic?

Strategies (all seed-averaged, replay over exp02 matrix + benign prior):
  random, ucb_iso, ucb_add, ucb_add_pibo, ttts_add, ttts_add_pibo   (from search_sota)

Per (model, collection) we also compute the DATA CHARACTERISTIC that should decide the winner:
  prior_info   = Pearson corr between the benign prior and the true verified over arms
                 (high -> prior is trustworthy; ~0 / negative -> saturated or misleading)
  recon_mean   = benign reconstruction mean (comprehension saturation)
Summary metric per strategy = normalized AUC of the verified@k budget curve (mean_k verified@k / oracle)
plus verified@3 / oracle (small-budget). We report per-cell winners and the across-cell consistency
(mean rank, win counts) split by whether the prior is informative.

Offline. Usage: python compare_strategies.py --root <exp02 results> [--suffix _mj|_lg|both] [--allow-incomplete]"""
import argparse, json, glob, collections
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import search_sota as S

STRATS = [("random", None), ("ucb_iso", dict(acq="ucb", additive=False, use_pibo=False)),
          ("ucb_add", dict(acq="ucb", additive=True, use_pibo=False)),
          ("ucb_add_pibo", dict(acq="ucb", additive=True, use_pibo=True)),
          ("ttts_add", dict(acq="ttts", additive=True, use_pibo=False)),
          ("ttts_add_pibo", dict(acq="ttts", additive=True, use_pibo=True))]


def prior_info(arms, ver, prior):
    a = np.array([prior.get(x, 0.5) for x in arms]); b = np.array([ver[x] for x in arms])
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def cell_run(arms, ver, prior, budget, reps):
    oracle = max(ver.values())
    rng = np.random.default_rng(0)
    rand = np.array([np.mean([max(ver[a] for a in list(np.random.default_rng(s).choice(arms, k, replace=False)))
                              for s in range(200)]) for k in range(1, budget + 1)])
    res = {"random": rand}
    for name, kw in STRATS:
        if name == "random":
            continue
        res[name] = np.array(S.avg(arms, ver, prior, budget, reps, **kw))
    auc = {n: round(float(c.mean() / oracle), 3) for n, c in res.items()}
    at3 = {n: round(float(c[min(2, budget-1)] / oracle), 3) for n, c in res.items()}
    return oracle, auc, at3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True); ap.add_argument("--suffix", default="both")
    ap.add_argument("--budget", type=int, default=8); ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--allow-incomplete", action="store_true")
    a = ap.parse_args()
    suffixes = ["_mj", "_lg"] if a.suffix == "both" else [a.suffix]
    rows = []
    for suf in suffixes:
        ver, pri = S.load(a.root, suf)
        for tag in sorted(ver):
            if tag not in pri:
                continue
            arms = list(ver[tag])
            if len(arms) < 292 and not a.allow_incomplete:
                print(f"skip {tag}: incomplete ({len(arms)}/292); use --allow-incomplete"); continue
            pinfo = prior_info(arms, ver[tag], pri[tag])
            bj = json.loads((Path(a.root) / "benign" / f"{tag}.json").read_text())
            _, axinfo = S.saturation_aware_prior(bj, arms)
            oracle, auc, at3 = cell_run(arms, ver[tag], pri[tag], a.budget, a.reps)
            bench = json.loads((Path(a.root) / "benign" / f"{tag}.json").read_text())
            rows.append(dict(tag=tag, n_arms=len(arms), oracle=round(oracle, 3),
                             prior_info=round(pinfo, 3), recon_mean=bench.get("benign_recon_mean"),
                             comp_informative=axinfo["comp_informative"], will_informative=axinfo["will_informative"],
                             auc=auc, at3=at3, winner_auc=max(auc, key=auc.get),
                             winner_at3=max(at3, key=at3.get)))
    # consistency summary
    names = [n for n, _ in STRATS]
    win_auc = collections.Counter(r["winner_auc"] for r in rows)
    win_at3 = collections.Counter(r["winner_at3"] for r in rows)
    meanrank = {n: 0.0 for n in names}
    for r in rows:
        order = sorted(names, key=lambda n: -r["auc"][n])
        for rank, n in enumerate(order, 1):
            meanrank[n] += rank
    for n in meanrank:
        meanrank[n] = round(meanrank[n] / max(len(rows), 1), 2)
    out = dict(cells=rows, n_cells=len(rows),
               wins_auc=dict(win_auc), wins_at3=dict(win_at3), mean_rank_auc=meanrank)
    print(json.dumps(out, indent=2))
    Path(a.root, "strategy_bakeoff.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", Path(a.root, "strategy_bakeoff.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
