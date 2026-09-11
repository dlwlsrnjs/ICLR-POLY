#!/usr/bin/env python3
"""EXP04e ablation: does dropping the SATURATED prior axis online (decided at benign-probe time, before
spending harmful queries) actually improve query efficiency? Logs the per-cell decision and effect.

For each collected (model, collection):
  - decision (from the benign probe only): comp_informative, will_informative  -> which axis is dropped
  - selector budget curve WITH drop  (saturation-aware prior) vs WITHOUT drop (raw stored prior)
    for the prior-using acquisitions (ucb_add_pibo, ttts_add_pibo)
  - drop_effect = AUC(drop) - AUC(nodrop);  positive => dropping the dead axis helped
Everything is written to axis_drop_ablation<suffix>.json so 'aggregate-then-drop was effective' is auditable.
Usage: python axis_drop_ablation.py --root <exp02 results> [--suffix _mj|_lg|both] [--allow-incomplete]"""
import argparse, json, glob
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import search_sota as S


def auc(arms, ver, prior, budget, reps, **kw):
    c = np.array(S.avg(arms, ver, prior, budget, reps, **kw))
    return round(float(c.mean() / max(ver.values())), 3), [round(float(x), 3) for x in c]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True); ap.add_argument("--suffix", default="both")
    ap.add_argument("--budget", type=int, default=8); ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--allow-incomplete", action="store_true")
    a = ap.parse_args()
    sufs = ["_mj", "_lg"] if a.suffix == "both" else [a.suffix]
    rows = []
    for suf in sufs:
        ver_raw, _ = S.load(a.root, suf, saturation_aware=False)   # raw stored prior (no drop)
        ver, pri_drop = S.load(a.root, suf, saturation_aware=True)  # saturation-aware (drop)
        for tag in sorted(ver):
            arms = list(ver[tag])
            if len(arms) < 292 and not a.allow_incomplete:
                print(f"skip {tag}: incomplete ({len(arms)}/292)"); continue
            bj = json.loads((Path(a.root) / "benign" / f"{tag}.json").read_text())
            praw, _ = S.saturation_aware_prior(bj, arms, info_range=1e9)  # info_range huge -> nothing dropped == raw-ish
            praw = {x: bj.get("prior", {}).get(x, 0.5) for x in arms}    # true raw stored prior
            pdrop, axinfo = S.saturation_aware_prior(bj, arms)
            rec = {}
            for name, kw in (("ucb_add_pibo", dict(acq="ucb", additive=True, use_pibo=True)),
                             ("ttts_add_pibo", dict(acq="ttts", additive=True, use_pibo=True))):
                a_nodrop, c_nodrop = auc(arms, ver[tag], praw, a.budget, a.reps, **kw)
                a_drop, c_drop = auc(arms, ver[tag], pdrop, a.budget, a.reps, **kw)
                rec[name] = dict(auc_nodrop=a_nodrop, auc_drop=a_drop,
                                 drop_effect=round(a_drop - a_nodrop, 3),
                                 curve_nodrop=c_nodrop, curve_drop=c_drop)
            row = dict(tag=tag, decision=axinfo, dropped_axis=(
                "willingness" if axinfo["comp_informative"] and not axinfo["will_informative"]
                else "comprehension" if axinfo["will_informative"] and not axinfo["comp_informative"]
                else "none" if axinfo["comp_informative"] and axinfo["will_informative"] else "both"),
                ablation=rec)
            rows.append(row)
            print(json.dumps(row, indent=2))
    eff = [r["ablation"]["ucb_add_pibo"]["drop_effect"] for r in rows]
    summ = dict(n_cells=len(rows), cells=rows,
                mean_drop_effect_ucb_pibo=round(float(np.mean(eff)), 3) if eff else None,
                cells_helped=sum(1 for e in eff if e > 0))
    Path(a.root, f"axis_drop_ablation{'' if a.suffix=='both' else a.suffix}.json").write_text(json.dumps(summ, indent=2))
    print("\nSUMMARY mean drop_effect (ucb_add_pibo):", summ["mean_drop_effect_ucb_pibo"],
          "| cells helped:", summ["cells_helped"], "/", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
