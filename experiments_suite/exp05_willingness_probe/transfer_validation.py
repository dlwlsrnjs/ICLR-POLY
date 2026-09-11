#!/usr/bin/env python3
"""Does the BENIGN willingness fingerprint (FalseReject) predict the HARMFUL comply ordering?

This is the empirical test behind the transfer/stealth claim. For every model that has both a
FalseReject willingness prior (benign, offline) and a harmful full-matrix, we:
  1. marginalize the harmful `verified` score over comprehension cells for each base willingness
     frame f in {plain,persona,fiction,pap}  ->  harmful_f  (per dataset + pooled),
  2. read the benign per-frame signal from the FR prior (nonref, and Δ-vs-plain),
  3. correlate benign vs harmful across the 4 frames (Pearson + Spearman) and check whether the
     benign argmax frame equals the harmful argmax frame (the decision the transfer selector makes).

If benign rankings track harmful rankings, the willingness axis can be learned on benign surrogates
and applied to a blind target -- no harmful willingness sweep on the target. Output:
results/transfer_validation.json (+ a printed table). Read-only over stored aggregates; no GPU.
"""
import glob, os, json, collections
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
MATRIX = HERE.parents[0] / "exp02_panel_collect" / "results" / "attack"
PRIORS = HERE / "results"
BASE_FRAMES = ["plain", "persona", "fiction", "pap"]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 1 else float("nan")


def harmful_per_frame(tag_ds, metric="verified"):
    """mean of `metric` (verified|unsafe|recon) over comprehension cells for each base frame."""
    acc = collections.defaultdict(list)
    for p in glob.glob(str(MATRIX / f"{tag_ds}__*.json")):
        name = os.path.basename(p)[:-5].split("__", 1)[1]  # after tag__
        if "__" not in name:
            continue
        cell, will = name.split("__", 1)
        if will in BASE_FRAMES:
            acc[will].append(json.load(open(p)).get(metric, 0.0))
    return {f: float(np.mean(acc[f])) for f in BASE_FRAMES if acc[f]}


def main():
    priors = {}
    for p in glob.glob(str(PRIORS / "willingness_prior_falsereject_*.json")):
        rec = json.load(open(p))
        priors[rec["tag"]] = rec

    out = {"models": {}, "summary": {}}
    # test the benign nonref signal against three harmful channels: comply(unsafe), verified, recon
    agree = {m: collections.Counter() for m in ("unsafe", "verified", "recon")}
    pear = {m: [] for m in ("unsafe", "verified", "recon")}
    spear = {m: [] for m in ("unsafe", "verified", "recon")}
    print(f"{'model':13} {'ds':3} {'benign_best':>11} | "
          f"{'unsafe_best':>11} {'p':>6} | {'verif_best':>10} {'p':>6} | {'recon_best':>10} {'p':>6}")
    for tag, rec in sorted(priors.items()):
        benign = {f: rec["per_frame"][f]["nonref"] for f in BASE_FRAMES}
        bbest = max(benign, key=benign.get)
        b = np.array([benign[f] for f in BASE_FRAMES])
        m = {"benign_nonref": benign, "benign_best": bbest, "datasets": {}}
        for ds in ("mj", "lg"):
            row = {}
            cells = {}
            for metric in ("unsafe", "verified", "recon"):
                hf = harmful_per_frame(f"{tag}_{ds}", metric)
                if len(hf) < len(BASE_FRAMES):
                    cells = None; break
                h = np.array([hf[f] for f in BASE_FRAMES])
                pv = float(np.corrcoef(b, h)[0, 1]) if b.std() > 0 and h.std() > 0 else float("nan")
                sv = spearman(b, h)
                hbest = max(hf, key=hf.get)
                row[metric] = dict(per_frame=hf, best=hbest, argmax_match=(bbest == hbest),
                                   pearson=round(pv, 3), spearman=round(sv, 3))
                agree[metric][bbest == hbest] += 1
                if not np.isnan(pv): pear[metric].append(pv)
                if not np.isnan(sv): spear[metric].append(sv)
                cells[metric] = (hbest, pv)
            if not cells:
                continue
            m["datasets"][ds] = row
            print(f"{tag:13} {ds:3} {bbest:>11} | "
                  f"{cells['unsafe'][0]:>11} {cells['unsafe'][1]:>6.2f} | "
                  f"{cells['verified'][0]:>10} {cells['verified'][1]:>6.2f} | "
                  f"{cells['recon'][0]:>10} {cells['recon'][1]:>6.2f}")
        out["models"][tag] = m

    def summ(metric):
        a = agree[metric]; n = a[True] + a[False]
        return dict(n=n, argmax_agreement=round(a[True] / n, 3) if n else None,
                    mean_pearson=round(float(np.mean(pear[metric])), 3) if pear[metric] else None,
                    mean_spearman=round(float(np.mean(spear[metric])), 3) if spear[metric] else None)
    out["summary"] = dict(
        by_channel={m: summ(m) for m in ("unsafe", "verified", "recon")},
        note="Benign FalseReject frame nonrefusal predicts harmful per-frame signal (marginalized over "
             "comprehension cells). Channels: unsafe=comply (what willingness should drive), "
             "verified=recon-gated success, recon=reconstruction. argmax = benign-best frame == "
             "harmful-best frame (the transfer selector's pick).",
    )
    # ---- feature search: which benign per-frame feature (if any) predicts harmful comply? ----
    # pooled across all model-datasets, correlate each benign feature vs harmful unsafe over frames.
    # (hedge is inverted: less hedging should mean more willing.)
    feats = {"nonref": lambda pf: pf["nonref"], "neg_hedge": lambda pf: -pf["hedge"],
             "fic_struct": lambda pf: pf["fic"], "words": lambda pf: pf["words"]}
    pooled = {k: {"b": [], "h": []} for k in feats}
    for tag, rec in sorted(priors.items()):
        for ds in ("mj", "lg"):
            hu = harmful_per_frame(f"{tag}_{ds}", "unsafe")
            if len(hu) < len(BASE_FRAMES):
                continue
            for k, fn in feats.items():
                for f in BASE_FRAMES:
                    pooled[k]["b"].append(fn(rec["per_frame"][f])); pooled[k]["h"].append(hu[f])
    fsearch = {}
    for k, d in pooled.items():
        b, h = np.array(d["b"]), np.array(d["h"])
        pv = float(np.corrcoef(b, h)[0, 1]) if len(b) > 1 and b.std() > 0 and h.std() > 0 else float("nan")
        fsearch[k] = round(pv, 3)
    out["summary"]["benign_feature_vs_harmful_unsafe_pooled_pearson"] = fsearch

    (PRIORS / "transfer_validation.json").write_text(json.dumps(out, indent=2))
    print("\nbenign feature -> harmful unsafe (pooled pearson over all frames/model-datasets):")
    for k, v in fsearch.items():
        print(f"  {k:10} {v:+.3f}")
    print("\nSUMMARY (benign FR nonref -> harmful channel):")
    for metric, s in out["summary"]["by_channel"].items():
        print(f"  {metric:9} n={s['n']}  argmax_agreement={s['argmax_agreement']}  "
              f"mean_pearson={s['mean_pearson']}  mean_spearman={s['mean_spearman']}")
    print(f"WROTE {PRIORS/'transfer_validation.json'}")


if __name__ == "__main__":
    main()
