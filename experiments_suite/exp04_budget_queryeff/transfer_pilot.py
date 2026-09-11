#!/usr/bin/env python3
"""PILOT: does per-frame 'benign adoption -> harmful comply' transfer? LOTO across models.

Per (model, collection) with BOTH a benign probe and a harmful matrix:
  features (benign fingerprint): recon_mean, recon_min, recon_max (capability profile) +
     per-frame benign signals persona/fiction/pap {nonrefusal, fiction_hold, detail}
  labels (harmful comply per willingness): comply[w] = mean_over_cells verified(cell,w)  for
     w in {plain,persona,fiction,pap,persona+fiction}
Leave-one-model-out: predict comply[w] for the held-out model from the OTHERS via kNN (k configurable)
in benign-feature space; pick best willingness; combine with the held-out model's OWN benign recon
(per cell) to choose the arm; score verified of that arm vs the model's oracle. Baselines: modal
willingness (most common best across others), and per-target oracle.

Reports, per held-out model: predicted best willingness vs actual, verified achieved / oracle, for
transfer vs modal. Needs >=3 models with both. Offline.
Usage: python transfer_pilot.py --root <results> [--suffix _mj|_lg|both] [--k 1]"""
import argparse, json, glob, re, collections
from pathlib import Path
import numpy as np

CELL = re.compile(r"g(\d+)_(ordered|shuffled)_n(\d+)__(.+)")
WILL = ["plain", "persona", "fiction", "pap", "persona+fiction"]


def load_cellrecon_and_matrix(root, tag):
    bj = json.loads((Path(root) / "benign" / f"{tag}.json").read_text())
    cr = bj.get("benign_recon_by_cell", {})
    sig = bj.get("frame_signals", {})
    ver = {}
    for f in glob.glob(str(Path(root) / "attack" / f"{tag}__*.json")):
        d = json.loads(Path(f).read_text())
        if CELL.match(d["method"]) and CELL.match(d["method"]).group(4) in WILL:
            ver[d["method"]] = d["verified"]
    return cr, sig, ver


def features(cr, sig):
    rv = list(cr.values()) or [0.5]
    feat = [np.mean(rv), min(rv), max(rv)]
    for fr in ("persona", "fiction", "pap"):
        s = sig.get(fr, {})
        feat += [s.get("nonrefusal", 0.5), s.get("fiction_hold", 0.5), s.get("detail", 0.5)]
    return np.array(feat, float)


def comply_labels(ver):
    out = {}
    for w in WILL:
        vs = [v for a, v in ver.items() if CELL.match(a).group(4) == w]
        out[w] = float(np.mean(vs)) if vs else 0.0
    return out


def best_arm_verified(cr, ver, comply):
    """pick cell x willingness maximizing recon[cell]*comply[will]; return that arm's true verified."""
    best, bv = None, -1
    for a in ver:
        m = CELL.match(a); cell = f"g{m.group(1)}_{m.group(2)}_n{m.group(3)}"; w = m.group(4)
        score = cr.get(cell, np.mean(list(cr.values()) or [0.5])) * comply.get(w, 0.0)
        if score > bv:
            bv, best = score, a
    return best, ver[best]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True); ap.add_argument("--suffix", default="both"); ap.add_argument("--k", type=int, default=1)
    a = ap.parse_args()
    sufs = ["_mj", "_lg"] if a.suffix == "both" else [a.suffix]
    cells = {}
    for suf in sufs:
        for bf in glob.glob(str(Path(a.root) / "benign" / f"*{suf}.json")):
            tag = Path(bf).stem
            cr, sig, ver = load_cellrecon_and_matrix(a.root, tag)
            if len(ver) >= 160 and cr:
                cells[tag] = dict(feat=features(cr, sig), comply=comply_labels(ver), cr=cr, ver=ver,
                                  oracle=max(ver.values()))
    tags = sorted(cells)
    if len(tags) < 3:
        print(f"need >=3 models with both; have {len(tags)}: {tags}"); return 0
    rows = []
    for t in tags:
        others = [o for o in tags if o != t]
        F = np.array([cells[o]["feat"] for o in others])
        # normalize features
        mu, sd = F.mean(0), F.std(0) + 1e-9
        Fn = (F - mu) / sd; q = (cells[t]["feat"] - mu) / sd
        dist = np.linalg.norm(Fn - q, axis=1)
        nn = [others[i] for i in np.argsort(dist)[:a.k]]
        pred_comply = {w: float(np.mean([cells[o]["comply"][w] for o in nn])) for w in WILL}
        modal = {w: float(np.mean([cells[o]["comply"][w] for o in others])) for w in WILL}
        _, v_transfer = best_arm_verified(cells[t]["cr"], cells[t]["ver"], pred_comply)
        _, v_modal = best_arm_verified(cells[t]["cr"], cells[t]["ver"], modal)
        actual_best = max(cells[t]["comply"], key=cells[t]["comply"].get)
        pred_best = max(pred_comply, key=pred_comply.get)
        rows.append(dict(tag=t, oracle=round(cells[t]["oracle"], 3),
                         actual_best_will=actual_best, pred_best_will=pred_best,
                         match=pred_best == actual_best, nn=nn,
                         transfer_verified=round(v_transfer, 3), modal_verified=round(v_modal, 3),
                         transfer_ratio=round(v_transfer / cells[t]["oracle"], 3),
                         modal_ratio=round(v_modal / cells[t]["oracle"], 3)))
    for r in rows:
        print(f"[{r['tag']}] oracle={r['oracle']} actual_best={r['actual_best_will']:16} "
              f"pred={r['pred_best_will']:16} match={r['match']} | transfer={r['transfer_verified']}({r['transfer_ratio']}) "
              f"modal={r['modal_verified']}({r['modal_ratio']}) nn={r['nn']}")
    mt = np.mean([r["transfer_ratio"] for r in rows]); mm = np.mean([r["modal_ratio"] for r in rows])
    acc = np.mean([r["match"] for r in rows])
    print(f"\nLOTO summary ({len(rows)} cells): willingness-match={acc:.2f} | "
          f"transfer {mt:.3f} of oracle vs modal {mm:.3f}")
    Path(a.root, "transfer_pilot.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
