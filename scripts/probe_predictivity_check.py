#!/usr/bin/env python3
"""Step 1 validation: do cheap black-box behavioral probes predict a target's best
obfuscation config? Leave-one-model-out over the 9 readable panel_v2 targets.

Probe fingerprint (all measured WITHOUT the harmful interleave joint labels we predict):
  a  = english_direct raw ASR        (alignment: readiness to comply with plain harm)
  fi = translated_Finnish raw ASR    (multilingual safety gap)
  cap_lo/hi/mean = benign FLORES reconstruction rate (separate benign items; reassembly ability)
Label = the interleave config (10) maximizing JOINT gated ASR on held-out (eval) items.

If a probe-based predictor beats the no-probe global-fixed config, the fingerprint carries
signal and the universal+adaptive direction is alive. 9 targets is underpowered: report as
a signal, not a conclusion. No new generation or judge calls.
"""
from __future__ import annotations
import json, os
from pathlib import Path
from collections import defaultdict
import numpy as np

PANEL = Path("private_artifacts/panel_v2")
INTERLEAVE = [f"interleave_{a}_n{n}" for a in ("ordered", "shuffled") for n in (2, 4, 6, 8, 10)]

def load_joint(m):
    rp = PANEL/m/"harm_recon/restricted_reconstruction_audit.jsonl"
    gp = PANEL/m/"harm_guard/restricted_qwen3guard_audit.jsonl"
    if not (os.access(rp, os.R_OK) and os.access(gp, os.R_OK)):
        return None
    R, U = {}, {}
    for l in open(rp):
        r = json.loads(l)
        if r["reconstruction_parse_valid"]:
            R[(r["item_id"], r["condition"])] = bool(r["semantic_reconstruction_equivalent"])
    for l in open(gp):
        r = json.loads(l)
        if r["guard_parse_valid"]:
            U[(r["item_id"], r["condition"])] = bool(r["guard_strict_unsafe"])
    return R, U

def main():
    models = sorted(p.name for p in PANEL.iterdir() if p.is_dir())
    data = {m: load_joint(m) for m in models}
    models = [m for m in models if data[m] is not None]
    items = sorted({k[0] for k in data[models[0]][0] if k[1] == "english_direct"})
    rng = np.random.default_rng(20260903)
    perm = rng.permutation(len(items)); calib = set(np.array(items)[perm[:125]]); ev = set(np.array(items)[perm[125:]])

    def asr(m, cond, itemset):
        _, U = data[m]
        vals = [U[(it, cond)] for it in items if it in itemset and (it, cond) in U]
        return float(np.mean(vals)) if vals else np.nan

    def joint_gated(m, cond, itemset):
        R, U = data[m]
        vals = [1.0 if (R.get((it, cond)) and U.get((it, cond))) else 0.0
                for it in items if it in itemset and (it, cond) in U and (it, cond) in R]
        return float(np.mean(vals)) if vals else np.nan

    feats, gated_eval, best_cfg = {}, {}, {}
    for m in models:
        benign = json.load(open(PANEL/m/"benign_recon.json"))
        cap = [benign.get(c, np.nan) for c in INTERLEAVE]
        feats[m] = np.array([
            asr(m, "english_direct", calib),                 # alignment
            asr(m, "translated_direct_Finnish", calib),      # multilingual gap
            float(np.nanmean(cap)),                          # reassembly capability (mean)
            float(np.nanmean([benign.get(f"interleave_ordered_n2", np.nan),
                              benign.get(f"interleave_shuffled_n2", np.nan)])),   # low-load cap
            float(np.nanmean([benign.get(f"interleave_ordered_n10", np.nan),
                              benign.get(f"interleave_shuffled_n10", np.nan)])),  # high-load cap
        ])
        g = np.array([joint_gated(m, c, ev) for c in INTERLEAVE])
        gated_eval[m] = g
        best_cfg[m] = int(np.nanargmax(g))

    F = np.array([feats[m] for m in models])
    Fz = (F - F.mean(0)) / (F.std(0) + 1e-9)
    G = np.array([gated_eval[m] for m in models])  # [M, 10]

    rows = []
    for hi, h in enumerate(models):
        tr = [i for i in range(len(models)) if i != hi]
        # 1) no-probe global fixed: best mean interleave config over training targets
        gfix = int(np.nanmean(G[tr], 0).argmax())
        # 2) nearest-probe: training target closest in fingerprint space -> its best config
        d = np.linalg.norm(Fz[tr] - Fz[hi], axis=1)
        nn = tr[int(d.argmin())]
        nn_cfg = int(np.nanargmax(G[nn]))
        # 3) ridge per-config regression on fingerprint -> argmax predicted
        lam = 1.0
        pred = []
        Xtr = np.column_stack([np.ones(len(tr)), Fz[tr]])
        for c in range(len(INTERLEAVE)):
            y = G[tr, c]
            w = np.linalg.solve(Xtr.T@Xtr + lam*np.eye(Xtr.shape[1]), Xtr.T@y)
            pred.append(np.array([1.0, *Fz[hi]])@w)
        ridge_cfg = int(np.argmax(pred))
        oracle = int(np.nanargmax(G[hi]))
        rows.append(dict(model=h,
                         oracle_cfg=INTERLEAVE[oracle], oracle=float(G[hi, oracle]),
                         global_fixed_cfg=INTERLEAVE[gfix], global_fixed=float(G[hi, gfix]),
                         nn_probe_cfg=INTERLEAVE[nn_cfg], nn_probe=float(G[hi, nn_cfg]), nn_neighbor=models[nn],
                         ridge_cfg=INTERLEAVE[ridge_cfg], ridge=float(G[hi, ridge_cfg])))

    def agg(key):
        v = np.array([r[key] for r in rows]); orc = np.array([r["oracle"] for r in rows])
        b = np.array([np.random.default_rng(s).choice(v, len(v)).mean() for s in range(3000)])
        return dict(mean=float(v.mean()), ci=[float(np.percentile(b,2.5)), float(np.percentile(b,97.5))],
                    regret=float((orc-v).mean()))
    summary = dict(
        scope="Do black-box behavioral probes predict the best obfuscation config? LOO over 9 targets.",
        note="9 targets is underpowered; a signal, not a conclusion. No leakage: probes exclude the "
             "harmful interleave joint labels being predicted; benign capability is a separate dataset.",
        features=["english_direct_raw_asr(calib)", "translated_Finnish_raw_asr(calib)",
                  "benign_recon_mean", "benign_recon_low_n2", "benign_recon_high_n10"],
        methods={k: agg(k) for k in ("oracle", "global_fixed", "nn_probe", "ridge")},
        per_model=rows)
    out = Path("results/probe_predictivity_20260903"); out.mkdir(parents=True, exist_ok=True)
    (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")

    print(f"targets: {models}")
    print("\nMean JOINT gated ASR of the predicted config on held-out target (LOO, eval items):")
    for k in ("oracle", "ridge", "nn_probe", "global_fixed"):
        a = summary["methods"][k]
        print(f"  {k:13s} {a['mean']:.3f}  95%CI[{a['ci'][0]:.3f},{a['ci'][1]:.3f}]  regret_vs_oracle={a['regret']:.3f}")
    # interpretable: does reassembly capability predict tolerated language load of the best config?
    caps = np.array([feats[m][2] for m in models])
    best_n = np.array([int(INTERLEAVE[best_cfg[m]].split("n")[-1]) for m in models])
    aln = np.array([feats[m][0] for m in models])
    def spearman(x, y):
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])
    print(f"\nSpearman(capability, best-config language load n) = {spearman(caps, best_n):.2f}")
    print(f"Spearman(alignment english_direct_asr, best gated) = {spearman(aln, [G[i,best_cfg[m]] for i,m in enumerate(models)]):.2f}")
    print("\nPer-model picks:")
    for r in rows:
        print(f"  {r['model']:12s} oracle={r['oracle_cfg']:22s}({r['oracle']:.2f})  "
              f"ridge={r['ridge_cfg']:22s}({r['ridge']:.2f})  nn={r['nn_probe_cfg']:22s}({r['nn_probe']:.2f})<-{r['nn_neighbor']}")

if __name__ == "__main__":
    main()
