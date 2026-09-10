#!/usr/bin/env python3
"""Cross-method comparison: PolyJigsaw (adaptive) vs other single-shot jailbreak methodologies,
panel means, raw unsafe + reconstruction-gated. Reads results/method_baselines_20260906 and the
adaptive numbers from results/sequential_resource_20260906 (+ disorder)."""
import json, glob, os
from pathlib import Path
import numpy as np

MB = {}
for f in glob.glob("results/method_baselines_20260906/*.json"):
    b = os.path.basename(f)
    if b.startswith("panel_"): continue
    d = json.loads(open(f).read()); MB[d["target"]] = d["methods"]

SEQ = {}
for f in glob.glob("results/sequential_resource_20260906/*.json"):
    b = os.path.basename(f)
    if b.startswith("panel_") or b.endswith("aggregate.json") or "_20260906.json" in b: continue
    d = json.loads(open(f).read())
    if "steps" in d: SEQ[d["target"]] = {s["k"]: s for s in d["steps"]}
DIS = {}
for f in glob.glob("results/disorder_sweep_20260906/*.json"):
    b = os.path.basename(f)
    if b.startswith("panel_"): continue
    d = json.loads(open(f).read()); DIS[d["target"]] = {r["delta"]: r["gated"] for r in d["rows"]}

METHODS = ["plain", "translated", "cipher_base64", "aim", "deepinception", "pap"]
QUERIES = {"plain": 1, "translated": 1, "cipher_base64": 1, "aim": 1, "deepinception": 1, "pap": 1}


def warm_start_k(R, tau=0.6):
    ks = sorted(R); good = [k for k in ks if R[k]["recon"] >= tau]
    return max(good) if good else ks[0]


def policy(R):
    """online_policy: benign-edge warm start + local recon-gated hillclimb. Returns (gated, queries)."""
    ks = sorted(R); lo, hi = ks[0], ks[-1]
    k0 = warm_start_k(R); best_k = k0; best = R[k0]["gated"]; q = 1; bad = 0; k = k0
    while k < hi:
        k += 1
        if R[k]["recon"] < 0.4:
            break
        q += 1; g = R[k]["gated"]
        if g > best:
            best = g; best_k = k; bad = 0
        else:
            bad += 1
            if bad >= 1:
                break
    if k0 - 1 >= lo:
        q += 1
        if R[k0 - 1]["gated"] > best:
            best = R[k0 - 1]["gated"]
    return best, q


tags = sorted(MB)
# single fixed config (1-shot ours) = best-on-average k across targets
fk = None
if SEQ:
    kall = sorted({k for t in SEQ for k in SEQ[t]})
    kavg = {k: np.mean([SEQ[t][k]["gated"] for t in SEQ if k in SEQ[t]]) for k in kall}
    fk = max(kavg, key=kavg.get)
print(f"방법 비교 ({len(tags)}모델). 각 셀 = raw unsafe / gated\n")
print(f"{'model':22s}" + "".join(f"{m[:10]:12s}" for m in METHODS) + f"{'ours_1shot':12s}{'ours_adapt':12s}")
agg = {m: {"u": [], "g": []} for m in METHODS}; o1, oa, oq = [], [], []
for t in tags:
    line = f"{t:22s}"
    for m in METHODS:
        d = MB[t].get(m, {}); u, g = d.get("unsafe", np.nan), d.get("gated", np.nan)
        agg[m]["u"].append(u); agg[m]["g"].append(g)
        line += f"{u:.2f}/{g:.2f}  "
    f1 = SEQ[t][fk]["gated"] if (t in SEQ and fk in SEQ[t]) else np.nan   # ours fixed 1-shot
    pg, pq = policy(SEQ[t]) if t in SEQ else (np.nan, np.nan)             # ours adaptive (policy budget)
    o1.append(f1); oa.append(pg); oq.append(pq)
    line += f"  -  /{f1:.2f}    -  /{pg:.2f}(q{pq})" if not np.isnan(f1) else ""
    print(line)
print("\n=== panel means (raw unsafe | gated | queries) ===")
for m in METHODS:
    print(f"  {m:16s} unsafe {np.nanmean(agg[m]['u']):.3f} | gated {np.nanmean(agg[m]['g']):.3f} | q {QUERIES[m]}")
print(f"  {'ours (1-shot fixed)':16s} unsafe   -   | gated {np.nanmean(o1):.3f} | q 1")
print(f"  {'ours (adaptive)':16s} unsafe   -   | gated {np.nanmean(oa):.3f} | q {np.nanmean(oq):.1f} (유해)")
print("\n주: ours는 재구성-게이트 보장(gated). 다른 방법 raw unsafe = 그들 논문 ASR 정의.")
print("    ours_1shot=고정설정 1쿼리(엄격 단발 비교), ours_adapt=온라인정책 ~2쿼리(warm-start+수정).")
Path("results/method_table_20260906.json").write_text(json.dumps(
    {"tags": tags, "methods": {m: {"unsafe": float(np.nanmean(agg[m]["u"])),
     "gated": float(np.nanmean(agg[m]["g"])), "queries": QUERIES[m]} for m in METHODS},
     "ours_1shot_gated": float(np.nanmean(o1)),
     "ours_adaptive_gated": float(np.nanmean(oa)), "ours_adaptive_queries": float(np.nanmean(oq))},
    indent=2) + "\n")
