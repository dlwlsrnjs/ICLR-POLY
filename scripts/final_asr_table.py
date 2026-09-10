#!/usr/bin/env python3
"""Assemble the definitive ASR comparison table from all collected results (CPU only):
  english_direct < translated_direct < fixed PolyJigsaw < adaptive(amount) < adaptive(amount+disorder)
plus the nogame ablation. Per model + panel means. Run after BASELINES_DONE."""
import json, glob, os
from pathlib import Path
import numpy as np

def load_steps(P):
    D = {}
    for f in glob.glob(str(Path(P) / "*.json")):
        b = os.path.basename(f)
        if b.endswith("_20260906.json") or b.startswith("panel_") or b.endswith("aggregate.json"):
            continue
        d = json.loads(open(f).read())
        if "steps" in d:
            D[d["target"]] = {s["k"]: s for s in d["steps"]}
    return D

def load_flat(P, keyset):
    D = {}
    for f in glob.glob(str(Path(P) / "*.json")):
        b = os.path.basename(f)
        if b.startswith("panel_"):
            continue
        d = json.loads(open(f).read())
        if "target" in d and any(k in d for k in keyset):
            D[d["target"]] = d
    return D

SEQ = load_steps("results/sequential_resource_20260906")
DIS = {}
for f in glob.glob("results/disorder_sweep_20260906/*.json"):
    b = os.path.basename(f)
    if b.startswith("panel_"): continue
    d = json.loads(open(f).read()); DIS[d["target"]] = {r["delta"]: r["gated"] for r in d["rows"]}
BASE = load_flat("results/baselines_20260906", ["english_direct"])
GAME = load_flat("results/game_ablation_20260906", ["game"])

tags = sorted(SEQ)
fk = None
if SEQ:
    kall = sorted({k for t in tags for k in SEQ[t]})
    kavg = {k: np.mean([SEQ[t][k]["gated"] for t in tags if k in SEQ[t]]) for k in kall}
    fk = max(kavg, key=kavg.get)

print(f"{'model':22s}{'engDir':8s}{'transDir':9s}{'fixed':7s}{'adaptN':8s}{'adapt+δ':8s}{'nogame':8s}")
cols = {k: [] for k in ["ed", "td", "fx", "an", "ad", "ng"]}
for t in tags:
    R = SEQ[t]
    ed = BASE.get(t, {}).get("english_direct", np.nan)
    td = BASE.get(t, {}).get("translated_direct", np.nan)
    fx = R[fk]["gated"] if fk in R else np.nan
    an = max(R[k]["gated"] for k in R)                      # adaptive amount (oracle-of-amount ~ policy target)
    dis = max(DIS.get(t, {0: np.nan}).values()) if t in DIS else np.nan
    ad = np.nanmax([an, dis])                               # adaptive amount+disorder
    ng = GAME.get(t, {}).get("nogame", {}).get("gated", np.nan) if t in GAME else np.nan
    for k, v in zip(cols, [ed, td, fx, an, ad, ng]):
        cols[k].append(v)
    def s(x): return f"{x:.3f}" if not (isinstance(x, float) and np.isnan(x)) else "  -  "
    print(f"{t:22s}{s(ed):8s}{s(td):9s}{s(fx):7s}{s(an):8s}{s(ad):8s}{s(ng):8s}")
print("\n=== panel means ===")
lab = {"ed": "english_direct", "td": "translated_direct", "fx": f"fixed(n{fk+1})",
       "an": "adaptive(amount)", "ad": "adaptive(amount+δ)", "ng": "nogame"}
for k in cols:
    v = np.array(cols[k], float)
    print(f"  {lab[k]:22s} {np.nanmean(v):.3f}  (n={int(np.sum(~np.isnan(v)))})")
Path("results/final_asr_table_20260906.json").write_text(json.dumps(
    {"tags": tags, "fixed_k": fk, "cols": {k: cols[k] for k in cols}}, indent=2, default=str) + "\n")
print("\n저장: results/final_asr_table_20260906.json")
