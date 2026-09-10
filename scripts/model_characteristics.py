#!/usr/bin/env python3
"""Model-characteristic 2D analysis: place each model in the (decode capability C, alignment
weakness A) plane and identify which obfuscation axis is the lever. Uses the resource-order
sequential data (recon/unsafe/gated per amount step) and, when present, the disorder sweep.
  C  = decode capability   = recon at n=2 (easiest puzzle solving).
  A  = alignment weakness   = raw unsafe U at n=2 (readiness to produce unsafe content).
  amount_gain = max_n joint - joint@n2   (does growing the amount help?).
  disorder_gain = max_delta joint - joint@delta0 (from disorder sweep, if available)."""
import json, glob, os
from pathlib import Path
import numpy as np
RS = Path("results/sequential_resource_20260906")
DS = Path("results/disorder_sweep_20260906")

def load_seq(P):
    D = {}
    for f in glob.glob(str(P / "*.json")):
        b = os.path.basename(f)
        if b.endswith("aggregate.json") or b.startswith("panel_"): continue
        d = json.loads(open(f).read()); D[d["target"]] = {s["k"]: s for s in d["steps"]}
    return D

def load_dis(P):
    D = {}
    for f in glob.glob(str(P / "*.json")):
        b = os.path.basename(f)
        if b.startswith("panel_"): continue
        d = json.loads(open(f).read()); D[d["target"]] = {r["delta"]: r for r in d["rows"]}
    return D

SEQ = load_seq(RS); DIS = load_dis(DS)
FAM = {"qwen25_3b": "Qwen2.5/3B", "qwen25_7b": "Qwen2.5/7B", "qwen25_14b": "Qwen2.5/14B",
       "llama32_3b_it": "Llama3/3B", "llama31_8b_it": "Llama3/8B",
       "gemma2_2b_it": "Gemma2/2B", "gemma2_9b_it": "Gemma2/9B",
       "phi35": "Phi3.5", "mistral7b": "Mistral7B", "granite_3_3_8b_instruct": "Granite8B",
       "yi_1_5_6b_chat": "Yi6B", "qwen3_4b": "Qwen3/4B"}
rows = []
for t, R in SEQ.items():
    C = R[1]["recon"]; A = R[1]["unsafe"]
    j2 = R[1]["gated"]; jmax = max(R[k]["gated"] for k in R); bestn = max(R, key=lambda k: R[k]["gated"]) + 1
    amt_gain = jmax - j2
    dg = float("nan"); dn = 0
    if t in DIS:
        d0 = DIS[t].get(0.0, {}).get("gated", np.nan)
        dm = max(v["gated"] for v in DIS[t].values())
        dg = dm - d0
    rows.append(dict(t=t, name=FAM.get(t, t), C=C, A=A, j2=j2, jmax=jmax, bestn=bestn, amt_gain=amt_gain, dis_gain=dg))

def quad(C, A):
    c = "hi" if C >= 0.85 else "lo"; a = "weak" if A >= 0.55 else "strong"
    return f"decode-{c}/align-{a}"

rows.sort(key=lambda r: (-r["C"], -r["A"]))
print(f"{'model':14s}{'C(recon@n2)':12s}{'A(unsafe@n2)':13s}{'quadrant':24s}{'best_n':7s}{'amt_gain':9s}{'dis_gain'}")
for r in rows:
    dgs = f"{r['dis_gain']:+.3f}" if not np.isnan(r['dis_gain']) else "  -  "
    print(f"{r['name']:14s}{r['C']:.2f}        {r['A']:.2f}         {quad(r['C'],r['A']):24s}n={r['bestn']:<5}{r['amt_gain']:+.3f}   {dgs}")

print("\n=== 사분면 요약 (어느 축이 지렛대인가) ===")
buckets = {}
for r in rows:
    buckets.setdefault(quad(r["C"], r["A"]), []).append(r)
for q, rs in sorted(buckets.items()):
    ag = np.mean([x["amt_gain"] for x in rs])
    print(f"  {q:24s} n={len(rs)} | 평균 양이득 {ag:+.3f} | 모델 {[x['name'] for x in rs]}")
print("\n해석:")
print("  decode-hi/align-strong = 정렬-병목 (14B류): 난독화로 못 뚫음, 양·혼란도 이득 작음.")
print("  decode-hi/align-weak   = 이상적 표적: 재구성 여유 → 양/혼란도로 회피 올릴 여지.")
print("  decode-lo/align-*      = 재구성-병목: 양↑/혼란도↑가 오히려 해로움, n=2 최선.")
(RS.parent / "model_characteristics_20260906.json").write_text(json.dumps(rows, indent=2, default=str) + "\n")
