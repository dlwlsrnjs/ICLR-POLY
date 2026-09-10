#!/usr/bin/env python3
"""Aggregate the harmful weak-language-vs-fixed panel into the paper's C3-language table.
Two adaptive rules, both benign-guided (no harmful supervision in the choice):
  (A) naive: for each n, EN + top-(n-1) benign-best languages vs EN + fixed order. Report mean Δ.
  (B) best-config: per target pick argmax over the (n, weak) grid by BENIGN decode-implied score,
      here approximated as the single benign-best language at n=2 (the benign profile's argmax),
      and compare its harmful joint ASR to the fixed policy's best single arm.
The honest paper number is the per-target adaptive (benign-chosen) minus the one fixed policy."""
import json, glob, os
from pathlib import Path
import numpy as np
OUT = Path("results/harmful_weaklang_20260904")
BEN = Path("results/benign_recon_sweep_20260904")

def benign_best_lang_n2(tag):
    d = json.loads((BEN / f"{tag}.json").read_text())
    langs = {c["name"][3:]: c["recon_pass"] for c in d["configs"]
             if c["name"].startswith("EN+") and c["n"] == 2 and "CJK" not in c["name"]}
    return max(langs, key=langs.get) if langs else None

rows = []
for f in sorted(glob.glob(str(OUT / "*.json"))):
    if os.path.basename(f) in ("aggregate.json",) or os.path.basename(f).startswith("panel_"): continue
    d = json.loads(open(f).read())
    t = d["target"]; R = {r["n"]: r for r in d["rows"]}
    # fixed policy = EN + fixed order, best n per target's fixed column (matched: same info fixed baseline uses)
    fixed_best_n = max(R, key=lambda n: R[n]["fixed_gated"])
    fixed_val = R[fixed_best_n]["fixed_gated"]
    # adaptive (benign-guided): benign profile says n=2 single-best language is the decode argmax
    adapt = R[2]  # n=2, EN + benign-best single language
    adapt_val = adapt["weak_gated"]
    naive_mean = float(np.mean([r["delta"] for r in d["rows"]]))
    rows.append(dict(target=t, weak_lang=adapt["weak_langs"][1] if len(adapt["weak_langs"]) > 1 else "?",
                     adaptive=adapt_val, fixed=fixed_val, delta=adapt_val - fixed_val,
                     naive_mean_delta=naive_mean,
                     n2=R[2]["delta"], n3=R[3]["delta"] if 3 in R else float("nan"),
                     n4=R[4]["delta"] if 4 in R else float("nan")))

rows.sort(key=lambda r: -r["delta"])
print(f"=== C3-language: 언어 적응(무해로 선택) vs 고정 언어, 실제 유해 공동 ASR ({len(rows)}대상) ===")
print(f"  {'target':24s} {'weak_lang':10s} {'적응(n2)':8s} {'고정best':8s} {'Δ':8s}  n2/n3/n4 Δ")
for r in rows:
    print(f"  {r['target']:24s} {r['weak_lang']:10s} {r['adaptive']:.3f}   {r['fixed']:.3f}   {r['delta']:+.3f}   "
          f"{r['n2']:+.2f}/{r['n3']:+.2f}/{r['n4']:+.2f}")
A = np.array([r["adaptive"] for r in rows]); F = np.array([r["fixed"] for r in rows])
print(f"\n  적응형 언어(무해선택) 평균 = {A.mean():.3f} | 고정 언어 평균 = {F.mean():.3f} | Δ = {A.mean()-F.mean():+.3f}")
wins = sum(1 for r in rows if r["delta"] > 0.001)
print(f"  대상 {len(rows)}개 중 {wins}개에서 언어 적응이 고정을 이김 (동률/패배 {len(rows)-wins})")
print(f"  naive(모든 n 평균) Δ = {np.mean([r['naive_mean_delta'] for r in rows]):+.3f}")
(OUT / "aggregate.json").write_text(json.dumps(dict(
    rows=rows, adaptive_mean=float(A.mean()), fixed_mean=float(F.mean()),
    delta=float(A.mean()-F.mean()), wins=wins, n=len(rows)), indent=2) + "\n")
print(f"\n저장: {OUT}/aggregate.json")
