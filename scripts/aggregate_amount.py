#!/usr/bin/env python3
"""Aggregate the amount-scaling sweep: does increasing the number of multilingual translations
(with the target's best-decoded languages) raise unsafe/joint ASR? Reports per-target curves
and the pooled trend of raw-unsafe and joint vs n."""
import json, glob, os
from pathlib import Path
import numpy as np
OUT = Path("results/harmful_amount_20260905")
NS = (2, 3, 4, 6, 8, 10)
data = {}
for f in glob.glob(str(OUT / "*.json")):
    b = os.path.basename(f)
    if b in ("aggregate.json",) or b.startswith("panel_"): continue
    d = json.loads(open(f).read()); data[d["target"]] = {r["n"]: r for r in d["rows"]}
tags = sorted(data)
print(f"=== 무해 텍스트 양(번역본 수 n) 스케일링, {len(tags)}대상 (weak=대상별 최강해독 언어) ===\n")
# per-target: unsafe and gated at each n
for t in tags:
    R = data[t]
    us = " ".join(f"n{n}:{R[n]['weak_unsafe']:.2f}" for n in NS if n in R)
    js = " ".join(f"{R[n]['weak_gated']:.2f}" for n in NS if n in R)
    ns_present = [n for n in NS if n in R]
    u0 = R[ns_present[0]]["weak_unsafe"]; uL = R[ns_present[-1]]["weak_unsafe"]
    print(f"  {t:24s} unsafe {us}   Δunsafe(끝-처음) {uL-u0:+.2f}")
# pooled trend
print("\n  pooled 평균 (대상 평균):")
print(f"  {'n':>3} {'recon':>7} {'unsafe':>7} {'gated(weak)':>12} {'gated(fixed)':>13}")
best_per_target = {}
for n in NS:
    rc = [data[t][n]["weak_recon"] for t in tags if n in data[t]]
    us = [data[t][n]["weak_unsafe"] for t in tags if n in data[t]]
    jw = [data[t][n]["weak_gated"] for t in tags if n in data[t]]
    jf = [data[t][n]["fixed_gated"] for t in tags if n in data[t]]
    if not rc: continue
    print(f"  {n:>3} {np.mean(rc):7.3f} {np.mean(us):7.3f} {np.mean(jw):12.3f} {np.mean(jf):13.3f}")
# unsafe slope vs n (pooled)
xs, ys = [], []
for t in tags:
    for n in NS:
        if n in data[t]:
            xs.append(n); ys.append(data[t][n]["weak_unsafe"])
xs, ys = np.array(xs, float), np.array(ys, float)
slope = np.polyfit(xs, ys, 1)[0]
print(f"\n  raw-unsafe vs n 기울기(pooled) = {slope:+.4f}/언어  corr={np.corrcoef(xs,ys)[0,1]:+.2f}")
# best amount per target vs the small-n baseline
gains = []
for t in tags:
    R = data[t]; ns = [n for n in NS if n in R]
    base = R[ns[0]]["weak_gated"]
    bestn = max(ns, key=lambda n: R[n]["weak_gated"])
    gains.append(R[bestn]["weak_gated"] - base)
    best_per_target[t] = (bestn, R[bestn]["weak_gated"], base)
print(f"\n  대상별 최적 양으로 올린 joint 이득(vs n2 weak) 평균 = {np.mean(gains):+.3f}")
for t in tags:
    bn, bv, base = best_per_target[t]
    print(f"    {t:24s} 최적 n={bn:<2} joint {bv:.3f} (n2 {base:.3f}, Δ{bv-base:+.3f})")
(OUT / "aggregate.json").write_text(json.dumps(dict(tags=tags, unsafe_slope=float(slope),
    best_per_target={t: dict(n=best_per_target[t][0], gated=best_per_target[t][1]) for t in tags}),
    indent=2) + "\n")
print(f"\n저장: {OUT}/aggregate.json")
