#!/usr/bin/env python3
"""Paper-grade analysis: obfuscation config (language + puzzle complexity) must be tuned
ADAPTIVELY per target, and the tuning is readable from a BENIGN decode probe (no harmful
supervision).

Three claims, three tables:
  C1 (adaptation is needed): the benign-optimal language and complexity DIFFER across targets.
  C2 (benign predicts harmful): benign decode rate per config correlates with harmful joint
     ASR per config (transfer); per-target Spearman + pooled correlation.
  C3 (adaptive > fixed): selecting the harmful config from the benign profile per target
     beats the single best fixed config, on true harmful joint ASR.

Benign data: results/benign_recon_sweep_20260904/<tag>.json (per-language + complexity decode).
Harmful data: private_artifacts/frag_factorial_20260903/<tag>/harmful_summary.json (32-arm gated).
"""
from __future__ import annotations
import json, os, re, glob
from pathlib import Path
import numpy as np

BEN = Path("results/benign_recon_sweep_20260904")
FR = Path("private_artifacts/frag_factorial_20260903")
CELL = re.compile(r"frag(\d+)_(ordered|shuffled)_n(\d+)")


def harmful_gated(tag):
    p = FR / tag / "harmful_summary.json"
    if not p.exists():
        return {}
    return {k: v["gated"] for k, v in json.loads(p.read_text()).items() if CELL.match(k)}


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return np.nan
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    tags = sorted(os.path.basename(p)[:-5] for p in glob.glob(str(BEN / "*.json"))
                  if os.path.basename(p) not in ("PANEL_DONE.marker",))
    tags = [t for t in tags if (BEN / f"{t}.json").exists() and (FR / t / "harmful_summary.json").exists()]
    print(f"분석 대상 {len(tags)}개: {tags}\n")

    # ---- C1: benign-optimal language + complexity vary by target ----
    print("=== C1. 대상마다 최적 언어·복잡도가 다르다 (적응 필요) ===")
    print(f"  {'target':22s} {'best_lang(n2)':14s} {'best_n':6s} {'ordered>shuf?':12s}")
    langset, nset = set(), set()
    per = {}
    for t in tags:
        d = json.loads((BEN / f"{t}.json").read_text())
        cfg = {c["name"]: c for c in d["configs"]}
        langs = {c["name"][3:]: c["recon_pass"] for c in d["configs"]
                 if c["name"].startswith("EN+") and c["n"] == 2 and "CJK" not in c["name"]}
        best_lang = max(langs, key=langs.get) if langs else "?"
        nvals = {n: cfg.get(f"n{n}", cfg.get("n3", {})).get("recon_pass", np.nan) for n in (3, 4, 6, 8)}
        nvals[2] = np.mean([v for k, v in langs.items()]) if langs else np.nan
        best_n = max((n for n in nvals if not np.isnan(nvals[n])), key=lambda n: nvals[n], default="?")
        osd = cfg.get("shuffled_n4", {}).get("recon_pass", np.nan)
        ordv = cfg.get("n4", {}).get("recon_pass", np.nan)
        per[t] = dict(best_lang=best_lang, best_n=best_n, langs=langs)
        langset.add(best_lang); nset.add(best_n)
        print(f"  {t:22s} {best_lang:14s} n={best_n:<4} ordered {ordv:.2f} vs shuf {osd:.2f}")
    print(f"  => 최적 언어 종류 {len(langset)}개 {sorted(langset)}; 최적 언어수 종류 {sorted(nset)}")
    print(f"     (하나로 고정할 수 없음 = 적응이 필요하다는 직접 증거)\n")

    # ---- C2: benign decode predicts harmful gated (transfer) ----
    print("=== C2. 무해 해독이 유해 공동 ASR을 예측한다 (전이) ===")
    # match benign complexity configs to harmful frag5 arms by (n, arrangement)
    pooled_b, pooled_h, corrs = [], [], []
    for t in tags:
        d = {c["name"]: c for c in json.loads((BEN / f"{t}.json").read_text())["configs"]}
        H = harmful_gated(t)
        b, h = [], []
        for n in (3, 4, 6, 8):
            bk = f"n{n}"; hk = f"frag5_ordered_n{n}"
            if bk in d and hk in H:
                b.append(d[bk]["recon_pass"]); h.append(H[hk])
        for nm, hk in [("shuffled_n4", "frag5_shuffled_n4"), ("shuffled_n6", "frag5_shuffled_n6")]:
            if nm in d and hk in H:
                b.append(d[nm]["recon_pass"]); h.append(H[hk])
        if len(b) >= 3:
            corrs.append(spearman(b, h)); pooled_b += b; pooled_h += h
    print(f"  pooled corr(무해 해독, 유해 공동ASR) = {np.corrcoef(pooled_b, pooled_h)[0,1]:.2f} (n={len(pooled_b)})")
    print(f"  대상 내 평균 Spearman = {np.nanmean(corrs):.2f}\n")

    # ---- C3: adaptive (benign-guided) vs fixed, on true harmful gated ----
    print("=== C3. 적응형(무해로 선택) vs 고정, 실제 유해 공동 ASR ===")
    # candidate harmful arms restricted to those measurable from benign complexity axes
    arms = ["frag5_ordered_n2", "frag5_ordered_n4", "frag5_ordered_n6", "frag5_ordered_n8",
            "frag5_shuffled_n4", "frag5_shuffled_n6"]
    benkey = {"frag5_ordered_n2": "n?", "frag5_ordered_n4": "n4", "frag5_ordered_n6": "n6",
              "frag5_ordered_n8": "n8", "frag5_shuffled_n4": "shuffled_n4", "frag5_shuffled_n6": "shuffled_n6"}
    rows = []
    for t in tags:
        d = {c["name"]: c for c in json.loads((BEN / f"{t}.json").read_text())["configs"]}
        H = harmful_gated(t)
        avail = [a for a in arms if a in H]
        if len(avail) < 3:
            continue
        # benign decode per n (n2 = mean single-language decode; else the nX complexity config)
        langs = [c["recon_pass"] for c in json.loads((BEN/f'{t}.json').read_text())["configs"]
                 if c["name"].startswith("EN+") and c["n"] == 2 and "CJK" not in c["name"]]
        bdec = {2: float(np.mean(langs)) if langs else 0.0}
        for n in (4, 6, 8):
            bdec[n] = d.get(f"n{n}", {}).get("recon_pass", 0.0)
        # ADAPTIVE (frontier rule): most obfuscated (largest n) that still decodes >= tau.
        TAU = 0.5
        ordered_avail = [a for a in avail if a.startswith("frag5_ordered_n")]
        def arm_n(a): return int(a.split("_n")[-1])
        good = [a for a in ordered_avail if bdec.get(arm_n(a), 0) >= TAU]
        adaptive = max(good, key=arm_n) if good else min(ordered_avail, key=arm_n, default=avail[0])
        oracle = max(avail, key=lambda a: H[a])
        rows.append(dict(target=t, adaptive_arm=adaptive, adaptive=H[adaptive],
                         oracle_arm=oracle, oracle=H[oracle], avail=avail, H=H))
    # fixed = single arm best on average across targets
    common = set.intersection(*[set(r["avail"]) for r in rows]) if rows else set()
    fixed_arm = max(common, key=lambda a: np.mean([r["H"][a] for r in rows])) if common else None
    ad = np.array([r["adaptive"] for r in rows]); orc = np.array([r["oracle"] for r in rows])
    fx = np.array([r["H"][fixed_arm] for r in rows]) if fixed_arm else np.array([np.nan])
    print(f"  대상 {len(rows)}개 | 고정 최선 arm = {fixed_arm}")
    print(f"  적응형(무해선택) 유해 공동ASR = {ad.mean():.3f} | 고정 = {fx.mean():.3f} | 오라클 = {orc.mean():.3f}")
    print(f"  적응형 - 고정 = {ad.mean()-fx.mean():+.3f} | 적응형 regret = {(orc-ad).mean():.3f}")
    print("\n  per-target: 적응형이 고른 arm / 유해ASR vs 고정 vs 오라클")
    for r in rows:
        print(f"    {r['target']:22s} 적응 {r['adaptive_arm']:18s}{r['adaptive']:.2f} | 고정 {r['H'].get(fixed_arm,float('nan')):.2f} | 오라클 {r['oracle']:.2f}")

    out = Path("results/adaptive_benign_20260904"); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(
        targets=tags, C1_per=per, C1_distinct_langs=sorted(langset), C1_distinct_n=sorted(nset),
        C2_pooled_corr=float(np.corrcoef(pooled_b, pooled_h)[0,1]), C2_mean_within=float(np.nanmean(corrs)),
        C3_fixed_arm=fixed_arm, C3_adaptive=float(ad.mean()), C3_fixed=float(fx.mean()),
        C3_oracle=float(orc.mean()), C3_rows=[{k: r[k] for k in ('target','adaptive_arm','adaptive','oracle_arm','oracle')} for r in rows]),
        indent=2, default=str) + "\n")
    print(f"\n저장: {out}/summary.json")


if __name__ == "__main__":
    main()
