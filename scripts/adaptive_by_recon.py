#!/usr/bin/env python3
"""Benign-driven adaptive: keep the resource-level language order (prior work) FIXED, and grow
the amount (add languages one at a time) AS LONG AS THE MODEL STILL SOLVES THE PUZZLE
(reconstruction rate >= tau). Stop when reconstruction collapses; pick the largest amount still
solvable. The stopping decision uses ONLY reconstruction (puzzle-solving) -- no harmful
supervision -- then we read off the harmful joint ASR at the chosen config to validate.

Compares, per target: adaptive-by-recon vs single fixed config vs oracle(best joint) and vs
adaptive-by-joint (harmful-supervised greedy) to show benign puzzle-solving drives adaptation
as well as harmful supervision does.
Reads results/sequential_resource_20260906/<tag>.json (steps carry gated/recon/unsafe per k)."""
import json, glob, os, sys
from pathlib import Path
import numpy as np

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/sequential_resource_20260906")
TAUS = [0.5, 0.6, 0.7]
PAT = 2   # patience: stop after `PAT` consecutive steps below tau

def load(P):
    D = {}
    for f in glob.glob(str(P / "*.json")):
        b = os.path.basename(f)
        if b.endswith("aggregate.json") or b.startswith("panel_"): continue
        d = json.loads(open(f).read()); D[d["target"]] = {s["k"]: s for s in d["steps"]}
    return D

def adaptive_recon(R, tau):
    """grow while recon>=tau; chosen = largest k with recon>=tau reached before stop."""
    ks = sorted(R); chosen = ks[0]; bad = 0
    for k in ks:
        if R[k]["recon"] >= tau:
            chosen = k; bad = 0
        else:
            bad += 1
            if bad >= PAT: break
    return chosen

def adaptive_joint(R, margin=0.05, pat=2, floor=0.4):
    ks = sorted(R); bk = ks[0]; best = R[ks[0]]["gated"]; bad = 0
    for k in ks[1:]:
        g = R[k]["gated"]
        if g > best: best = g; bk = k; bad = 0
        elif g < best - margin or R[k]["recon"] < floor:
            bad += 1
            if bad >= pat: break
        else: bad = 0
    return bk

def main():
    D = load(SRC)
    tags = sorted(D)
    order = json.loads(Path("results/lang_rank_20260905/resource_order.json").read_text())["order"]
    print(f"자원 순서(선행연구): {order}")
    print(f"대상 {len(tags)}: {tags}\n")
    # single fixed = best-on-average k
    kall = sorted({k for t in tags for k in D[t]})
    kavg = {k: np.mean([D[t][k]["gated"] for t in tags if k in D[t]]) for k in kall}
    fk = max(kavg, key=kavg.get)
    fixed = {t: D[t][fk]["gated"] for t in tags}
    oracle = {t: max(D[t][k]["gated"] for k in D[t]) for t in tags}
    joint_ad = {t: D[t][adaptive_joint(D[t])]["gated"] for t in tags}
    print(f"단일 고정 최선 k={fk}(n{fk+1}) 평균 J = {np.mean(list(fixed.values())):.3f}")
    print(f"오라클(최선 k) 평균 J = {np.mean(list(oracle.values())):.3f}")
    print(f"적응(유해 joint 구동) 평균 J = {np.mean(list(joint_ad.values())):.3f}\n")
    for tau in TAUS:
        rec_ad = {}
        for t in tags:
            k = adaptive_recon(D[t], tau)
            rec_ad[t] = D[t][k]["gated"]
        rj = np.array([rec_ad[t] for t in tags]); fj = np.array([fixed[t] for t in tags])
        oj = np.array([oracle[t] for t in tags]); jj = np.array([joint_ad[t] for t in tags])
        wins = int(np.sum(rj > fj + 1e-9))
        print(f"[recon τ={tau}] 적응(퍼즐해결 구동, 무해감독) 평균 J = {rj.mean():.3f} | "
              f"vs고정 {rj.mean()-fj.mean():+.3f} 이김 {wins}/{len(tags)} | "
              f"vs유해적응 {rj.mean()-jj.mean():+.3f} | regret(오라클) {oj.mean()-rj.mean():.3f}")
        if tau == 0.6:
            print("   per-target (τ=0.6): 선택 n / recon-adaptive J / 고정 J / 오라클")
            for t in tags:
                k = adaptive_recon(D[t], tau)
                print(f"     {t:24s} n={k+1:<3} J={D[t][k]['gated']:.3f} (recon {D[t][k]['recon']:.2f}) "
                      f"| 고정 {fixed[t]:.3f} | 오라클 {oracle[t]:.3f}")
    print("\n※ 정지 판단에 유해 라벨 미사용 — recon(퍼즐 해결)만으로 양을 적응. 유해 joint는 검증용.")

if __name__ == "__main__":
    main()
