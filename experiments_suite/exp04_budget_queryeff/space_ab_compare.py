#!/usr/bin/env python3
"""Decide the arm space: A = comprehension(F) x language(n) x arrangement (grid, no willingness) vs
B = A + WEAK willingness (single frames plain/persona/fiction per cell, no stacks/pap/role).
Both are subsets of the collected 292 matrix, so no extra collection is needed. For each model x
collection we report, restricted to each space: oracle, benign-only selection (prior argmax -> verified,
the stealthy free mode), and harmful adaptive search @budget. Tagged held-in / held-out.

Winner criterion per mode:
  stealth mode  -> higher benign-only achieved verified (no harmful target queries)
  perf mode     -> higher adaptive-search verified
Usage: python space_ab_compare.py --root <exp02 results> [--suffix _mj|_lg|both] [--budget 8]"""
import argparse, json, glob, re
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import search_sota as S
import adaptive_subspace as A

CELL = re.compile(r"g(\d+)_(ordered|shuffled)_n(\d+)__(.+)")
HELD_IN = {"qwen25_3b", "qwen25_7b", "qwen25_14b", "qwen25_32b", "llama32_3b_it",
           "llama31_8b_it", "gemma2_2b_it", "gemma2_9b_it", "gemma2_27b"}
SPACES = {"A_comp_lang_order": {"plain"},
          "B_weak_will": {"plain", "persona", "fiction"},
          "C_medium_will": {"plain", "persona", "fiction", "pap", "persona+fiction"},
          "D_full_stacks": None}   # None = all willingness (full 292 grid arms)


def in_space(arm, wills):
    m = CELL.match(arm)
    return bool(m) and (wills is None or m.group(4) in wills)


def bench(root, suffix):
    ver_all, pri = S.load(root, suffix)          # saturation-aware prior
    rows = []
    for tag in sorted(ver_all):
        full = ver_all[tag]
        gridC = [a for a in full if CELL.match(a) and CELL.match(a).group(4) in {"plain","persona","fiction","pap","persona+fiction"}]
        grid_all = [a for a in full if CELL.match(a)]
        if len(gridC) < 160:            # need space C complete to compare A/B/C
            continue
        has_full = len(grid_all) >= 288  # D (all stacks) only if fully collected
        base = tag[:-3] if tag.endswith(("_mj", "_lg")) else tag
        held = "held_in" if base in HELD_IN else "held_out"
        rec = {"tag": tag, "held": held}
        for sname, wills in SPACES.items():
            if sname == "D_full_stacks" and not has_full:
                continue
            arms = [a for a in full if in_space(a, wills)]
            ver = {a: full[a] for a in arms}
            prior = {a: pri[tag].get(a, 0.5) for a in arms}
            oracle = max(ver.values())
            bo_arm = max(prior, key=prior.get); benign_only = ver[bo_arm]
            X = np.array([A.feats8(a) for a in arms])
            best, _ = A.adaptive_run(arms, X, ver, prior, 8, 2.0, np.random.default_rng(0))
            # avg adaptive over seeds
            adap = np.mean([A.adaptive_run(arms, X, ver, prior, 8, 2.0, np.random.default_rng(s))[0] for s in range(12)])
            rec[sname] = dict(n_arms=len(arms), oracle=round(oracle, 3),
                              benign_only=round(benign_only, 3), adaptive8=round(float(adap), 3))
        rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True); ap.add_argument("--suffix", default="both")
    a = ap.parse_args()
    sufs = ["_mj", "_lg"] if a.suffix == "both" else [a.suffix]
    allrows = []
    for suf in sufs:
        allrows += bench(a.root, suf)
    for r in allrows:
        print(f"\n[{r['tag']}] ({r['held']})")
        for s in SPACES:
            if s not in r: continue
            x = r[s]
            print(f"  {s:20} n={x['n_arms']:3}  oracle={x['oracle']}  benign_only={x['benign_only']}  adaptive@8={x['adaptive8']}")
    # per-space mean over cells (held-in / held-out split)
    import collections as _c
    for held in ("held_in", "held_out"):
        cells = [r for r in allrows if r["held"] == held]
        if not cells: continue
        print(f"\n=== {held} ({len(cells)} cells) mean over cells ===")
        for s in SPACES:
            for mode in ("oracle", "benign_only", "adaptive8"):
                pass
            cc=[r for r in cells if s in r]
            if not cc: continue
            o=np.mean([r[s]["oracle"] for r in cc]); bo=np.mean([r[s]["benign_only"] for r in cc]); ad=np.mean([r[s]["adaptive8"] for r in cc])
            print(f"  {s:20} oracle={o:.3f}  benign_only={bo:.3f}  adaptive@8={ad:.3f}")
    Path(a.root, "space_ab_compare.json").write_text(json.dumps(allrows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
