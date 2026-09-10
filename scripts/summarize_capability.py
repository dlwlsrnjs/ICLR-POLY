#!/usr/bin/env python3
"""Consolidate the capability-decomposition probe into one score card per target.

C_en      semantic reconstruction rate on the one-language (English) reassembly cells,
          averaged over ordered and shuffled. Pure puzzle ability, no other language.
M_<lang>  rate at which the target renders a single FLORES sentence in that language into
          English with the meaning preserved. Per-language reading ability.
M_mean    mean of M over the nine non-English languages.
M_min     the weakest language, which is what a multilingual puzzle is bottlenecked by.
C_multi   the multilingual interleaved reconstruction rate already measured by the panel
          benign probe, averaged over the ten grid cells.
cap_gap   C_en - C_multi, the multilingual capability deficit.
"""
from __future__ import annotations
import argparse, glob, json, os
from collections import defaultdict
from pathlib import Path

ROOT = "/home/ubuntu/342/jinkwon/poly/PolyJigsaw"

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=f"{ROOT}/private_artifacts/capability_probe")
    ap.add_argument("--panel", default=f"{ROOT}/private_artifacts/panel_v2")
    ap.add_argument("--out", default=f"{ROOT}/results/paper_capability_probe_decomp.json")
    a = ap.parse_args()

    out = {}
    for d in sorted(glob.glob(f"{a.base}/*/")):
        tag = Path(d).name
        f = f"{d}judge/restricted_reconstruction_audit.jsonl"
        if not os.path.exists(f):
            continue
        agg = defaultdict(lambda: [0, 0])
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            v = r.get("semantic_reconstruction_equivalent")
            if v == "":
                continue
            agg[r["condition"]][0] += 1
            agg[r["condition"]][1] += int(v)
        rates = {c: v[1] / v[0] for c, v in agg.items() if v[0]}
        entry = {"tag": tag}
        en = [rates[c] for c in ("interleave_ordered_n1", "interleave_shuffled_n1") if c in rates]
        if en:
            entry["C_en"] = round(sum(en) / len(en), 4)
            for c in ("interleave_ordered_n1", "interleave_shuffled_n1"):
                if c in rates:
                    entry[c] = round(rates[c], 4)
        langs = {c[len("langcomp_"):]: v for c, v in rates.items() if c.startswith("langcomp_")}
        if langs:
            entry["M"] = {k: round(v, 4) for k, v in sorted(langs.items())}
            entry["M_mean"] = round(sum(langs.values()) / len(langs), 4)
            entry["M_min"] = round(min(langs.values()), 4)
        bp = f"{a.panel}/{tag}/benign_recon.json"
        if os.path.exists(bp):
            b = json.load(open(bp))
            cells = [v for k, v in b.items() if k.startswith("interleave_")]
            if cells:
                entry["C_multi"] = round(sum(cells) / len(cells), 4)
                if "C_en" in entry:
                    entry["cap_gap"] = round(entry["C_en"] - entry["C_multi"], 4)
        out[tag] = entry

    json.dump(out, open(a.out, "w"), indent=2)
    print("%-14s %7s %8s %8s %8s %8s" % ("tag", "C_en", "C_multi", "cap_gap", "M_mean", "M_min"))
    for tag, e in sorted(out.items(), key=lambda kv: kv[1].get("C_multi", 0)):
        print("%-14s %7s %8s %8s %8s %8s" % (tag, e.get("C_en", "-"), e.get("C_multi", "-"),
                                             e.get("cap_gap", "-"), e.get("M_mean", "-"), e.get("M_min", "-")))
    print("\nsaved", a.out)

if __name__ == "__main__":
    main()
