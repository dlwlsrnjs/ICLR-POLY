#!/usr/bin/env python3
"""Consolidate the alignment probes into one score card per target.

A_over        over-refusal rate on the 250 SAFE XSTest prompts (benign; from
              score_overrefusal.py). High = the target refuses things that only sound unsafe.
A_direct      compliance on direct English harmful requests (Lingua dev sample), judged by
              the official Qwen3Guard recipe. High = weak English refusal barrier.
A_multi       the same requests in four non-English languages, official translations only.
              High = weak multilingual refusal barrier.
A_xstest      compliance on the 200 blunt UNSAFE XSTest prompts. A public-benchmark stand-in
              for the barrier, so the score can be looked up rather than measured.
gap           A_multi - A_direct, the English/multilingual alignment gap.
"""
from __future__ import annotations
import argparse, glob, json, os
from pathlib import Path

ROOT = "/home/ubuntu/342/jinkwon/poly/PolyJigsaw"
TAG2MODEL = {}

def guard_rate(path: str) -> tuple[float, int] | None:
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    if not rows:
        return None
    return sum(1 for r in rows if r.get("guard_strict_unsafe")) / len(rows), len(rows)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=f"{ROOT}/private_artifacts/alignment_probe")
    ap.add_argument("--overrefusal", default=f"{ROOT}/results/paper_overrefusal_probe.json")
    ap.add_argument("--out", default=f"{ROOT}/results/paper_alignment_probe.json")
    ap.add_argument("--tag-map", default=f"{ROOT}/results/panel_tag_map.json")
    a = ap.parse_args()

    tag_map = json.load(open(a.tag_map)) if os.path.exists(a.tag_map) else {}
    over = json.load(open(a.overrefusal)) if os.path.exists(a.overrefusal) else {}
    model2over = {m: v for m, v in over.items()}

    out = {}
    for d in sorted(glob.glob(f"{a.base}/*/")):
        tag = Path(d).name
        if tag == "audit":
            continue
        entry = {"tag": tag}
        model = tag_map.get(tag)
        # model name recovered from any probe output if the map has no entry
        for sub in ("over", "direct", "multi", "xsunsafe"):
            for fn in ("probe_outputs.jsonl", "restricted_probe_outputs.jsonl"):
                p = f"{d}{sub}/{fn}"
                if os.path.exists(p) and model is None:
                    model = json.loads(open(p, encoding="utf-8").readline())["target_model"]
        entry["model"] = model
        if model in model2over:
            entry["A_over_judge"] = model2over[model].get("A_over_judge")
            entry["A_over_rule"] = model2over[model].get("A_over_rule")
        for sub, key in (("direct", "A_direct"), ("multi", "A_multi"), ("xsunsafe", "A_xstest")):
            g = guard_rate(f"{d}{sub}/guard/restricted_qwen3guard_audit.jsonl")
            if g:
                entry[key] = round(g[0], 4)
                entry[f"n_{sub}"] = g[1]
        if "A_multi" in entry and "A_direct" in entry:
            entry["gap"] = round(entry["A_multi"] - entry["A_direct"], 4)
        out[tag] = entry

    json.dump(out, open(a.out, "w"), indent=2)
    hdr = ("tag", "A_over", "A_direct", "A_multi", "gap", "A_xstest")
    print("%-14s %8s %9s %8s %8s %9s" % hdr)
    for tag, e in sorted(out.items()):
        print("%-14s %8s %9s %8s %8s %9s" % (
            tag, e.get("A_over_judge", "-"), e.get("A_direct", "-"), e.get("A_multi", "-"),
            e.get("gap", "-"), e.get("A_xstest", "-")))
    print("\nsaved", a.out)

if __name__ == "__main__":
    main()
