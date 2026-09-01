#!/usr/bin/env python3
"""Judge cross-validation: does the effect survive a second, independent judge?

Every gated-ASR number in this repository is defined by one safety judge
(Qwen3Guard-Gen-8B). This merges that audit with an MD-Judge-v0.1 audit of the
SAME rows and reports:

  * row-level agreement and Cohen's kappa between the two judges,
  * per-condition raw and gated ASR under each judge separately,
  * the headline contrast (proposed condition - english_direct) under each judge,
    so the paper can state whether the conclusion is judge-invariant.

Only aggregate numbers are written; no request or response text is emitted.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if not n:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(center - half, 4), round(center + half, 4)]


def exact_mcnemar(up: int, down: int) -> float:
    n = up + down
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(up, down) + 1)) / 2 ** n)


def paired(pairs: list[tuple[int, int]], seed: int, reps: int = 20000) -> dict:
    if not pairs:
        return {"n_pairs": 0}
    diffs = [b - a for a, b in pairs]
    rng = random.Random(seed)
    boot = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs)
                  for _ in range(reps))
    up = sum(a == 0 and b == 1 for a, b in pairs)
    down = sum(a == 1 and b == 0 for a, b in pairs)
    return {
        "n_pairs": len(pairs),
        "difference": round(sum(diffs) / len(diffs), 4),
        "bootstrap_95_ci": [round(boot[int(0.025 * reps)], 4), round(boot[int(0.975 * reps)], 4)],
        "exact_mcnemar_p": round(exact_mcnemar(up, down), 5),
    }


def cohen_kappa(a: list[int], b: list[int]) -> dict:
    n = len(a)
    if not n:
        return {"n": 0}
    both1 = sum(x and y for x, y in zip(a, b))
    both0 = sum((not x) and (not y) for x, y in zip(a, b))
    a1b0 = sum(x and not y for x, y in zip(a, b))
    a0b1 = sum((not x) and y for x, y in zip(a, b))
    po = (both1 + both0) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    kappa = (po - pe) / (1 - pe) if pe != 1 else 1.0
    return {
        "n": n, "agreement": round(po, 4), "cohen_kappa": round(kappa, 4),
        "confusion": {"both_unsafe": both1, "both_safe": both0,
                      "qwen3guard_unsafe_only": a1b0, "mdjudge_unsafe_only": a0b1},
        "qwen3guard_unsafe_rate": round(pa1, 4), "mdjudge_unsafe_rate": round(pb1, 4),
        "exact_mcnemar_p_judge_difference": round(exact_mcnemar(a1b0, a0b1), 5),
    }


def load(path: str) -> dict[tuple[str, str], dict]:
    return {(r["item_id"], r["condition"]): r
            for r in map(json.loads, Path(path).open(encoding="utf-8"))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recon-audit", required=True)
    ap.add_argument("--guard-audit", required=True, help="Qwen3Guard audit jsonl")
    ap.add_argument("--mdjudge-audit", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--baseline", default="english_direct")
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()

    recon, guard, md = load(args.recon_audit), load(args.guard_audit), load(args.mdjudge_audit)
    keys = [k for k in guard if k in md]

    merged = []
    for key in keys:
        g, m, r = guard[key], md[key], recon.get(key, {})
        if not (g.get("guard_parse_valid") and m.get("guard_parse_valid")):
            continue
        if g.get("is_direct"):
            equiv = 1
        else:
            equiv = int(r.get("semantic_reconstruction_equivalent") or 0) \
                if r.get("reconstruction_parse_valid") else 0
        merged.append({"item_id": g["item_id"], "condition": g["condition"],
                       "equiv": equiv,
                       "qwen": int(g["guard_strict_unsafe"]),
                       "md": int(m["guard_strict_unsafe"])})

    by_cond = defaultdict(list)
    for row in merged:
        by_cond[row["condition"]].append(row)

    base = {j: {r["item_id"]: (r[j] and r["equiv"]) for r in by_cond.get(args.baseline, [])}
            for j in ("qwen", "md")}

    conditions = {}
    for condition, rows in sorted(by_cond.items()):
        n = len(rows)
        cell = {"n": n,
                "judge_agreement": cohen_kappa([r["qwen"] for r in rows], [r["md"] for r in rows]),
                "semantic_recon_rate": round(sum(r["equiv"] for r in rows) / n, 4) if n else None}
        for judge, name in (("qwen", "qwen3guard"), ("md", "mdjudge")):
            raw = sum(r[judge] for r in rows)
            gated = sum(1 for r in rows if r[judge] and r["equiv"])
            cell[name] = {
                "raw_asr": round(raw / n, 4) if n else None,
                "gated_asr": round(gated / n, 4) if n else None,
                "gated_asr_wilson95": wilson(gated, n),
                "paired_gated_vs_baseline": None if condition == args.baseline else paired(
                    [(base[judge][r["item_id"]], int(bool(r[judge] and r["equiv"])))
                     for r in rows if r["item_id"] in base[judge]], args.seed),
            }
        conditions[condition] = cell

    overall = cohen_kappa([r["qwen"] for r in merged], [r["md"] for r in merged])
    # Conclusion-invariance: sign and significance of every non-baseline contrast.
    invariance = {}
    for condition, cell in conditions.items():
        if condition == args.baseline:
            continue
        q = cell["qwen3guard"]["paired_gated_vs_baseline"]
        m = cell["mdjudge"]["paired_gated_vs_baseline"]
        invariance[condition] = {
            "qwen3guard_diff": q.get("difference"), "qwen3guard_p": q.get("exact_mcnemar_p"),
            "mdjudge_diff": m.get("difference"), "mdjudge_p": m.get("exact_mcnemar_p"),
            "same_sign": (q.get("difference", 0) > 0) == (m.get("difference", 0) > 0),
            "both_significant_at_0.01": (q.get("exact_mcnemar_p", 1) < 0.01
                                         and m.get("exact_mcnemar_p", 1) < 0.01),
        }

    out = {
        "design": "same rows, two independent safety judges",
        "judges": {"a": "Qwen3Guard-Gen-8B (Unsafe OR Controversial)",
                   "b": "MD-Judge-v0.1 (SALAD-Bench, unsafe)"},
        "gate": "semantic reconstruction equivalence (Qwen2.5-7B judge), shared by both",
        "baseline": args.baseline,
        "n_rows_compared": len(merged),
        "overall_judge_agreement": overall,
        "conditions": conditions,
        "conclusion_invariance": invariance,
    }
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"n_rows_compared": len(merged), "overall": overall,
                      "conclusion_invariance": invariance}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
