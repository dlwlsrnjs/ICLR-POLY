#!/usr/bin/env python3
"""Distil the L40S confirmatory result files into committable summaries.

The primary result files are 17-52MB because they carry every per-record replay row.  Only the
protocol block and the aggregated summary are needed to typeset the paper, so this script copies
those two pieces into experiments_suite/exp06_confirmatory_selector/results/*_summary.json.  The
full files stay in the private bucket under
hf://buckets/jin-kwon/poly/PolyJigsaw/0913/L40S-only/ (see docs/L40S_CODE_GAP_2026-09-15.md).

Source directory defaults to the local bucket download; override with POLY_L40S_RESULTS.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SOURCE = Path(os.environ.get(
    "POLY_L40S_RESULTS", "/home/ubuntu/342/jinkwon/poly/bucket_0913_priority"))
OUT = Path(__file__).resolve().parents[1] / "experiments_suite/exp06_confirmatory_selector/results"

CONFIRM = "advanced_additive_bo/confirmatory_20260914/exp04_budget_queryeff/results"
EXP04 = "exp04_budget_queryeff/results"

JOBS = {
    "itemheldout_advanced_bo_posterior_summary.json":
        f"{CONFIRM}/pilot_itemheldout_advanced_bo_posterior_l40s17_15seeds_20260914.json",
    "itemheldout_advanced_bo_summary.json":
        f"{CONFIRM}/pilot_itemheldout_advanced_bo_l40s17_15seeds_20260914.json",
    "multifidelity_itemcost_bo_summary.json":
        f"{CONFIRM}/pilot_multifidelity_itemcost_bo_l40s17_20260914.json",
    "family_loo_asr_selector_summary.json":
        f"{EXP04}/pilot_family_loo_asr_selector_l40s17_b16_20260914.json",
    "kernel_ablation_summary.json":
        f"{EXP04}/pilot_kernel_ablation_l40s17_b8_20260914.json",
}

KEEP = ("protocol", "summary", "oracle_family_macro", "method_roles")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    missing = []
    for name, relative in JOBS.items():
        path = SOURCE / relative
        if not path.exists():
            missing.append(str(path))
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        distilled = {k: payload[k] for k in KEEP if k in payload}
        distilled["source_artifact"] = Path(relative).name
        distilled["source_bucket"] = (
            "hf://buckets/jin-kwon/poly/PolyJigsaw/0913/L40S-only/" + str(Path(relative).parent))
        (OUT / name).write_text(json.dumps(distilled, indent=1), encoding="utf-8")
        methods = sorted({row.get("method", row.get("kernel")) for row in distilled.get("summary", [])})
        print(f"{name}: {(OUT / name).stat().st_size / 1000:.0f} kB, methods={methods}")
    for path in missing:
        print(f"MISSING (bucket only): {path}", file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
