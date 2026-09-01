#!/usr/bin/env python3
"""Fold a gated-ASR condition summary into the language-load x ordering matrix.

summarize_polyjig_comparison.py emits one entry per condition name; this pivots
`interleave_<ordering>_n<K>` into a matrix over K and ordering, keeps the
baselines alongside, and names the peak cell (the "sweet spot" the reports ask
for) together with the ordered-minus-shuffled gap at each K.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", required=True, help="output of summarize_polyjig_comparison.py")
    ap.add_argument("--output", required=True)
    ap.add_argument("--baseline", default="english_direct")
    args = ap.parse_args()

    data = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    conds = data["conditions"]

    matrix: dict[str, dict[int, dict]] = {}
    for name, cell in conds.items():
        match = re.fullmatch(r"interleave_(ordered|shuffled)_n(\d+)", name)
        if not match:
            continue
        ordering, k = match.group(1), int(match.group(2))
        matrix.setdefault(ordering, {})[k] = {
            "n": cell["n"],
            "raw_asr": cell["raw_asr"],
            "semantic_recon_rate": cell["semantic_recon_rate"],
            "gated_asr": cell["gated_asr"],
            "gated_asr_wilson95": cell["gated_asr_wilson95"],
            "diff_vs_baseline": (cell.get("paired_gated_vs_baseline") or {}).get("difference"),
            "exact_mcnemar_p": (cell.get("paired_gated_vs_baseline") or {}).get("exact_mcnemar_p"),
        }

    loads = sorted({k for row in matrix.values() for k in row})
    ordering_gap = {
        k: round(matrix["ordered"][k]["gated_asr"] - matrix["shuffled"][k]["gated_asr"], 4)
        for k in loads
        if "ordered" in matrix and "shuffled" in matrix
        and k in matrix["ordered"] and k in matrix["shuffled"]
    }

    cells = [(o, k, c["gated_asr"]) for o, row in matrix.items() for k, c in row.items()
             if c["gated_asr"] is not None]
    peak = max(cells, key=lambda t: t[2]) if cells else None

    baselines = {name: {"gated_asr": cell["gated_asr"], "n": cell["n"]}
                 for name, cell in conds.items()
                 if not name.startswith("interleave_")}

    out = {
        "source_summary": args.summary,
        "metric": "reconstruction-gated ASR (semantic equivalence AND strict-unsafe)",
        "baselines": baselines,
        "language_loads": loads,
        "matrix": matrix,
        "ordered_minus_shuffled_gated_gap": ordering_gap,
        "peak_cell": ({"ordering": peak[0], "num_languages": peak[1], "gated_asr": peak[2]}
                      if peak else None),
        "recon_collapse_check": {
            ordering: {k: row[k]["semantic_recon_rate"] for k in sorted(row)}
            for ordering, row in matrix.items()
        },
    }
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")

    # Readable table for the terminal / report.
    base = conds.get(args.baseline, {}).get("gated_asr")
    print(f"baseline {args.baseline}: gated ASR {base}")
    header = "  n  " + "".join(f"{o:>22s}" for o in sorted(matrix))
    print(header)
    for k in loads:
        row = f"{k:>3d}  "
        for ordering in sorted(matrix):
            cell = matrix[ordering].get(k)
            row += (f"{cell['gated_asr']:.4f} (rec {cell['semantic_recon_rate']:.2f})".rjust(22)
                    if cell else "".rjust(22))
        print(row)
    print("ordered - shuffled gap:", ordering_gap)
    print("peak:", out["peak_cell"])


if __name__ == "__main__":
    main()
