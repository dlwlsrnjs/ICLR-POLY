#!/usr/bin/env python3
"""Combined table across all target/engine configs (harm-free aggregates).

Aligns condition names (strips the vl_ prefix so text and VL configs line up) and
prints gated ASR per canonical condition per config, so we can read off both
text-vs-VL and HF-vs-vLLM effects.
"""

from __future__ import annotations

import json
from pathlib import Path

CONFIGS = [
    ("A_hf  (text, HF)",       "results/full_textdom_textmodel_hf_summary.json",   "english_direct"),
    ("A_vllm(text, vLLM)",     "results/full_textdom_textmodel_vllm_summary.json", "english_direct"),
    ("B_vllm(VL real img)",    "results/full_textdom_vlreal_summary.json",         "vl_english_direct"),
    ("C_vllm(VL blank img)",   "results/full_textdom_vlblank_summary.json",        "vl_english_direct"),
]
CANON = ["english_direct", "translated_direct_Finnish",
         "interleave_ordered_n4", "interleave_ordered_n10",
         "slot_k1", "slot_k2", "slot_k3"]


def canon(cond: str) -> str:
    return cond[3:] if cond.startswith("vl_") else cond


def main() -> None:
    loaded = []
    for label, path, base in CONFIGS:
        p = Path(path)
        if not p.exists():
            loaded.append((label, None))
            continue
        d = json.load(p.open())
        by_canon = {canon(k): v for k, v in d["conditions"].items()}
        loaded.append((label, by_canon))

    combined = {"metric": "reconstruction-gated ASR (semantic-equiv AND Qwen3Guard strict-unsafe)",
                "note": "text configs harm-in-text; VL configs same items with/without image. "
                        "engine column lets HF vs vLLM be compared for the text model.",
                "configs": {}, "table": {}}

    # Text table
    header = f"{'condition':<26}" + "".join(f"{lbl:<22}" for lbl, _ in loaded)
    lines = ["GATED ASR by condition x config  (n in parens)", header]
    for cond in CANON:
        cells = []
        for label, by in loaded:
            if by is None or cond not in by:
                cells.append(f"{'—':<22}")
            else:
                v = by[cond]
                cells.append(f"{v['gated_asr']:.3f} (n={v['n']})".ljust(22))
        lines.append(f"{cond:<26}" + "".join(cells))
    table_text = "\n".join(lines)

    for label, by in loaded:
        combined["configs"][label] = ("pending" if by is None else
                                      {c: by[c]["gated_asr"] for c in CANON if c in by})
    combined["table"]["gated_asr"] = {
        cond: {label: (None if by is None or cond not in by else by[cond]["gated_asr"])
               for label, by in loaded}
        for cond in CANON
    }
    Path("results/full_textdom_ALL_configs_summary.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n")
    print(table_text)
    print("\nwrote results/full_textdom_ALL_configs_summary.json")


if __name__ == "__main__":
    main()
