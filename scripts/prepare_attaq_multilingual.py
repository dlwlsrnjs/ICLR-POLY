#!/usr/bin/env python3
"""Adapt the AttaQ-multilingual red-teaming dataset into our aligned schema.

Source: github.com/ellisalicante/multilingual-multimodal-red-teaming (MIT).
It ships one JSON list per language (aligned by index), each item being
{"label": category, "input": harmful prompt, ...}. We reuse the supplied
translations VERBATIM (no re-translation) and only emit our
{item_id, scenario, risk_type, original, questions:{Lang: text}} schema, which
the existing interleaving / slot pipeline consumes without modification.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# AttaQ language code -> full name used by our puzzle builder.
LANG_MAP = {
    "en": "English", "fr": "French", "es": "Spanish", "de": "German",
    "cs": "Czech", "sl": "Slovenian", "va": "Valencian",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attaq-dir", required=True, help="e.g. attaq/datasets/AttaQ_sample1")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    src = Path(args.attaq_dir)
    per_lang = {}
    for code, name in LANG_MAP.items():
        path = src / f"{code}.json"
        per_lang[name] = json.loads(path.read_text(encoding="utf-8"))
    counts = {name: len(items) for name, items in per_lang.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"language files misaligned: {counts}")
    n = counts["English"]

    rows = []
    for i in range(n):
        questions = {name: per_lang[name][i]["input"].strip() for name in LANG_MAP.values()}
        if any(not q for q in questions.values()):
            continue
        rows.append({
            "item_id": f"attaq__{src.name}__{i:04d}",
            "scenario": per_lang["English"][i].get("label", "unknown"),
            "risk_type": "attaq_text",
            "original": questions["English"],
            "questions": questions,
        })

    out = Path(args.output)
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(json.dumps({
        "n_items": len(rows), "languages": list(LANG_MAP.values()),
        "scenarios": dict(sorted(Counter(r["scenario"] for r in rows).items())),
        "translations": "reused verbatim from AttaQ-multilingual (NLLB/MT); not re-translated",
        "output": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
