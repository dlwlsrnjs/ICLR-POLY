#!/usr/bin/env python3
"""Rebuild the two panel input files from their public sources.

harm grid : the first N items of the frozen test split carried in the Text-Dominant
            Lingua-SafetyBench selection file itself (dev 2879 / validation 959 /
            test 960). Controlled text -> 0600.
benign probe : parallel FLORES-200 dev sentences in the same record schema, used to
            read a target's reconstruction capability without any harmful content.

Rebuilding from source lets a new target join the existing panel on exactly the same
items instead of a fresh sample, so panel numbers stay paired.
"""
from __future__ import annotations
import argparse, json, os, random
from pathlib import Path

FULL = "/home/ubuntu/342/jinkwon/datasets/lingua_safetybench_text/lingua_polyjigsaw_textdominant_full.jsonl"
FLORES = "/home/ubuntu/342/jinkwon/datasets/flores200/dev.parquet"
LANG_BY_CODE = {"arb_Arab": "Arabic", "zho_Hans": "Chinese", "eng_Latn": "English",
                "fin_Latn": "Finnish", "fra_Latn": "French", "deu_Latn": "German",
                "jpn_Jpan": "Japanese", "nob_Latn": "Norwegian", "rus_Cyrl": "Russian",
                "spa_Latn": "Spanish"}

def secure_write(path: Path, rows: list[dict]) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.chmod(path, 0o600)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-harm", type=int, default=250)
    ap.add_argument("--n-benign", type=int, default=40)
    ap.add_argument("--split", default="test", choices=("dev", "validation", "test"))
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(FULL, encoding="utf-8")]
    test_rows = [r for r in rows if r["split"] == a.split]
    harm = []
    for r in test_rows[: a.n_harm]:
        r = dict(r)
        r.setdefault("original", r["questions"]["English"])
        r.setdefault("language_assignment", [])
        harm.append(r)
    secure_write(out / "harm_grid.jsonl", harm)

    import pandas as pd
    df = pd.read_parquet(FLORES)
    benign = []
    for i in range(a.n_benign):
        row = df.iloc[i]
        q = {LANG_BY_CODE[c]: str(row[c]) for c in LANG_BY_CODE}
        benign.append({"item_id": f"flores_dev__{i:04d}", "scenario": "Benign Control",
                       "risk_type": "benign", "original": q["English"], "questions": q,
                       "image_paths": {}, "language_assignment": []})
    (out / "benign_probe.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in benign), encoding="utf-8")

    print(json.dumps({"harm_items": len(harm), "first_harm_id": harm[0]["item_id"],
                      "last_harm_id": harm[-1]["item_id"], "split": a.split,
                      "split_size": len(test_rows),
                      "benign_items": len(benign), "outdir": str(out)}, indent=2))

if __name__ == "__main__":
    main()
