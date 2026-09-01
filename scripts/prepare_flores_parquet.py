#!/usr/bin/env python3
"""Materialize a local FLORES-200 parquet with the ten benign-control columns.

The interleaving probe expects a Parquet file whose columns are FLORES language
codes (``eng_Latn``, ``zho_Hans``, ...). This downloads an official FLORES split
via ``datasets`` and writes only the ten columns the probe uses. FLORES is a
benign parallel-sentence corpus and is used purely as a false-positive control.
"""

from __future__ import annotations

import argparse
from pathlib import Path

CODES = [
    "arb_Arab", "zho_Hans", "eng_Latn", "fin_Latn", "fra_Latn",
    "deu_Latn", "jpn_Jpan", "nob_Latn", "rus_Cyrl", "spa_Latn",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-name", default="facebook/flores")
    parser.add_argument("--config", default="all")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--output", type=Path, default=Path("../datasets/flores200/dev.parquet"))
    args = parser.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import load_dataset

    dataset = load_dataset(args.hf_name, args.config, split=args.split)
    columns = dataset.column_names

    def resolve(code: str) -> str:
        for candidate in (code, f"sentence_{code}"):
            if candidate in columns:
                return candidate
        raise KeyError(f"FLORES column for {code} not found; available sample: {columns[:6]}")

    resolved = {code: resolve(code) for code in CODES}
    table = pa.table({code: dataset[source] for code, source in resolved.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output)
    print(f"wrote {table.num_rows} rows x {table.num_columns} cols -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
