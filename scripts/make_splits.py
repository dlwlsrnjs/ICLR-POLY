#!/usr/bin/env python3
"""Deterministic stratified dev/test split for config-freeze discipline.

No model training happens anywhere in this project; this split only isolates
hyper-parameter selection from final reporting. Hyper-parameters (best language
load n, slot k, defense prompt) are chosen looking ONLY at dev items; the paper's
main comparison tables are reported on the held-out test items, which are never
used for any selection. Split is stratified by scenario/category and seeded, so
it is reproducible and balanced.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="jsonl with item_id + scenario")
    ap.add_argument("--output", required=True, help="jsonl {item_id, scenario, split}")
    ap.add_argument("--test-frac", type=float, default=0.6,
                    help="fraction held out for final reporting (default 0.6)")
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.data).open(encoding="utf-8")]
    by_scenario = defaultdict(list)
    for r in rows:
        by_scenario[r.get("scenario", "?")].append(r["item_id"])

    rng = random.Random(args.seed)
    assignment = {}
    counts = {"dev": 0, "test": 0}
    for scenario, ids in sorted(by_scenario.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        n_test = round(len(ids) * args.test_frac)
        for i, item_id in enumerate(ids):
            split = "test" if i < n_test else "dev"
            assignment[item_id] = (scenario, split)
            counts[split] += 1

    out = Path(args.output)
    with out.open("w", encoding="utf-8") as f:
        for item_id, (scenario, split) in sorted(assignment.items()):
            f.write(json.dumps({"item_id": item_id, "scenario": scenario, "split": split},
                               ensure_ascii=False) + "\n")

    per_scen = defaultdict(lambda: {"dev": 0, "test": 0})
    for _, (scenario, split) in assignment.items():
        per_scen[scenario][split] += 1
    print(json.dumps({"data": args.data, "n": len(rows), "test_frac": args.test_frac,
                      "seed": args.seed, "totals": counts,
                      "per_scenario": {k: dict(v) for k, v in sorted(per_scen.items())},
                      "output": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
