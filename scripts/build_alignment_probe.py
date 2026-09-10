#!/usr/bin/env python3
"""Build the two alignment probes used to quantify a target's refusal barrier A.

probe_overrefusal : the 250 SAFE prompts of XSTest v2 (Roettger et al., NAACL 2024).
    Entirely benign content. Measures how readily a model refuses requests that
    only *sound* unsafe, i.e. its refusal propensity, at zero harmful-query cost.
probe_direct_harm : a small sample of direct English requests from the Lingua
    dev split. Measures the refusal barrier on genuinely harmful input, and costs
    a handful of harmful queries. Controlled text -> written 0600.

Both are attacker-side measurements taken on DEV items only, so they never touch
the held-out test items used for reported results.
"""
from __future__ import annotations
import argparse, json, os, random
from pathlib import Path

XSTEST = "/home/ubuntu/342/jinkwon/datasets/xstest/xstest_safe_250.jsonl"
LINGUA_DEV = "/home/ubuntu/342/jinkwon/datasets/lingua_safetybench_text/textdom_dev_full.jsonl"

def secure_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.chmod(path, 0o600)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-direct", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260903)
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    safe = [json.loads(l) for l in open(XSTEST, encoding="utf-8")]
    over = [{"probe": "overrefusal", "id": r["id"], "type": r["type"], "prompt": r["prompt"]} for r in safe]
    (out / "probe_overrefusal.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in over), encoding="utf-8")

    dev = [json.loads(l) for l in open(LINGUA_DEV, encoding="utf-8")]
    by_scen: dict[str, list] = {}
    for r in dev:
        by_scen.setdefault(r.get("scenario", "?"), []).append(r)
    rng = random.Random(a.seed)
    picked = []
    scens = sorted(by_scen)
    per = max(1, a.n_direct // len(scens))
    for s in scens:
        ids = sorted(by_scen[s], key=lambda r: r["item_id"])
        rng.shuffle(ids)
        picked += ids[:per]
    picked = picked[: a.n_direct]
    direct = [{"probe": "direct_harm", "id": r["item_id"], "type": r.get("scenario", "?"),
               "prompt": r["questions"]["English"], "original": r["original"]} for r in picked]
    secure_write(out / "probe_direct_harm.jsonl", direct)

    print(json.dumps({"overrefusal": len(over), "direct_harm": len(direct),
                      "scenarios": scens, "seed": a.seed, "outdir": str(out)}, indent=2))

if __name__ == "__main__":
    main()
