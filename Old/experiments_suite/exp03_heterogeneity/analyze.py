#!/usr/bin/env python3
"""EXP03 heterogeneity analysis: reads the full-matrix aggregates collected by exp02
(<root>/attack/<tag>__<arm>.json, verified per arm per model) and produces the paper's core
"no fixed configuration is optimal" numbers per dataset:
  - per-model best arm (oracle) and its verified ASR
  - best FIXED arm (argmax of mean verified across models) and its per-model spread
  - number of DISTINCT winning arms across the panel
Dataset-agnostic reader; the dataset separation lives in exp02's collection (mj vs lg roots/tags).
Offline (reads JSON only). Usage: python analyze.py --root <exp02 results dir> [--suffix _mj|_lg]"""
import argparse, json, glob, collections
from pathlib import Path


def load_matrix(root, suffix):
    """-> {model_tag: {arm: verified}} from attack aggregates named <tag>__<arm>.json."""
    mat = collections.defaultdict(dict)
    for f in glob.glob(str(Path(root) / "attack" / "*.json")):
        d = json.loads(Path(f).read_text())
        tag, method = d["tag"], d["method"]
        if suffix and not tag.endswith(suffix):
            continue
        # full-matrix labels are the arm name; skip baseline/ours[...] labels
        if method.startswith("ours[") or method in ("plain", "translated", "cipher_base64", "aim",
                                                     "deepinception", "pap"):
            continue
        mat[tag][method] = d["verified"]
    return mat


def analyze(mat, expected_arms=292, allow_incomplete=False):
    models = sorted(mat)
    arms = sorted({a for m in mat.values() for a in m})
    counts = {model: len(mat[model]) for model in models}
    incomplete = {model: count for model, count in counts.items() if count != expected_arms}
    if incomplete and not allow_incomplete:
        raise ValueError(
            f"incomplete full matrix: expected {expected_arms} arms per model, got {incomplete}; "
            "resume collection or pass --allow-incomplete for a progress-only summary")
    best_per_model = {t: max(mat[t], key=mat[t].get) for t in models if mat[t]}
    oracle = {t: mat[t][best_per_model[t]] for t in best_per_model}
    # best fixed arm = argmax mean verified across models (only arms present for all models)
    common = [a for a in arms if all(a in mat[t] for t in models)]
    fixed_mean = {a: sum(mat[t][a] for t in models) / len(models) for a in common} if common else {}
    best_fixed = max(fixed_mean, key=fixed_mean.get) if fixed_mean else None
    distinct = sorted(set(best_per_model.values()))
    return dict(models=models, n_arms=len(arms), arm_counts=counts, complete=not incomplete,
                expected_arms=expected_arms, best_per_model=best_per_model, oracle=oracle,
                best_fixed=best_fixed, best_fixed_mean=(fixed_mean.get(best_fixed) if best_fixed else None),
                distinct_winners=distinct, n_distinct=len(distinct))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--suffix", default="", help="_mj or _lg to select a dataset's tags")
    ap.add_argument("--out", default="")
    ap.add_argument("--expected-arms", type=int, default=292)
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="write a progress-only summary instead of failing on partial matrices")
    a = ap.parse_args()
    mat = load_matrix(a.root, a.suffix)
    if not mat:
        raise SystemExit(f"no full-matrix aggregates under {a.root}/attack (run exp02 collector first)")
    try:
        res = analyze(mat, a.expected_arms, a.allow_incomplete)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(dict(n_models=len(res["models"]), n_arms=res["n_arms"],
                          n_distinct_winners=res["n_distinct"], best_fixed=res["best_fixed"],
                          best_per_model=res["best_per_model"]), indent=2))
    out = a.out or str(Path(a.root) / f"heterogeneity{a.suffix or ''}.json")
    Path(out).write_text(json.dumps(res, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
