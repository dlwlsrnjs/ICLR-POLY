#!/usr/bin/env python3
"""EXP04 budget / query-efficiency: how many harmful configuration evaluations to reach the oracle,
with the benign warm-start prior vs an uninformed (random) search over the SAME 292-arm space.

Inputs (per model, per collection), all produced by exp02:
  <root>/attack/<tag>__<arm>.json   verified per arm  (the full matrix)
  <root>/benign/<tag>.json          'prior' over arms (harmless warm start)

For each model tag:
  oracle          = max verified over all arms
  ours@k          = best verified among the top-k prior-ranked arms (greedy on the free prior)
  random@k        = expected best verified among k random arms (mean over R draws)
  q_ours / q_rand = smallest k reaching 95% of oracle (ours vs uninformed)
Offline (reads JSON only). Usage:
  python analyze.py --root <exp02 results> --suffix _mj [--budget 8] [--reps 400]"""
import argparse, json, glob, collections, random
from pathlib import Path


def load(root, suffix):
    mat = collections.defaultdict(dict)
    for f in glob.glob(str(Path(root) / "attack" / "*.json")):
        d = json.loads(Path(f).read_text())
        tag, method = d["tag"], d["method"]
        if suffix and not tag.endswith(suffix):
            continue
        if method.startswith("ours[") or method in ("plain", "translated", "cipher_base64",
                                                     "aim", "deepinception", "pap"):
            continue
        mat[tag][method] = d["verified"]
    priors = {}
    for f in glob.glob(str(Path(root) / "benign" / f"*{suffix}.json")):
        d = json.loads(Path(f).read_text())
        priors[d["tag"]] = d.get("prior", {})
    return mat, priors


def curve(ver, prior, budget, reps):
    arms = list(ver)
    oracle = max(ver.values())
    ranked = sorted(arms, key=lambda a: -prior.get(a, 0.0))
    ours = []
    best = 0.0
    for k in range(1, budget + 1):
        best = max(ver[a] for a in ranked[:k])
        ours.append(round(best, 3))
    rng = random.Random(0)
    rand = []
    for k in range(1, budget + 1):
        s = 0.0
        for _ in range(reps):
            s += max(ver[a] for a in rng.sample(arms, min(k, len(arms))))
        rand.append(round(s / reps, 3))
    def q_to(frac_curve):
        for k, v in enumerate(frac_curve, 1):
            if v >= 0.95 * oracle:
                return k
        return None
    return dict(oracle=round(oracle, 3), ours_at_k=ours, random_at_k=rand,
                q_ours=q_to(ours), q_random=q_to(rand), n_arms=len(arms))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--suffix", default="_mj")
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--reps", type=int, default=400)
    a = ap.parse_args()
    mat, priors = load(a.root, a.suffix)
    if not mat:
        raise SystemExit(f"no matrix under {a.root}/attack for suffix {a.suffix}")
    out = {}
    for tag in sorted(mat):
        if tag not in priors:
            print(f"skip {tag}: no benign prior"); continue
        out[tag] = curve(mat[tag], priors[tag], a.budget, a.reps)
        print(json.dumps({tag: out[tag]}, indent=2))
    p = Path(a.root) / f"queryeff{a.suffix}.json"
    p.write_text(json.dumps(out, indent=2)); print("wrote", p)


if __name__ == "__main__":
    raise SystemExit(main())
