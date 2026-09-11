#!/usr/bin/env python3
"""EXP04b stronger search than prior-greedy/BAI: structure-aware GP-UCB over an arm-feature space,
so the search explores even when the benign prior is saturated/miscalibrated -- and never loses to
random. Replays over the collected verified matrix (exp02) + benign prior.

Strategies compared at each budget k (harmful configuration evaluations):
  random        uniform arms (baseline that must be beaten)
  prior_greedy  observe top-k by benign prior (the weak strategy that lost to random)
  coverage      stratified pick spanning willingness families x comprehension cells (robust floor)
  gp_ucb        GP posterior (RBF over features, mean = benign prior) with UCB acquisition; explores
                high-uncertainty regions so it finds the winning region despite a flat/bad prior

Feature(arm) = [F/12, n/10, shuffled, persona, fiction, pap, role, plain]. Offline; numpy only.
Usage: python search_strategies.py --root <exp02 results> --suffix _mj [--budget 8] [--beta 2.0] [--reps 300]"""
import argparse, json, glob, re, collections, random
from pathlib import Path
import numpy as np

CELL = re.compile(r"g(\d+)_(ordered|shuffled)_n(\d+)__(.+)")


def feat(arm):
    m = CELL.match(arm)
    if not m:  # single-vector baselines: put them as low-comprehension willingness points
        w = {"m_aim": (1, 0, 0), "m_deepinception": (0, 1, 0), "m_pap": (0, 0, 1)}.get(arm, (0, 0, 0))
        return np.array([0, 0, 0, w[0], w[1], w[2], 0, 1 if arm == "m_translated" else 0], float)
    F, arr, n, will = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
    parts = will.split("+")
    persona = float("persona" in parts); fiction = float("fiction" in parts); pap = float("pap" in parts)
    role = float(will == "role"); plain = float(will == "plain")
    return np.array([F / 12, n / 10, float(arr == "shuffled"),
                     persona or role, fiction, pap, role, plain], float)


def load(root, suffix):
    ver = {}
    for f in glob.glob(str(Path(root) / "attack" / "*.json")):
        d = json.loads(Path(f).read_text())
        if suffix and not d["tag"].endswith(suffix):
            continue
        if d["method"].startswith("ours[") or d["method"] in (
                "plain", "translated", "cipher_base64", "aim", "deepinception", "pap"):
            continue
        ver.setdefault(d["tag"], {})[d["method"]] = d["verified"]
    priors = {}
    for f in glob.glob(str(Path(root) / "benign" / f"*{suffix}.json")):
        d = json.loads(Path(f).read_text()); priors[d["tag"]] = d.get("prior", {})
    return ver, priors


def _kernel(X, ls):
    d = (X[:, None, :] - X[None, :, :]) / ls[None, None, :]
    return np.exp(-(d ** 2).sum(-1) / 2)


def gp_ucb(arms, X, y_fn, prior, budget, beta, seed_cov=0, prior_weight=0.0,
           ls=None, noise=0.02, seed_order=None, rng=None):
    """GP-UCB. mean = prior_weight*prior + (1-prior_weight)*0.5 (flat by default, so a saturated/bad
    prior cannot dominate). Per-dimension lengthscales: short on willingness dims so persona/fiction/
    pap regions stay distinct. Optionally seed the first `seed_cov` picks by a coverage order so the
    willingness axis is sampled early (robust >= random)."""
    n = len(arms)
    if ls is None:
        # dims: [F,n,shuffled, persona/role, fiction, pap, role, plain]
        ls = np.array([0.6, 0.6, 0.9, 0.35, 0.35, 0.35, 0.5, 0.5])
    K = _kernel(X, ls)
    pm = np.array([prior.get(a, 0.5) for a in arms], float)
    m = prior_weight * pm + (1 - prior_weight) * 0.5
    if rng is None: rng = np.random.default_rng(0)
    obs_idx, obs_y = [], []
    best = 0.0; curve = []
    seed = (seed_order or [])[:seed_cov]
    for step in range(budget):
        if step < len(seed):
            pick = arms.index(seed[step])
        else:
            if not obs_idx:
                mu = m.copy(); sd = np.ones(n)
            else:
                I = np.array(obs_idx)
                Kinv = np.linalg.inv(K[np.ix_(I, I)] + noise * np.eye(len(I)))
                Ks = K[:, I]; resid = np.array(obs_y) - m[I]
                mu = m + Ks @ Kinv @ resid
                var = 1.0 - np.einsum("ij,jk,ik->i", Ks, Kinv, Ks)
                sd = np.sqrt(np.clip(var, 1e-6, None))
            acq = mu + beta * sd + rng.normal(0, 1e-6, size=n)
            for j in obs_idx:
                acq[j] = -1e9
            if not obs_idx:
                pick = int(rng.integers(n))
            else:
                pick = int(np.argmax(acq))
        if pick in obs_idx:
            rem = [i for i in range(n) if i not in obs_idx]
            pick = rem[0]
        yv = y_fn(arms[pick]); obs_idx.append(pick); obs_y.append(yv)
        best = max(best, yv); curve.append(round(best, 3))
    return curve


def coverage_order(arms):
    """Stratify: cycle through willingness families, within each spread over (F,n,arr)."""
    by_will = collections.defaultdict(list)
    for a in arms:
        m = CELL.match(a); w = m.group(4) if m else a
        by_will[w].append(a)
    for w in by_will:
        by_will[w].sort(key=lambda a: (CELL.match(a).group(1), CELL.match(a).group(3)) if CELL.match(a) else ("", ""))
    order, wills = [], list(by_will)
    i = 0
    while len(order) < len(arms):
        w = wills[i % len(wills)]
        if by_will[w]:
            order.append(by_will[w].pop(0))
        i += 1
    return order


def run(ver, prior, budget, beta, reps):
    arms = list(ver); oracle = max(ver.values())
    X = np.array([feat(a) for a in arms])
    # random
    rng = random.Random(0); rand = []
    for k in range(1, budget + 1):
        s = sum(max(ver[a] for a in rng.sample(arms, min(k, len(arms)))) for _ in range(reps)) / reps
        rand.append(round(s, 3))
    # prior greedy
    ranked = sorted(arms, key=lambda a: -prior.get(a, 0)); pg = []; b = 0
    for k in range(1, budget + 1):
        b = max(ver[a] for a in ranked[:k]); pg.append(round(b, 3))
    # coverage
    cov_o = coverage_order(arms); cov = []; b = 0
    for k in range(1, budget + 1):
        b = max(ver[a] for a in cov_o[:k]); cov.append(round(b, 3))
    cov_o = coverage_order(arms)
    import numpy as _np
    def _avg(**kw):
        cs = _np.array([gp_ucb(arms, X, lambda a: ver[a], prior, budget, beta, rng=_np.random.default_rng(s), **kw) for s in range(24)])
        return [round(float(x), 3) for x in cs.mean(0)]
    gp = _avg(prior_weight=0.0)
    gp_seed = _avg(prior_weight=0.0, seed_cov=4, seed_order=cov_o)
    def q(c):
        for k, v in enumerate(c, 1):
            if v >= 0.95 * oracle:
                return k
        return None
    return dict(oracle=round(oracle, 3), n_arms=len(arms),
                random=rand, prior_greedy=pg, coverage=cov, gp_ucb=gp, gp_ucb_seeded=gp_seed,
                q_random=q(rand), q_prior_greedy=q(pg), q_coverage=q(cov), q_gp_ucb=q(gp), q_gp_ucb_seeded=q(gp_seed))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True); ap.add_argument("--suffix", default="_mj")
    ap.add_argument("--budget", type=int, default=8); ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--reps", type=int, default=300)
    a = ap.parse_args()
    ver, priors = load(a.root, a.suffix)
    out = {}
    for tag in sorted(ver):
        if tag not in priors:
            continue
        out[tag] = run(ver[tag], priors[tag], a.budget, a.beta, a.reps)
        r = out[tag]
        print(f"[{tag}] oracle={r['oracle']} n={r['n_arms']}")
        for s in ("random", "prior_greedy", "coverage", "gp_ucb", "gp_ucb_seeded"):
            print(f"   {s:13} {r[s]}  q95={r['q_'+s]}")
    Path(a.root, f"search_strategies{a.suffix}.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
