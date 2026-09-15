#!/usr/bin/env python3
"""Reproduce the paper's selector comparison on Lingua-SafetyBench alone, 17-model panel.

Lingua used the same translation language (Norwegian) in both collections, so it is the clean
benchmark for asking whether the per-target selector really beats fixed attacks.

Scoring follows scripts/arm_scoring.py: clear-text arms (AIM, DeepInception, PAP, translation) are
ungated (unsafe rate); every hidden-request arm is reconstruction-gated (recon AND unsafe).

Items are split in half: the selector may only observe calibration items, and every reported number
is on the disjoint test half, so the oracle and the search share the same winner's-curse exposure.
"""
from __future__ import annotations
import json, re, sys, math, random
from pathlib import Path
import numpy as np

SCRATCH = Path(__file__).resolve().parent
BEN = Path('/home/ubuntu/342/jinkwon/poly/bucket_0913/L40S-only/experiments_suite/exp02_panel_collect/results/benign')
CLEAR = {'m_aim', 'm_deepinception', 'm_pap', 'm_translated'}
G = re.compile(r'^g(\d+)_(ordered|shuffled)_n(\d+)__(.+)$')
FRAMES = ['plain', 'persona', 'fiction', 'pap', 'persona+fiction']

U0 = 0.6      # prior weight, paper Eq. (5)
BETA = 1.0    # UCB exploration, paper picks beta=1
SIGMA = 0.35  # kernel amplitude
LS = 1.0      # kernel length scale
NOISE = 0.02


def features(arm):
    """Normalised encoding: fragment count, order, language load, frame one-hot, clear-text flag."""
    m = G.match(arm)
    if not m:
        f = [0.0, 0.0, 0.0] + [0.0] * len(FRAMES) + [1.0]
        f[3 + FRAMES.index('persona')] = 1.0 if arm == 'm_aim' else 0.0
        f[3 + FRAMES.index('fiction')] = 1.0 if arm == 'm_deepinception' else 0.0
        f[3 + FRAMES.index('pap')] = 1.0 if arm == 'm_pap' else 0.0
        return np.array(f)
    frag, order, n, frame = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
    f = [frag / 12.0, 1.0 if order == 'shuffled' else 0.0, n / 8.0] + [0.0] * len(FRAMES) + [0.0]
    if frame in FRAMES:
        f[3 + FRAMES.index(frame)] = 1.0
    return np.array(f)


def gp_posterior(X, Xq, y, mu0, mu0q):
    """Standard GP posterior with an RBF kernel and a probe-derived prior mean."""
    def k(A, B):
        d = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return SIGMA ** 2 * np.exp(-d / (2 * LS ** 2))
    if len(X) == 0:
        return mu0q, np.full(len(Xq), SIGMA)
    K = k(X, X) + NOISE * np.eye(len(X))
    Ks = k(Xq, X)
    a = np.linalg.solve(K, y - mu0)
    mean = mu0q + Ks @ a
    v = np.linalg.solve(K, Ks.T)
    var = np.clip(SIGMA ** 2 - (Ks * v.T).sum(1), 1e-9, None)
    return mean, np.sqrt(var)


def run(budget, seed, matrix, priors, tags, arms, calib, test, mode):
    """One replay. Returns {tag: test score of the recommended arm}."""
    rng = random.Random(seed)
    out = {}
    F = np.array([features(a) for a in arms])
    for tag in tags:
        cal = {a: float(np.mean([matrix[tag][a][i] for i in calib])) for a in arms}
        tst = {a: float(np.mean([matrix[tag][a][i] for i in test])) for a in arms}
        probe = np.array([priors[tag].get(a, 0.0) for a in arms], float)
        if mode == 'random':
            picked = rng.sample(range(len(arms)), min(budget, len(arms)))
            best = max(picked, key=lambda i: cal[arms[i]])
        elif mode in ('ours', 'uninformed'):
            mu0 = U0 * probe if mode == 'ours' else np.zeros(len(arms))
            queried, ys = [], []
            for _ in range(budget):
                if queried:
                    mean, sd = gp_posterior(F[queried], F, np.array(ys), mu0[queried], mu0)
                else:
                    mean, sd = mu0.copy(), np.full(len(arms), SIGMA)
                acq = mean + BETA * sd
                for i in queried:
                    acq[i] = -1e9
                nxt = int(np.argmax(acq))
                queried.append(nxt)
                ys.append(cal[arms[nxt]])
            best = max(queried, key=lambda i: cal[arms[i]])
        elif mode == 'prior_only':
            best = int(np.argmax(probe))
        else:
            raise ValueError(mode)
        out[tag] = tst[arms[best]]
    return out


def main():
    data = json.loads((SCRATCH / 'item_matrix.json').read_text())
    tags, items, matrix = data['tags'], data['items'], data['matrix']
    arms = sorted(matrix[tags[0]])
    priors = {t: json.loads((BEN / f'{t}.json').read_text())['prior'] for t in tags}

    n = len(items)
    results = {k: [] for k in ('oracle', 'fixed', 'random3', 'random8', 'unin3', 'unin8',
                               'ours3', 'ours8', 'prior_only', 'aim', 'deepinception', 'pap', 'translated')}
    SPLITS = 40
    for s in range(SPLITS):
        rng = random.Random(1000 + s)
        idx = list(range(n))
        rng.shuffle(idx)
        calib, test = idx[: n // 2], idx[n // 2:]
        tst = {t: {a: float(np.mean([matrix[t][a][i] for i in test])) for a in arms} for t in tags}
        cal = {t: {a: float(np.mean([matrix[t][a][i] for i in calib])) for a in arms} for t in tags}
        # best fixed configuration, chosen on OTHER models' calibration halves (leave-one-model-out)
        fixed = {}
        for t in tags:
            others = [x for x in tags if x != t]
            mean = {a: float(np.mean([cal[o][a] for o in others])) for a in arms}
            fixed[t] = tst[t][max(mean, key=mean.get)]
        for k, v in (('oracle', {t: max(tst[t].values()) for t in tags}), ('fixed', fixed)):
            results[k].append(v)
        for name, arm in (('aim', 'm_aim'), ('deepinception', 'm_deepinception'),
                          ('pap', 'm_pap'), ('translated', 'm_translated')):
            results[name].append({t: tst[t][arm] for t in tags})
        results['prior_only'].append(run(0, s, matrix, priors, tags, arms, calib, test, 'prior_only'))
        for b, suf in ((3, '3'), (8, '8')):
            results['random' + suf].append(run(b, s, matrix, priors, tags, arms, calib, test, 'random'))
            results['unin' + suf].append(run(b, s, matrix, priors, tags, arms, calib, test, 'uninformed'))
            results['ours' + suf].append(run(b, s, matrix, priors, tags, arms, calib, test, 'ours'))
        print(f"  split {s + 1}/{SPLITS} done", flush=True)

    per_model = {k: {t: float(np.mean([r[t] for r in v])) for t in tags} for k, v in results.items()}
    json.dump({'per_model': per_model, 'splits': SPLITS, 'n_items': n, 'arms': len(arms)},
              open(SCRATCH / 'lg_selector_results.json', 'w'), indent=1)
    print('\n=== Lingua-SafetyBench, 17 models, verified scoring, held-out item half ===')
    order = ['pap', 'translated', 'deepinception', 'aim', 'fixed', 'prior_only',
             'random3', 'unin3', 'ours3', 'random8', 'unin8', 'ours8', 'oracle']
    label = {'pap': 'PAP', 'translated': 'low-resource translation', 'deepinception': 'DeepInception',
             'aim': 'AIM persona', 'fixed': 'best fixed (leave-one-model-out)',
             'prior_only': 'benign prior only, 0 queries', 'random3': 'random, 3 queries',
             'unin3': 'uninformed GP, 3', 'ours3': 'benign-warm GP (ours), 3',
             'random8': 'random, 8', 'unin8': 'uninformed GP, 8', 'ours8': 'benign-warm GP (ours), 8',
             'oracle': 'oracle (test-half best arm)'}
    for k in order:
        print(f"  {label[k]:38s} {np.mean(list(per_model[k].values())):.3f}")

    def paired(a, b):
        d = np.array([per_model[a][t] - per_model[b][t] for t in tags])
        rng = np.random.default_rng(0)
        bs = d[rng.integers(0, len(d), size=(20000, len(d)))].mean(1)
        wins = int((d > 0).sum())
        # exact two-sided sign-flip test over all 2^n sign assignments
        k = len(d)
        masks = np.arange(1 << k, dtype=np.int64)[:, None]
        signs = 1 - 2 * ((masks >> np.arange(k)) & 1).astype(np.int8)
        stats = (signs * d).mean(1)
        p = float((np.abs(stats) >= abs(d.mean()) - 1e-12).mean())
        return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), wins, p

    print('\n=== paired over 17 models ===')
    for a, b in (('ours3', 'fixed'), ('ours8', 'fixed'), ('ours3', 'aim'), ('ours8', 'aim'),
                 ('ours3', 'unin3'), ('ours8', 'unin8'), ('ours3', 'random3'), ('ours8', 'random8')):
        m, lo, hi, w, p = paired(a, b)
        print(f"  {a:6s} vs {b:8s}  delta {m:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]  wins {w}/17  p={p:.4f}")


if __name__ == '__main__':
    main()
