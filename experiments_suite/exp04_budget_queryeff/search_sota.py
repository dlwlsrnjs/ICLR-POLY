#!/usr/bin/env python3
"""EXP04c SOTA search tailored to the factorized, prior-misspecifiable 292-arm space.

Techniques:
  - ADDITIVE (factorized) GP kernel  k = k_comprehension(F,n,arr) + k_willingness(multi-hot),
    matching verified ~ recon x comply, so one observation generalizes across a whole willingness
    family or comprehension slice.
  - pi-BO decaying prior (Hvarfner et al. 2022): benign prior enters the acquisition as pi(x)^{gamma/(t+1)}
    -- a good prior accelerates early, a saturated/wrong prior washes out (never worse than uninformed).
  - Acquisitions: UCB and Top-Two Thompson Sampling (TTTS) on the GP posterior (SOTA best-arm ID).

Replay over exp02 verified matrix + benign prior. Seed-averaged. numpy only.
Usage: python search_sota.py --root <exp02 results> --suffix _mj [--budget 8] [--reps 24]"""
import argparse, json, glob, re, collections
from pathlib import Path
import numpy as np

CELL = re.compile(r"g(\d+)_(ordered|shuffled)_n(\d+)__(.+)")


def feats(arm):
    m = CELL.match(arm)
    if not m:
        w = {"m_aim": (1, 0, 0), "m_deepinception": (0, 1, 0), "m_pap": (0, 0, 1)}.get(arm, (0, 0, 0))
        comp = np.array([0., 0., 0.]); will = np.array([w[0], w[1], w[2], 0., 1.])
        return comp, will
    F, arr, n, wl = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
    p = wl.split("+")
    comp = np.array([F / 12, n / 10, float(arr == "shuffled")])
    will = np.array([float("persona" in p or wl == "role"), float("fiction" in p),
                     float("pap" in p), float(wl == "role"), float(wl == "plain")])
    return comp, will


def rbf(A, B, ls):
    d = (A[:, None, :] - B[None, :, :]) / ls
    return np.exp(-(d ** 2).sum(-1) / 2)




def saturation_aware_prior(bj, arms, info_range=0.10):
    """Drop uninformative (saturated) axes: keep an axis only if its benign signal varies across its
    groups by more than `info_range`. prior = product of SURVIVING factors (uniform if all dropped).
      comprehension axis: benign reconstruction per CELL (varies by F,n,arr)
      willingness axis   : benign frame signal per FRAME (persona/fiction/pap); saturates on helpful models
    This is the free fix for prior misspecification: a constant factor carries no ranking information and
    only hurts, so we discard it and let the structured search explore that axis."""
    import numpy as _np
    cr = bj.get("benign_recon_by_cell", {}); sig = bj.get("frame_signals", {})
    rmean = bj.get("benign_recon_mean", 0.5)
    cvals = list(cr.values())
    comp_inf = (max(cvals) - min(cvals) > info_range) if cvals else False
    fv = {f: sig[f].get("nonrefusal", 0.5) for f in sig}
    will_inf = (max(fv.values()) - min(fv.values()) > info_range) if fv else False

    def cell_of(a):
        m = CELL.match(a); return f"g{m.group(1)}_{m.group(2)}_n{m.group(3)}" if m else None

    def will_of(a):
        m = CELL.match(a)
        wl = m.group(4) if m else a
        parts = wl.split("+")
        f = 1.0
        for pt, key in (("persona", "persona"), ("fiction", "fiction"), ("pap", "pap")):
            if pt in parts:
                f *= fv.get(key, 0.5)
        if wl == "role":
            f *= fv.get("persona", 0.5)
        return f  # plain -> 1.0 (no willingness frame)

    prior = {}
    for a in arms:
        p = 1.0
        if comp_inf:
            c = cell_of(a); p *= cr.get(c, rmean) if c else rmean
        if will_inf:
            p *= will_of(a)
        if not comp_inf and not will_inf:
            p = 0.5
        prior[a] = p
    return prior, dict(comp_informative=comp_inf, will_informative=will_inf)


def load(root, suffix, saturation_aware=True):
    ver = {}
    for f in glob.glob(str(Path(root) / "attack" / "*.json")):
        d = json.loads(Path(f).read_text())
        if suffix and not d["tag"].endswith(suffix):
            continue
        if d["method"].startswith("ours[") or d["method"] in (
                "plain", "translated", "cipher_base64", "aim", "deepinception", "pap"):
            continue
        ver.setdefault(d["tag"], {})[d["method"]] = d["verified"]
    pri = {}
    for f in glob.glob(str(Path(root) / "benign" / f"*{suffix}.json")):
        d = json.loads(Path(f).read_text())
        if saturation_aware and d["tag"] in ver:
            pri[d["tag"]], _ = saturation_aware_prior(d, list(ver[d["tag"]]))
        else:
            pri[d["tag"]] = d.get("prior", {})
    return ver, pri


def search(arms, ver, prior, budget, acq, additive, use_pibo, rng, beta=2.0, gamma=3.0):
    n = len(arms)
    C = np.array([feats(a)[0] for a in arms]); W = np.array([feats(a)[1] for a in arms])
    if additive:
        K = 0.5 * rbf(C, C, np.array([0.6, 0.6, 0.9])) + 0.5 * rbf(W, W, np.array([0.4, 0.4, 0.4, 0.5, 0.5]))
    else:
        X = np.concatenate([C, W], 1)
        K = rbf(X, X, np.array([0.6, 0.6, 0.9, 0.4, 0.4, 0.4, 0.5, 0.5]))
    m = np.full(n, 0.5)
    pv = np.array([prior.get(a, 0.5) for a in arms]); pv = pv / (pv.max() + 1e-9)   # prior over optimum
    obs, ys = [], []; best = 0.0; curve = []; noise = 0.02
    for t in range(budget):
        if not obs:
            mu = m.copy(); sd = np.ones(n)
        else:
            I = np.array(obs); Kinv = np.linalg.inv(K[np.ix_(I, I)] + noise * np.eye(len(I)))
            Ks = K[:, I]; mu = m + Ks @ Kinv @ (np.array(ys) - m[I])
            var = np.clip(1.0 - np.einsum("ij,jk,ik->i", Ks, Kinv, Ks), 1e-6, None); sd = np.sqrt(var)
        if acq == "ucb":
            a = mu + beta * sd
        elif acq == "ttts":
            s1 = mu + sd * rng.standard_normal(n)
            champ = int(np.argmax([-1e9 if i in obs else s1[i] for i in range(n)]))
            if rng.random() < 0.5:
                pick0 = champ
            else:
                pick0 = champ
                for _ in range(20):
                    s2 = mu + sd * rng.standard_normal(n)
                    c2 = int(np.argmax([-1e9 if i in obs else s2[i] for i in range(n)]))
                    if c2 != champ:
                        pick0 = c2; break
            a = np.full(n, -1e9); a[pick0] = 1.0
        if use_pibo:
            a = a + (gamma / (t + 1)) * np.log(pv + 1e-6)   # decaying log-prior tilt
        for j in obs:
            a[j] = -1e9
        if not obs and acq == "ucb":
            pick = int(rng.integers(n))
        else:
            pick = int(np.argmax(a + rng.normal(0, 1e-6, n)))
        y = ver[arms[pick]]; obs.append(pick); ys.append(y); best = max(best, y); curve.append(best)
    return np.array(curve)


def avg(arms, ver, prior, budget, reps, **kw):
    cs = np.array([search(arms, ver, prior, budget, rng=np.random.default_rng(s), **kw) for s in range(reps)])
    return [round(float(x), 3) for x in cs.mean(0)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True); ap.add_argument("--suffix", default="_mj")
    ap.add_argument("--budget", type=int, default=8); ap.add_argument("--reps", type=int, default=24)
    a = ap.parse_args()
    ver, pri = load(a.root, a.suffix)
    out = {}
    for tag in sorted(ver):
        if tag not in pri:
            continue
        v, p = ver[tag], pri[tag]; arms = list(v); oracle = max(v.values())
        rng = np.random.default_rng(0)
        rand = [round(float(np.mean([max(v[a] for a in rng.sample(arms, k)) if False else
                    max(v[a] for a in list(np.random.default_rng(s).choice(arms, k, replace=False)))
                    for s in range(200)])), 3) for k in range(1, a.budget + 1)]
        res = dict(oracle=round(oracle, 3), n_arms=len(arms), random=rand,
                   ucb_iso=avg(arms, v, p, a.budget, a.reps, acq="ucb", additive=False, use_pibo=False),
                   ucb_add=avg(arms, v, p, a.budget, a.reps, acq="ucb", additive=True, use_pibo=False),
                   ucb_add_pibo=avg(arms, v, p, a.budget, a.reps, acq="ucb", additive=True, use_pibo=True),
                   ttts_add=avg(arms, v, p, a.budget, a.reps, acq="ttts", additive=True, use_pibo=False),
                   ttts_add_pibo=avg(arms, v, p, a.budget, a.reps, acq="ttts", additive=True, use_pibo=True))
        out[tag] = res
        print(f"[{tag}] oracle={res['oracle']} n={res['n_arms']}")
        for s in ("random", "ucb_iso", "ucb_add", "ucb_add_pibo", "ttts_add", "ttts_add_pibo"):
            print(f"   {s:14} {res[s]}")
    Path(a.root, f"search_sota{a.suffix}.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
