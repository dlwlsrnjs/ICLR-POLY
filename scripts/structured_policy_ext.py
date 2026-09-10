#!/usr/bin/env python3
"""Extended domain-structured policy (~11 interpretable params) using more fingerprint
richness: alignment A, multilingual weakness M, and the CAPABILITY CURVE (C_lo at low
load, C_hi at high load) instead of a single capability number.

  D(n,F,arr) = a_n*(n/10) + a_F*(F/12) + a_shuf*[shuffled]          # difficulty axes (3)
  cap(n)     = C_lo + (C_hi - C_lo)*((n-2)/8)                        # capability degrades with load
  C_eff      = c0 + c1*cap(n)                                        # capability scale (2)
  R_hat = sigmoid(k*(C_eff - D))                                     # reconstruction gate (1)
  U_hat = sigmoid(u0 + u_a*A + u_d*D + u_m*M + u_am*A*M)             # evasion; multilingual weakness M (5)
  J_hat = R_hat * U_hat

theta = (a_n, a_F, a_shuf, c0, c1, k, u0, u_a, u_d, u_m, u_am)  -> 11 numbers, all readable.
Fingerprint per target: A=ctx[0], M=ctx[1], C_lo=ctx[3], C_hi=ctx[4]. Compared LOO against
the 6-param base structured policy and the fixed config, on the full 32-arm joint table.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np

CELL = re.compile(r"frag(\d+)_(ordered|shuffled)_n(\d+)")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def arm_parts(arm_names):
    n = np.array([int(CELL.match(c).group(3)) for c in arm_names], float)
    F = np.array([int(CELL.match(c).group(1)) for c in arm_names], float)
    shuf = np.array([CELL.match(c).group(2) == "shuffled" for c in arm_names], float)
    return n, F, shuf


def jmat(theta, parts, A, M, C_lo, C_hi):
    """Vectorized J_hat: fingerprints are [T]; returns [T, A_arms]."""
    a_n, a_F, a_shuf, c0, c1, k, u0, u_a, u_d, u_m, u_am = theta
    n, F, shuf = parts
    D = a_n * (n / 10.0) + a_F * (F / 12.0) + a_shuf * shuf              # [A]
    cap = C_lo[:, None] + (C_hi - C_lo)[:, None] * ((n[None, :] - 2) / 8.0)  # [T, A]
    C_eff = c0 + c1 * cap
    R = sigmoid(k * (C_eff - D[None, :]))
    U = sigmoid(u0 + u_a * A[:, None] + u_d * D[None, :] + u_m * M[:, None] + u_am * (A * M)[:, None])
    return R * U


def fit(theta0, parts, A, M, C_lo, C_hi, Jobs, iters=2500, lr=0.25):
    theta = np.array(theta0, float)
    def loss(th):
        return float(((jmat(th, parts, A, M, C_lo, C_hi) - Jobs) ** 2).mean())
    for _ in range(iters):
        g = np.zeros_like(theta); base = loss(theta)
        for i in range(len(theta)):
            d = theta.copy(); d[i] += 1e-4
            g[i] = (loss(d) - base) / 1e-4
        theta -= lr * g
    return theta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="results/context_selector_full_train_20260904")
    ap.add_argument("--outdir", default="results/structured_policy_ext_20260904")
    a = ap.parse_args()
    blob = np.load(Path(a.data) / "context_values.npz")
    man = json.loads((Path(a.data) / "manifest.json").read_text())
    J = blob["J"]; ctx = blob["context"]; splits = blob["splits"]
    arm_names = man["arms"]; tags = man["targets"]; fams = man["families"]; T = len(tags)
    parts = arm_parts(arm_names)
    A = ctx[:, 0].astype(float); M = ctx[:, 1].astype(float)
    C_lo = 0.2 + ctx[:, 3].astype(float); C_hi = 0.2 + ctx[:, 4].astype(float)
    tr_i = splits == "train"; ev_i = splits == "test"
    Jtrain = J[:, tr_i, :].mean(1); Jeval = J[:, ev_i, :].mean(1)

    theta0 = [1.0, 0.3, 0.1, 0.0, 1.0, 4.0, -1.0, 2.0, 1.0, 1.0, 0.0]
    names = ["a_n", "a_F", "a_shuf", "c0", "c1", "k", "u0", "u_a", "u_d", "u_m", "u_am"]
    fam_list = sorted(set(fams))
    rows = []
    for held in range(T):
        trn = [i for i in range(T) if fams[i] != fams[held]]
        theta = fit(theta0, parts, A[trn], M[trn], C_lo[trn], C_hi[trn], Jtrain[trn], iters=1500)
        pred = jmat(theta, parts, A[held:held+1], M[held:held+1], C_lo[held:held+1], C_hi[held:held+1])[0]
        pick = int(pred.argmax())
        gfix = int(Jeval[trn].mean(0).argmax())
        orc = int(Jeval[held].argmax())
        rows.append(dict(target=tags[held], family=fams[held],
                         ext_arm=arm_names[pick], ext_gated=float(Jeval[held, pick]),
                         fixed_gated=float(Jeval[held, gfix]), oracle_gated=float(Jeval[held, orc]),
                         A=round(float(A[held]), 2), M=round(float(M[held]), 2),
                         C_lo=round(float(C_lo[held]), 2), C_hi=round(float(C_hi[held]), 2)))

    def agg(key):
        v = np.array([r[key] for r in rows]); orc = np.array([r["oracle_gated"] for r in rows])
        b = np.array([np.random.default_rng(s).choice(v, len(v)).mean() for s in range(3000)])
        return dict(mean=float(v.mean()), ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                    regret=float((orc - v).mean()))
    theta_all = fit(theta0, parts, A, M, C_lo, C_hi, Jtrain, iters=2500)
    summary = dict(scope="Extended domain-structured policy (11 params, richer fingerprint), family LOO.",
                   n_targets=T, families=fam_list, theta_names=names,
                   theta_global=[round(float(x), 3) for x in theta_all],
                   extended=agg("ext_gated"), fixed=agg("fixed_gated"), oracle=agg("oracle_gated"),
                   per_target=rows)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("theta:", dict(zip(names, summary["theta_global"])))
    print("\nLOO joint gated ASR (family hold-out):")
    for k in ("extended", "fixed", "oracle"):
        s = summary[k]; print(f"  {k:9s} {s['mean']:.3f}  CI[{s['ci'][0]:.3f},{s['ci'][1]:.3f}]  regret={s['regret']:.3f}")
    print("\nper-target (A,M,C_lo,C_hi -> ext pick vs fixed vs oracle):")
    for r in rows:
        print(f"  {r['target']:20s} A={r['A']:.2f} M={r['M']:.2f} Clo={r['C_lo']:.2f} Chi={r['C_hi']:.2f}  "
              f"ext={r['ext_arm']:16s}{r['ext_gated']:.2f}  fix={r['fixed_gated']:.2f}  orc={r['oracle_gated']:.2f}")


if __name__ == "__main__":
    main()
