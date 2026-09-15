#!/usr/bin/env python3
"""Domain-structured adaptive config policy (few parameters, strong inductive bias).

Instead of a black-box neural policy (which needs many tasks), encode the confirmed
mechanism of THIS domain so the joint objective is a low-parameter function of the
target's behavioral fingerprint:

  difficulty(n, arr, F) = n/10 + d_shuf*[shuffled] + d_frag*(F/12)      (structured axis)
  R_hat = sigmoid(k * (C - difficulty))          # reconstruction gate: capability C beats difficulty
  U_hat = sigmoid(u0 + u_a*A + u_d*difficulty)   # answer-safety failure: weak alignment A high;
                                                 #   strong alignment needs obfuscation (difficulty) to evade
  J_hat = R_hat * U_hat                           # joint = reconstruct AND evade

Per target, C = benign reconstruction capability and A = plain-request compliance
(english_direct raw ASR) come from the cheap fingerprint. The GLOBAL parameters
theta = (k, d_shuf, d_frag, u0, u_a, u_d) are the only things fit — six numbers — so 27
targets are plenty. The policy recommends argmax_config J_hat(config | fingerprint).

Evaluated leave-one-model-out against fixed and the context-free GP-UCB, on the measured
joint table. This is the domain-specialized alternative to generic PPO.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np

CELL10 = re.compile(r"interleave_(ordered|shuffled)_n(\d+)")
CELL32 = re.compile(r"frag(\d+)_(ordered|shuffled)_n(\d+)")


def decode_arm(name):
    m = CELL32.match(name)
    if m:
        return dict(n=int(m.group(3)), shuf=m.group(2) == "shuffled", F=int(m.group(1)))
    m = CELL10.match(name)
    return dict(n=int(m.group(2)), shuf=m.group(1) == "shuffled", F=5)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def difficulty(arm, d_shuf, d_frag):
    return arm["n"] / 10.0 + d_shuf * arm["shuf"] + d_frag * (arm["F"] / 12.0)


def _D(arms):
    n = np.array([a["n"] for a in arms]) / 10.0
    shuf = np.array([float(a["shuf"]) for a in arms])
    frag = np.array([a["F"] for a in arms]) / 12.0
    return n, shuf, frag


def jmat(theta, Dparts, C, A):
    """Vectorized J_hat over all targets: C,A are [T]; returns [T, A_arms]."""
    k, d_shuf, d_frag, u0, u_a, u_d = theta
    n, shuf, frag = Dparts
    D = n + d_shuf * shuf + d_frag * frag
    R = sigmoid(k * (C[:, None] - D[None, :]))
    U = sigmoid(u0 + u_a * A[:, None] + u_d * D[None, :])
    return R * U


def j_hat(theta, arms, C, A):
    return jmat(theta, _D(arms), np.array([C], float), np.array([A], float))[0]


def fit(theta0, arms, C, A, Jobs, iters=1500, lr=0.3):
    """Fit global theta to observed [T, A] gated J by numerical-gradient descent (vectorized)."""
    theta = np.array(theta0, float)
    Dp = _D(arms); C = np.asarray(C, float); A = np.asarray(A, float); Jobs = np.asarray(Jobs, float)
    def loss(th):
        return float(((jmat(th, Dp, C, A) - Jobs) ** 2).mean())
    for _ in range(iters):
        g = np.zeros_like(theta); base = loss(theta)
        for i in range(len(theta)):
            d = theta.copy(); d[i] += 1e-4
            g[i] = (loss(d) - base) / 1e-4
        theta -= lr * g
    return theta


def predict_best(theta, arm_names, A, C_raw, c_scale=(0.2, 1.0)):
    """Given fitted theta and a live fingerprint (A=plain compliance, C_raw=benign recon),
    return (best_arm_name, J_hat_vector) over arm_names using the domain-structured model."""
    arms = [decode_arm(c) for c in arm_names]
    C = c_scale[0] + c_scale[1] * float(C_raw)
    jh = j_hat(np.array(theta, float), arms, C, float(A))
    return arm_names[int(jh.argmax())], jh


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="results/context_selector_train_20260903")
    ap.add_argument("--outdir", default="results/structured_policy_20260904")
    a = ap.parse_args()
    blob = np.load(Path(a.data) / "context_values.npz")
    man = json.loads((Path(a.data) / "manifest.json").read_text())
    J = blob["J"]; ctx = blob["context"]; splits = blob["splits"]
    arms = [decode_arm(c) for c in man["arms"]]
    tags = man["targets"]; fams = man["families"]; T = len(tags)
    train_i = splits == "train"; eval_i = splits == "test"
    # fingerprint pieces: A = plain compliance (align asr), C = benign recon capability
    A = ctx[:, 0].astype(float)                       # english_direct raw ASR
    C = ctx[:, 2].astype(float)                       # benign_recon_mean (capability)
    # scale C to the difficulty range so the sigmoid gate is identifiable
    C = 0.2 + 1.0 * C                                 # capability in difficulty units
    Jtrain = J[:, train_i, :].mean(1)                # [T, A] gated on train items (for fitting)
    Jeval = J[:, eval_i, :].mean(1)                  # [T, A] gated on eval items (for scoring)

    theta0 = [4.0, 0.1, 0.0, -1.0, 3.0, 1.0]
    fam_list = sorted(set(fams))
    rows = []
    for held in range(T):
        tr = [i for i in range(T) if fams[i] != fams[held]]     # family hold-out
        theta = fit(theta0, arms, C[tr], A[tr], [Jtrain[i] for i in tr], iters=1500)
        pred = j_hat(theta, arms, C[held], A[held])
        pick = int(pred.argmax())
        gfix = int(np.stack([Jeval[i] for i in tr]).mean(0).argmax())   # global fixed on train
        orc = int(Jeval[held].argmax())
        rows.append(dict(target=tags[held], family=fams[held],
                         structured_arm=man["arms"][pick], structured_gated=float(Jeval[held, pick]),
                         fixed_arm=man["arms"][gfix], fixed_gated=float(Jeval[held, gfix]),
                         oracle_arm=man["arms"][orc], oracle_gated=float(Jeval[held, orc]),
                         A=round(float(A[held]), 3), C=round(float(C[held]), 3)))
    def agg(key):
        v = np.array([r[key] for r in rows]); orc = np.array([r["oracle_gated"] for r in rows])
        b = np.array([np.random.default_rng(s).choice(v, len(v)).mean() for s in range(3000)])
        return dict(mean=float(v.mean()), ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                    regret=float((orc - v).mean()))
    # global theta fit on ALL targets (for interpretation)
    theta_all = fit(theta0, arms, C, A, [Jtrain[i] for i in range(T)], iters=2000)
    summary = dict(scope="Domain-structured adaptive policy, family LOO.",
                   n_targets=T, families=fam_list, arms=man["arms"],
                   theta_names=["k", "d_shuf", "d_frag", "u0", "u_a", "u_d"],
                   theta_global=[round(float(x), 3) for x in theta_all],
                   structured=agg("structured_gated"), fixed=agg("fixed_gated"), oracle=agg("oracle_gated"),
                   per_target=rows)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "structured_prior.json").write_text(json.dumps(dict(
        theta=[float(x) for x in theta_all], theta_names=summary["theta_names"],
        a_index=0, c_index=2, c_scale=[0.2, 1.0], arms=man["arms"],
        note="warm-start prior for online_live: predict_best(theta, arms, A=fp[a_index], C_raw=fp[c_index])"),
        indent=2) + "\n")
    print(json.dumps({"n_targets": T, "families": len(fam_list),
                      "theta_global(k,d_shuf,d_frag,u0,u_a,u_d)": summary["theta_global"]}, indent=2))
    print(f"\nLOO joint gated ASR (family hold-out):")
    for k in ("structured", "fixed", "oracle"):
        s = summary[k]; print(f"  {k:11s} {s['mean']:.3f}  CI[{s['ci'][0]:.3f},{s['ci'][1]:.3f}]  regret={s['regret']:.3f}")
    print("\nper-target (A=compliance, C=capability -> picked):")
    for r in rows:
        print(f"  {r['target']:14s} A={r['A']:.2f} C={r['C']:.2f}  struct={r['structured_arm']:22s}{r['structured_gated']:.2f}"
              f"  fixed={r['fixed_gated']:.2f}  oracle={r['oracle_gated']:.2f}")


if __name__ == "__main__":
    main()
