#!/usr/bin/env python3
"""Hybrid online adaptation loop (PAIR-style), environment-pluggable.

offline -> universal prior + selection policy (trained elsewhere).
online  -> verify/fine-tune the recommendation on the real target with a few queries:
  1. fingerprint the target with the fixed probe battery (cheap, black-box);
  2. warm-start: policy recommends an arm from the fingerprint (budget 0);
  3. probe that arm on a small calibration batch, read the joint judge score J;
  4. if not yet successful, use the observed feedback to pick the next arm (multi-turn),
     with EARLY STOP on success; report queries-to-first-success and ASR-vs-query.

Environment is pluggable. ReplayEnv uses the already-measured panel_v2 joint table; because
generation is temperature 0 (deterministic), a replayed cell equals the live outcome for any
arm in the menu, so the loop MECHANICS validate faithfully offline. LiveEnv (target vLLM +
the two judges) swaps in later for off-menu/novel configs. Same loop code either way.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


class ReplayEnv:
    """Deterministic replay from the measured joint table (results/context_selector_train_*)."""
    def __init__(self, data):
        blob = np.load(Path(data) / "context_values.npz")
        man = json.loads((Path(data) / "manifest.json").read_text())
        self.J = blob["J"]; self.ctx = blob["context"]; self.splits = blob["splits"]
        self.arms = man["arms"]; self.tags = man["targets"]; self.features = blob["features"]
        self.calib = np.where(self.splits == "train")[0]
        self.evalix = np.where(self.splits == "test")[0]

    def fingerprint(self, ti):
        return self.ctx[ti].copy()

    def probe(self, ti, arm, n_items, rng):
        # a real online query would call the target+judges on n_items; here replay the cells
        items = rng.choice(self.calib, size=min(n_items, len(self.calib)), replace=False)
        return float(np.nanmean(self.J[ti, items, arm])), len(items)

    def eval_gated(self, ti, arm):
        return float(np.nanmean(self.J[ti, self.evalix, arm]))

    def oracle(self, ti):
        g = np.nanmean(self.J[ti][self.evalix], 0)
        return int(np.nanargmax(g)), float(np.nanmax(g))


def gp_recommend(prior, observed, arm_feats, explore=1.5):
    """Fixed-kernel GP posterior mean+ucb over arms given observed (arm->value)."""
    A = len(prior)
    if not observed:
        return int(np.argmax(prior))
    idx = np.array(list(observed)); yv = np.array([observed[a] for a in idx], dtype=float)
    K = 0.04 * np.exp(-0.5 * ((arm_feats[:, None, :] - arm_feats[None, :, :]) ** 2).sum(-1))
    ps = prior[idx]; noise = 0.0025 + np.clip(ps, .05, .95) * (1 - np.clip(ps, .05, .95)) / 8
    Koo_inv = np.linalg.inv(K[np.ix_(idx, idx)] + np.diag(noise))
    Kxi = K[:, idx]                                  # [A, n_obs]
    mean = prior + Kxi @ (Koo_inv @ (yv - ps))       # [A]
    var = np.clip(np.diag(K) - np.einsum("ai,ij,aj->a", Kxi, Koo_inv, Kxi), 0, None)
    ucb = mean + explore * np.sqrt(var)
    ucb[idx] = -1e9                                  # do not re-probe observed arms
    return int(np.argmax(ucb))


def run_target(env, ti, arm_feats, prior, budget, batch, threshold, rng):
    """One online adaptation episode. Anchors (arm 0 and last) are the two free warm probes."""
    fp = env.fingerprint(ti)                       # cheap black-box fingerprint (context)
    observed = {}
    queries = 0
    trajectory = []
    # warm start: prior argmax (a policy would map fp->arm; prior is the context-free fallback)
    order = [int(np.argmax(prior))]
    for _ in range(budget):
        nxt = gp_recommend(prior, observed, arm_feats)
        order.append(nxt)
        # probe current best-unobserved arm
        arm = order[-1]
        val, used = env.probe(ti, arm, batch, rng)
        observed[arm] = val; queries += 1
        eg = env.eval_gated(ti, arm)
        trajectory.append(dict(query=queries, arm=env.arms[arm], calib_J=round(val, 3), eval_gated=round(eg, 3)))
        if eg >= threshold:                        # PAIR-style early stop on success
            break
    # final recommendation = best observed by calib J, else prior argmax
    rec = max(observed, key=observed.get) if observed else int(np.argmax(prior))
    orc_arm, orc_val = env.oracle(ti)
    return dict(target=env.tags[ti], fingerprint=[round(float(x), 3) for x in fp],
                queries=queries, recommended=env.arms[rec], recommended_gated=env.eval_gated(ti, rec),
                first_success_query=next((t["query"] for t in trajectory if t["eval_gated"] >= threshold), None),
                oracle=env.arms[orc_arm], oracle_gated=round(orc_val, 3), trajectory=trajectory)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="results/context_selector_train_20260903")
    ap.add_argument("--env", default="replay", choices=["replay"])  # live env added in Phase B
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--outdir", default="results/online_adapt_20260903")
    a = ap.parse_args()
    env = ReplayEnv(a.data)
    man = json.loads((Path(a.data) / "manifest.json").read_text())
    arm_feats = np.asarray(env.features, np.float32)
    prior = np.nanmean(env.J[:, env.calib, :], (0, 1))   # context-free prior over arms
    rng = np.random.default_rng(20260903)
    rows = [run_target(env, ti, arm_feats, prior, a.budget, a.batch, a.threshold, rng) for ti in range(len(env.tags))]
    q = np.array([r["queries"] for r in rows]); rg = np.array([r["recommended_gated"] for r in rows])
    orc = np.array([r["oracle_gated"] for r in rows])
    fs = [r["first_success_query"] for r in rows]
    summary = dict(env=a.env, budget=a.budget, threshold=a.threshold, n_targets=len(rows),
                   mean_queries=float(q.mean()), mean_recommended_gated=float(rg.mean()),
                   mean_oracle_gated=float(orc.mean()), mean_regret=float((orc - rg).mean()),
                   solved_within_budget=int(sum(x is not None for x in fs)),
                   mean_first_success_query=float(np.mean([x for x in fs if x is not None]) if any(fs) else np.nan),
                   per_target=rows)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("env", "budget", "threshold", "n_targets", "mean_queries",
                      "mean_recommended_gated", "mean_oracle_gated", "mean_regret",
                      "solved_within_budget", "mean_first_success_query")}, indent=2))
    print("\nper-target (queries -> recommended gated / oracle):")
    for r in rows:
        print(f"  {r['target']:14s} q={r['queries']} first_success={r['first_success_query']} "
              f"rec={r['recommended']:22s} {r['recommended_gated']:.3f} / oracle {r['oracle_gated']:.3f}")


if __name__ == "__main__":
    main()
