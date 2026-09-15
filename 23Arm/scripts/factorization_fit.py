#!/usr/bin/env python3
"""Empirical check of the working factorization verified(c) ~= recon(c) * comply(c) that motivates the
plaintext warm start (reviewer W1). Offline over the measured arm matrices, no GPU.

We restrict to the HIDDEN (reconstruction-gated) arms -- for the clear-text arms recon==1 and the
score IS the unsafe rate, so the factorization is trivially exact and uninformative. For each panel
target and each hidden configuration c we read:
    verified(c) = gated = P(reconstruct AND unsafe)
    recon(c)    = P(reconstruct)                      (measurable on harmless plaintext)
    Umarg(c)    = P(unsafe)                            (marginal answer rate)
    cond(c)     = verified(c)/recon(c) = P(unsafe | reconstruct)  (the true conditional willingness)

The approximation the paper uses treats comply as roughly configuration-independent, i.e.
verified(c) ~= recon(c) * Umarg(c). We report (i) how well that product tracks verified across
configurations (Pearson r, mean absolute error), and (ii) how much the conditional willingness cond(c)
actually varies across configurations within a target (its spread) -- the quantity that decides whether
the approximation is a law or a heuristic. We also report how well recon alone ranks verified
(Spearman), which is the warm start's actual job.

Writes paper/factorization_numbers.tex and results/factorization_fit_20260908.json.
"""
from __future__ import annotations
import io, json, contextlib, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
with contextlib.redirect_stdout(io.StringIO()):
    import mj_bandit_full as MJ
    import lingua_bandit_full as LG
from arm_scoring import CLEARTEXT_ARMS


def _spy(mod):
    """Wrap the module's arm_score so we capture the raw (gated, unsafe) per arm name for the last
    arms_for() call. mod._row looks up arm_score in module globals at call time, so patching the
    module attribute is enough (and reuses the exact, canonical arm loader)."""
    cap = {}
    orig = mod.arm_score

    def spy(name, gated, unsafe):
        cap[name] = (float(gated), float(unsafe))
        return orig(name, gated, unsafe)
    mod.arm_score = spy
    return cap, (lambda: setattr(mod, "arm_score", orig))


def _panel(mod):
    return getattr(mod, "STRONG", None) or mod.MODELS


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() < 1e-9 or y.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    def rank(v):
        order = np.argsort(np.argsort(v))
        return order.astype(float)
    return pearson(rank(np.asarray(x, float)), rank(np.asarray(y, float)))


def collect(mod):
    cap, restore = _spy(mod)
    rows = {}
    for t in _panel(mod):
        arms = mod.arms_for(t)          # (name, score, recon, feat); populates cap for t
        recs = []
        for (name, score, recon, _feat) in arms:
            if name in CLEARTEXT_ARMS:
                continue                # hidden arms only
            gated, unsafe = cap[name]   # raw marginal unsafe from the spy
            recs.append(dict(name=name, ver=gated, recon=recon, umarg=unsafe,
                             cond=(gated / recon if recon > 1e-6 else float("nan"))))
        rows[t] = recs
    restore()
    return rows


def analyse(rows):
    per = {}
    pool_ver, pool_prod, pool_recon = [], [], []
    for t, recs in rows.items():
        ver = np.array([r["ver"] for r in recs])
        recon = np.array([r["recon"] for r in recs])
        umarg = np.array([r["umarg"] for r in recs])
        cond = np.array([r["cond"] for r in recs])
        prod = recon * umarg
        per[t] = dict(n=len(recs), r_prod=pearson(ver, prod),
                      mae_prod=float(np.mean(np.abs(ver - prod))),
                      rho_recon=spearman(recon, ver),
                      cond_mean=float(np.nanmean(cond)), cond_std=float(np.nanstd(cond)))
        pool_ver += list(ver); pool_prod += list(prod); pool_recon += list(recon)
    agg = dict(n_arms=per[next(iter(per))]["n"], n_targets=len(rows),
               r_prod_pool=pearson(pool_ver, pool_prod),
               mae_prod_pool=float(np.mean(np.abs(np.array(pool_ver) - np.array(pool_prod)))),
               rho_recon_pool=spearman(pool_recon, pool_ver),
               cond_std_mean=float(np.mean([per[t]["cond_std"] for t in per])),
               cond_std_max=float(np.max([per[t]["cond_std"] for t in per])))
    return per, agg


def main():
    out, macros = {}, []
    for key, mod in (("MJ", MJ), ("LG", LG)):
        rows = collect(mod)
        per, agg = analyse(rows)
        out[key] = dict(per_target={t: per[t] for t in per}, agg=agg)
        macros += [
            f"\\newcommand{{\\pjfactr{key}}}{{{agg['r_prod_pool']:.2f}}}",
            f"\\newcommand{{\\pjfactmae{key}}}{{{agg['mae_prod_pool']:.3f}}}",
            f"\\newcommand{{\\pjfactrho{key}}}{{{agg['rho_recon_pool']:.2f}}}",
            f"\\newcommand{{\\pjfactcondstd{key}}}{{{agg['cond_std_mean']:.3f}}}",
            f"\\newcommand{{\\pjfactcondstdmax{key}}}{{{agg['cond_std_max']:.3f}}}",
        ]
        print(f"=== {key}: {agg['n_arms']} hidden arms x {agg['n_targets']} targets ===")
        print(f"  pooled Pearson r(verified, recon*Umarg) = {agg['r_prod_pool']:.3f}")
        print(f"  pooled MAE |verified - recon*Umarg|      = {agg['mae_prod_pool']:.3f}")
        print(f"  pooled Spearman rho(recon, verified)     = {agg['rho_recon_pool']:.3f}")
        print(f"  conditional P(U|R) std across configs: mean {agg['cond_std_mean']:.3f}  max {agg['cond_std_max']:.3f}")
        for t in per:
            p = per[t]
            print(f"    {t:15s} n={p['n']} r_prod={p['r_prod']:.3f} mae={p['mae_prod']:.3f} "
                  f"rho_recon={p['rho_recon']:.3f} cond={p['cond_mean']:.3f}+-{p['cond_std']:.3f}")
    macros.append(f"\\newcommand{{\\pjfactnhidden}}{{{out['MJ']['agg']['n_arms']}}}")
    Path("results/factorization_fit_20260908.json").write_text(json.dumps(out, indent=2))
    Path("paper/factorization_numbers.tex").write_text(
        "% auto-generated by scripts/factorization_fit.py -- do not edit\n" + "\n".join(macros) + "\n")
    print("wrote paper/factorization_numbers.tex, results/factorization_fit_20260908.json")


if __name__ == "__main__":
    raise SystemExit(main())
