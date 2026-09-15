#!/usr/bin/env python3
"""How much is a per-target oracle inflated by being an in-sample maximum?

The first version of this correction assumed every configuration was equally good, which is the most
conservative null: it charges the maximum possible selection bias. Where per-item outcomes exist we
can measure the bias instead of bounding it, by choosing the best configuration on one random half of
the items and scoring it on the other half. On the domain study's five configurations that split-half
estimate is about a fifth of the all-equal bound, so using the bound would over-correct.

This script reports both, and applies the measured (split-half) correction to the 23-configuration
and 3-configuration ceilings by scaling the all-equal bound with the ratio measured on the data we
have per-item outcomes for. The scaling is stated as such: it assumes the ratio of true bias to
worst-case bias carries from five configurations to twenty-three, which is the weakest assumption we
could find that still uses a measurement rather than a bound.

Writes results/winners_curse_20260908.json and macros in paper/robust_numbers.tex.
"""
from __future__ import annotations
import io, json, contextlib, re, sys
from pathlib import Path
from statistics import mean
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
with contextlib.redirect_stdout(io.StringIO()):
    import mj_bandit_full as MJ
    import lingua_bandit_full as LG
from domain_analysis import ORDER, CFG, COLLECTIONS  # noqa: E402

REPS = 3000
SINGLE = ("m_aim", "m_deepinception", "m_pap")
rng = np.random.default_rng(0)


def all_equal_bias(level, n_arms, n_items, reps=REPS):
    """Expected in-sample maximum minus the truth when every arm equals `level`."""
    sd = np.sqrt(max(level * (1 - level), .01) / n_items)
    draws = np.clip(level + rng.normal(0, sd, size=(reps, n_arms)), 0, 1).max(1)
    return float(np.mean(draws)) - level


def split_half(path):
    """(measured bias, all-equal bias) on the per-item data, averaged over targets."""
    meas, bound = [], []
    for t in ORDER:
        f = Path(path) / f"{t}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        cfgs = [c for c in CFG if c in d["configs"]]
        ids = [it["item_id"] for it in d["configs"][cfgs[0]]["items"]]
        pos = {k: i for i, k in enumerate(ids)}
        M = np.zeros((len(ids), len(cfgs)))
        for j, c in enumerate(cfgs):
            for it in d["configs"][c]["items"]:
                M[pos[it["item_id"]], j] = it["gated"]
        n, full = len(ids), M.mean(0)
        ins, out = [], []
        for _ in range(REPS):
            p = rng.permutation(n); h1, h2 = p[: n // 2], p[n // 2:]
            j = int(np.argmax(M[h1].mean(0)))
            ins.append(full.max()); out.append(M[h2].mean(0)[j])
        meas.append(float(np.mean(ins)) - float(np.mean(out)))
        bound.append(all_equal_bias(float(full.mean()), len(cfgs), n))
    return (mean(meas), mean(bound), len(meas)) if meas else (None, None, 0)


def main():
    out = {}
    for name, mod, n_items in (("MultiJail", MJ, 64), ("Lingua-SafetyBench", LG, 40)):
        models = mod.STRONG if hasattr(mod, "STRONG") else mod.MODELS
        names, G = mod.names, mod.G
        idx = {a: i for i, a in enumerate(names)}
        meas, bound, n_t = split_half(COLLECTIONS[name])
        ratio = (meas / bound) if (meas is not None and bound) else None

        oracle = mean(G[t].max() for t in models)
        base_oracle = mean(max(G[t][idx[a]] for a in SINGLE) for t in models)
        b23 = mean(all_equal_bias(float(np.mean(G[t])), len(names), n_items) for t in models)
        b3 = mean(all_equal_bias(float(np.mean([G[t][idx[a]] for a in SINGLE])), len(SINGLE), n_items)
                  for t in models)
        out[name] = {
            "per_item_targets": n_t,
            "split_half_bias": round(meas, 4) if meas is not None else None,
            "all_equal_bias_same_setting": round(bound, 4) if bound is not None else None,
            "ratio_measured_to_bound": round(ratio, 3) if ratio else None,
            "oracle": round(oracle, 3), "base_oracle": round(base_oracle, 3),
            "bound_23": round(b23, 3), "bound_3": round(b3, 3),
            "corrected_23": round(oracle - b23 * (ratio or 1), 3),
            "corrected_3": round(base_oracle - b3 * (ratio or 1), 3),
            "gain_raw": round(oracle - base_oracle, 3),
            "gain_bound": round((oracle - b23) - (base_oracle - b3), 3),
            "gain_measured": round((oracle - b23 * (ratio or 1)) - (base_oracle - b3 * (ratio or 1)), 3),
        }
        o = out[name]
        print(f"=== {name} ===")
        print(f"  per-item split-half bias {o['split_half_bias']:+.4f} vs all-equal bound "
              f"{o['all_equal_bias_same_setting']:+.4f}  (ratio {o['ratio_measured_to_bound']})")
        print(f"  ceiling 23 arms {o['oracle']:.3f} -> {o['corrected_23']:.3f}; "
              f"3 arms {o['base_oracle']:.3f} -> {o['corrected_3']:.3f}")
        print(f"  gain: raw {o['gain_raw']:+.3f}  worst-case-corrected {o['gain_bound']:+.3f}  "
              f"measured-corrected {o['gain_measured']:+.3f}")
    Path("results/winners_curse_20260908.json").write_text(json.dumps(out, indent=2))

    lines = []
    for name in out:
        key = "MJ" if name == "MultiJail" else "LG"
        o = out[name]
        lines += [f"\\newcommand{{\\pjcurseratio{key}}}{{{o['ratio_measured_to_bound']:.2f}}}",
                  f"\\newcommand{{\\pjceilcorr{key}}}{{{o['corrected_23']:.3f}}}",
                  f"\\newcommand{{\\pjbasecorr{key}}}{{{o['corrected_3']:.3f}}}",
                  f"\\newcommand{{\\pjgainmeas{key}}}{{{o['gain_measured']:+.3f}}}"]
    p = Path("paper/robust_numbers.tex")
    txt = p.read_text()
    start, end = "% begin winners_curse", "% end winners_curse"
    body = "\n".join([start] + lines + [end])
    txt = (re.sub(re.escape(start) + r".*?" + re.escape(end), body, txt, flags=re.S)
           if start in txt else txt.rstrip("\n") + "\n" + body + "\n")
    p.write_text(txt)
    print("\nwrote results/winners_curse_20260908.json and macros")


if __name__ == "__main__":
    raise SystemExit(main())
