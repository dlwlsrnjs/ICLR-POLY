#!/usr/bin/env python3
"""Three checks a reviewer asked for, all offline on the stored arm matrices.

1. Adaptive over the published single-vector attacks. Trying AIM, DeepInception and PAP once each
   costs exactly three harmful queries and needs no configuration space, no probe and no GP. It is
   the strongest cheap baseline and the paper has to beat it, not just their fixed averages.

2. The winner's curse in the oracle. The per-target oracle is an in-sample maximum over 23 arms
   measured on 40 to 64 items, so it is biased upward. We estimate the bias by simulating a target
   whose arms are all equal to that target's mean and taking the same maximum, and report the
   debiased ceiling.

3. Leave-one-family-out. The cumulative ablation adds families in one order, which flatters whichever
   family is added first. Removing each family from the full space says what it is actually worth.

Writes results/robustness_20260908.json, paper/tab_sel_robust.tex and macros.
"""
from __future__ import annotations
import io, json, contextlib, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
with contextlib.redirect_stdout(io.StringIO()):
    import mj_bandit_full as MJ
    import lingua_bandit_full as LG

DS = (("MultiJail", MJ, 64), ("Lingua-SafetyBench", LG, 40))
SINGLE = ("m_aim", "m_deepinception", "m_pap")
FAMILIES = {"amount": "amt_", "disorder": "dis_", "composition": "combo_",
            "role separation": "tri_", "single-vector": "m_"}
REPS = 4000
rng = np.random.default_rng(0)


def noisy(p, n):
    return np.clip(p + rng.normal(0, np.sqrt(np.maximum(p * (1 - p), .01) / n)), 0, 1)


def main():
    out = {}
    for name, mod, n_items in DS:
        models = mod.STRONG if hasattr(mod, "STRONG") else mod.MODELS
        names, G = mod.names, mod.G
        idx = {a: i for i, a in enumerate(names)}

        # 1. try the three published attacks once each, keep the best observed
        vals = []
        for _ in range(REPS):
            per = []
            for t in models:
                true = np.array([G[t][idx[a]] for a in SINGLE])
                per.append(true[int(np.argmax(noisy(true, n_items)))])
            vals.append(np.mean(per))
        base3 = float(np.mean(vals))
        base3_oracle = float(np.mean([max(G[t][idx[a]] for a in SINGLE) for t in models]))

        # 2. winner's curse, applied symmetrically. A max over 23 arms is far more biased than a max
        #    over 3, so debiasing only our oracle would flatter the comparison.
        def curse(arms_idx):
            b = []
            for t in models:
                m = float(np.mean([G[t][i] for i in arms_idx]))
                draws = noisy(np.full((REPS, len(arms_idx)), m), n_items).max(1)
                b.append(float(np.mean(draws)) - m)
            return float(np.mean(b))
        bias = [curse(range(len(names)))]
        bias3 = curse([idx[a] for a in SINGLE])
        oracle = float(np.mean([G[t].max() for t in models]))
        fixed_arm = int(np.argmax(np.mean([G[t] for t in models], axis=0)))
        fixed = float(np.mean([G[t][fixed_arm] for t in models]))

        # 3. leave-one-family-out on the full space
        full = oracle
        loo = {}
        for fam, pre in FAMILIES.items():
            # single-vector = the three transforms only, not the translated baseline (also m_-prefixed)
            drop = (lambda a: a.startswith(pre) and a != "m_translated") if fam == "single-vector" \
                   else (lambda a: a.startswith(pre))
            keep = [i for i, a in enumerate(names) if not drop(a)]
            loo[fam] = round(full - float(np.mean([max(G[t][i] for i in keep) for t in models])), 3)
        keep = [i for i, a in enumerate(names) if a != "m_translated"]
        loo["translation"] = round(full - float(np.mean([max(G[t][i] for i in keep) for t in models])), 3)

        out[name] = {
            "n_items": n_items,
            "three_published_attacks_adaptive": round(base3, 3),
            "three_published_attacks_oracle": round(base3_oracle, 3),
            "three_published_attacks_curse": round(bias3, 3),
            "three_published_attacks_oracle_debiased": round(base3_oracle - bias3, 3),
            "ceiling_gain_raw": round(oracle - base3_oracle, 3),
            "ceiling_gain_debiased": round((oracle - float(np.mean(bias))) - (base3_oracle - bias3), 3),
            "oracle": round(oracle, 3),
            "oracle_winners_curse": round(float(np.mean(bias)), 3),
            "oracle_debiased": round(oracle - float(np.mean(bias)), 3),
            "best_fixed": round(fixed, 3), "best_fixed_arm": names[fixed_arm],
            "leave_one_family_out": loo,
        }
        print(f"=== {name} ===")
        print(f"  세 공개 공격을 한 번씩 (3쿼리): {base3:.3f}   그 오라클: {base3_oracle:.3f}")
        print(f"  전체 오라클 {oracle:.3f}  승자의 저주 보정 -{np.mean(bias):.3f}  → {oracle-np.mean(bias):.3f}")
        print(f"  최선 고정 {fixed:.3f} ({names[fixed_arm]})")
        print(f"  계열 제거 시 오라클 손실: {loo}")
    Path("results/robustness_20260908.json").write_text(json.dumps(out, indent=2))

    L = ["\\begin{table}[t]", "\\centering", "\\small",
         "\\caption{Debiasing and family-ablation checks. \\emph{Oracle, debiased} subtracts the "
         "winner's curse of taking a maximum over 23 arms measured on this many items, estimated by "
         "simulating a target whose arms are all equal. \\emph{Family removed} is the oracle lost by "
         "deleting one family from the full space, which is what each family is worth once the others "
         "are present. Best-fixed and published-attack baselines are in Table~\\ref{tab:sel_main}.}",
         "\\label{tab:sel_robust}", "\\begin{tabular}{lcc}", "\\toprule",
         "& MultiJail & Lingua-SafetyBench \\\\", "\\midrule"]
    k = list(out)
    L.append("ceiling over 23 configurations, as measured & " + " & ".join(f"{out[n]['oracle']:.3f}" for n in k) + " \\\\")
    L.append("\\quad winner's curse over 23 arms & " +
             " & ".join(f"$-${out[n]['oracle_winners_curse']:.3f}" for n in k) + " \\\\")
    L.append("\\quad debiased & " + " & ".join(f"{out[n]['oracle_debiased']:.3f}" for n in k) + " \\\\")
    L.append("ceiling over the 3 published attacks & " + " & ".join(f"{out[n]['three_published_attacks_oracle']:.3f}" for n in k) + " \\\\")
    L.append("\\quad winner's curse over 3 arms & " +
             " & ".join(f"$-${out[n]['three_published_attacks_curse']:.3f}" for n in k) + " \\\\")
    L.append("\\quad debiased & " + " & ".join(f"{out[n]['three_published_attacks_oracle_debiased']:.3f}" for n in k) + " \\\\")
    L.append("\\midrule")
    L.append("ceiling gain from the richer space, as measured & " + " & ".join(f"{out[n]['ceiling_gain_raw']:+.3f}" for n in k) + " \\\\")
    L.append("ceiling gain, both debiased & " + " & ".join(f"{out[n]['ceiling_gain_debiased']:+.3f}" for n in k) + " \\\\")
    L.append("\\midrule")
    for fam in list(FAMILIES) + ["translation"]:
        L.append(f"oracle lost if {fam} removed & " +
                 " & ".join(f"{out[n]['leave_one_family_out'][fam]:.3f}" for n in k) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    Path("paper/tab_sel_robust.tex").write_text("\n".join(L))

    DIG = str.maketrans({"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
                         "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"})
    lines = ["% auto-generated by scripts/robustness_checks.py -- do not edit"]
    for n in k:
        key = "MJ" if n == "MultiJail" else "LG"
        o = out[n]
        lines += [f"\\newcommand{{\\pjbase{key}}}{{{o['three_published_attacks_adaptive']:.3f}}}",
                  f"\\newcommand{{\\pjbaseorc{key}}}{{{o['three_published_attacks_oracle']:.3f}}}",
                  f"\\newcommand{{\\pjcurse{key}}}{{{o['oracle_winners_curse']:.3f}}}",
                  f"\\newcommand{{\\pjoracledeb{key}}}{{{o['oracle_debiased']:.3f}}}",
                  f"\\newcommand{{\\pjcursethree{key}}}{{{o['three_published_attacks_curse']:.3f}}}",
                  f"\\newcommand{{\\pjbaseorcdeb{key}}}{{{o['three_published_attacks_oracle_debiased']:.3f}}}",
                  f"\\newcommand{{\\pjgainraw{key}}}{{{o['ceiling_gain_raw']:+.3f}}}",
                  f"\\newcommand{{\\pjgaindeb{key}}}{{{o['ceiling_gain_debiased']:+.3f}}}"]
        for fam in list(FAMILIES) + ["translation"]:
            tag = fam.split()[0].replace("-", "")
            lines.append(f"\\newcommand{{\\pjloo{tag}{key}}}{{{o['leave_one_family_out'][fam]:.3f}}}")
    Path("paper/robust_numbers.tex").write_text("\n".join(lines) + "\n")
    print("\nwrote paper/tab_sel_robust.tex, paper/robust_numbers.tex")


if __name__ == "__main__":
    raise SystemExit(main())
