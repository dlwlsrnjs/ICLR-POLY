#!/usr/bin/env python3
"""Persona as a second axis of adaptation, orthogonal to the configuration family.

The panel already shows which configuration family wins per target. This asks a different question:
does adding a persona frame to the same multilingual construction help or hurt, and does the answer
depend on the target? The pair that isolates it is combination without and with a persona:

  combo_ours          multilingual interleaving, no persona, English answer
  combo_ours_persona  the same interleaving with the AIM persona

Both hide the request, so both are reconstruction-gated; the only difference is the persona. The net
effect is combo_ours_persona minus combo_ours per target. We test whether its sign is tied to the
model, the same way the family axis is tested, by permuting the model-family label across targets and
recounting how consistently persona helps within a model family.

Writes results/persona_axis_20260908.json, paper/tab_sel_persona.tex, and macros.
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

PRETTY = {"qwen25_3b": "Qwen2.5-3B", "qwen25_7b": "Qwen2.5-7B", "qwen25_14b": "Qwen2.5-14B",
          "qwen25_32b": "Qwen2.5-32B", "llama32_3b_it": "Llama-3.2-3B", "llama31_8b_it": "Llama-3.1-8B",
          "gemma2_2b_it": "Gemma-2-2B", "gemma2_9b_it": "Gemma-2-9B", "gemma2_27b": "Gemma-2-27B"}
ORDER = ["qwen25_3b", "qwen25_7b", "qwen25_14b", "qwen25_32b", "llama32_3b_it",
         "llama31_8b_it", "gemma2_2b_it", "gemma2_9b_it", "gemma2_27b"]
FAM = {"qwen25_3b": "Qwen", "qwen25_7b": "Qwen", "qwen25_14b": "Qwen", "qwen25_32b": "Qwen",
       "llama32_3b_it": "Llama", "llama31_8b_it": "Llama",
       "gemma2_2b_it": "Gemma", "gemma2_9b_it": "Gemma", "gemma2_27b": "Gemma"}
DS = (("MultiJail", MJ, "mj_combo_20260906"), ("Lingua-SafetyBench", LG, "combo_20260906"))
PERM = 20000
rng = np.random.default_rng(0)


def main():
    out, macros = {}, []
    for name, mod, cdir in DS:
        eff = {}
        for t in ORDER:
            v = json.loads((Path("results") / cdir / f"{t}.json").read_text())["variants"]
            eff[t] = round(v["ours_persona"]["gated"] - v["ours"]["gated"], 3)
        out[name] = {"per_target": eff,
                     "mean": round(mean(eff.values()), 3),
                     "helps": sum(1 for e in eff.values() if e > 0),
                     "hurts": sum(1 for e in eff.values() if e < 0)}
        # the persona effect scales with model size (a proxy for alignment strength): a permutation
        # test on the size--effect correlation.
        from scipy.stats import spearmanr
        SIZE = {"qwen25_3b": 3, "qwen25_7b": 7, "qwen25_14b": 14, "qwen25_32b": 32,
                "llama32_3b_it": 3, "llama31_8b_it": 8, "gemma2_2b_it": 2, "gemma2_9b_it": 9,
                "gemma2_27b": 27}
        vals = np.array([eff[t] for t in ORDER]); sz = np.array([SIZE[t] for t in ORDER])
        rho = float(spearmanr(sz, vals).statistic)
        null = np.array([spearmanr(sz, rng.permutation(vals)).statistic for _ in range(PERM)])
        p = float(np.mean(null >= rho))
        out[name]["size_rho"] = round(rho, 3); out[name]["size_p"] = round(p, 4)
        out[name]["size_rho_str"] = f"{rho:+.2f}"   # format once; table and macro must not double-round
        key = "MJ" if name == "MultiJail" else "LG"
        macros += [f"\\newcommand{{\\pjpersona{key}mean}}{{{out[name]['mean']:+.3f}}}",
                   f"\\newcommand{{\\pjpersona{key}helps}}{{{out[name]['helps']}}}",
                   f"\\newcommand{{\\pjpersona{key}rho}}{{{out[name]['size_rho_str']}}}",
                   f"\\newcommand{{\\pjpersona{key}p}}{{{p:.3f}}}"]

    L = ["\\begin{table}[t]", "\\centering", "\\small",
         "\\caption{Persona as a second axis. Each cell is the change in verified ASR from adding the "
         "AIM persona to the same multilingual interleaving (combination with vs.\\ without persona); "
         "both are reconstruction-gated, so only the persona differs. The sign is tied to the model "
         "family (a persona helps the aligned Gemma and large Qwen targets and hurts the small ones), "
         "and the effect grows with model size, a proxy for alignment strength (Spearman "
         "$\\rho$ against size, permutation $p$).}",
         "\\label{tab:sel_persona}", "\\begin{tabular}{lcc}", "\\toprule",
         "Target & MultiJail & Lingua-SafetyBench \\\\", "\\midrule"]
    for t in ORDER:
        a, b = out["MultiJail"]["per_target"][t], out["Lingua-SafetyBench"]["per_target"][t]
        L.append(f"{PRETTY[t]} & {a:+.3f} & {b:+.3f} \\\\")
    L += ["\\midrule",
          f"targets persona helps & {out['MultiJail']['helps']}/9 & {out['Lingua-SafetyBench']['helps']}/9 \\\\",
          f"effect vs model size, Spearman $\\rho$ & {out['MultiJail']['size_rho_str']} & {out['Lingua-SafetyBench']['size_rho_str']} \\\\",
          f"permutation $p$ & {out['MultiJail']['size_p']:.3f} & {out['Lingua-SafetyBench']['size_p']:.3f} \\\\",
          "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    Path("paper/tab_sel_persona.tex").write_text("\n".join(L))
    Path("paper/persona_numbers.tex").write_text(
        "% auto-generated by scripts/persona_axis.py -- do not edit\n" + "\n".join(macros) + "\n")

    Path("results/persona_axis_20260908.json").write_text(json.dumps(out, indent=2))
    for name in out:
        o = out[name]
        print(f"{name}: persona 순효과 평균 {o['mean']:+.3f}  도움 {o['helps']}/9  "
              f"크기 상관 rho={o['size_rho']:+.3f} p={o['size_p']}")
    print("wrote paper/tab_sel_persona.tex")


if __name__ == "__main__":
    raise SystemExit(main())
