#!/usr/bin/env python3
"""Is the best configuration stable inside a target, across harm domains?

Reads the per-item judgments written by domain_breakdown.py (which stores each item's scenario next
to its verified outcome) and asks, per target: does the configuration that wins on the target's
average also win on each individual domain? Scoring follows scripts/arm_scoring.py, so the
clear-text single-vector configuration is ungated and the hidden-request ones are gated.

Writes results/domain_analysis_20260908.json, paper/tab_sel_domain.tex and paper/domain_numbers.tex.
Offline, no GPU.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))

PRETTY = {"qwen25_3b": "Qwen2.5-3B", "qwen25_7b": "Qwen2.5-7B", "qwen25_14b": "Qwen2.5-14B",
          "qwen25_32b": "Qwen2.5-32B", "llama32_3b_it": "Llama-3.2-3B", "llama31_8b_it": "Llama-3.1-8B",
          "gemma2_2b_it": "Gemma-2-2B", "gemma2_9b_it": "Gemma-2-9B", "gemma2_27b": "Gemma-2-27B"}
ORDER = ["qwen25_3b", "qwen25_7b", "qwen25_14b", "qwen25_32b", "llama32_3b_it",
         "llama31_8b_it", "gemma2_2b_it", "gemma2_9b_it", "gemma2_27b"]
CFG = {"amt_n4": "amount $n{=}4$", "combo_ours_persona": "composition+persona",
       "tri_hi_wl": "role-split", "tri_triple": "role-split+persona", "m_aim": "AIM"}
COLLECTIONS = {"Lingua-SafetyBench": "results/domain_breakdown_20260907",
               "MultiJail": "results/domain_breakdown_mj_20260908"}
MIN_ITEMS = 10          # a domain with fewer items is too small to call a winner


def domain_of(scenario):
    """MultiJail stores a multi-label list as a (sometimes truncated) string; use the first label."""
    s = str(scenario).strip()
    if s.startswith("["):
        s = s[1:].lstrip("'\"")
        for sep in ("',", '",'):
            if sep in s:
                s = s.split(sep)[0]
        s = s.rstrip("]'\" ")
    return s[:34]


def analyse(path):
    out = {}
    for t in ORDER:
        f = Path(path) / f"{t}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        cfgs = [c for c in CFG if c in d["configs"]]
        overall = {c: mean(it["gated"] for it in d["configs"][c]["items"]) for c in cfgs}
        avg_best = max(overall, key=overall.get)
        by = {}
        for c in cfgs:
            for it in d["configs"][c]["items"]:
                by.setdefault(domain_of(it["scenario"]), {}).setdefault(c, []).append(it["gated"])
        doms = {k: v for k, v in by.items() if len(v[cfgs[0]]) >= MIN_ITEMS}
        rows, wins, losses = {}, {}, []
        for dom, per in sorted(doms.items()):
            m = {c: mean(per[c]) for c in cfgs}
            b = max(m, key=m.get)
            rows[dom] = {"n": len(per[cfgs[0]]), "best": b, "best_value": round(m[b], 3),
                         "avg_best_value": round(m[avg_best], 3)}
            wins[b] = wins.get(b, 0) + 1
            losses.append(m[b] - m[avg_best])
        out[t] = {"avg_best": avg_best, "avg_best_overall": round(overall[avg_best], 3),
                  "n_domains": len(doms), "avg_best_wins": wins.get(avg_best, 0),
                  "distinct_winners": len(wins), "max_gap": round(max(losses), 3) if losses else 0.0,
                  "mean_gap": round(mean(losses), 3) if losses else 0.0,
                  "per_domain": rows, "overall": {c: round(v, 3) for c, v in overall.items()}}
    return out


def main():
    res = {name: analyse(p) for name, p in COLLECTIONS.items()}
    res = {k: v for k, v in res.items() if v}
    Path("results/domain_analysis_20260908.json").write_text(json.dumps(res, indent=2))

    L = ["\\begin{table}[t]", "\\centering", "\\small",
         "\\caption{The configuration chosen on a target's average is usually not the best choice for "
         "its individual harm domains. \\emph{Average-best wins} counts the domains on which the "
         "average-best configuration is still best; \\emph{distinct winners} counts how many different "
         "configurations win at least one domain; \\emph{max gap} is the largest verified-ASR loss "
         "from using the average-best configuration in a domain instead of that domain's best.}",
         "\\label{tab:sel_domain}", "\\begin{tabular}{llccc}", "\\toprule",
         "Collection & Target & average-best wins & distinct winners & max gap \\\\", "\\midrule"]
    macros = {}
    for name, per in res.items():
        first = True
        for t in ORDER:
            if t not in per:
                continue
            r = per[t]
            lead = name if first else ""
            first = False
            L.append(f"{lead} & {PRETTY[t]} & {r['avg_best_wins']}/{r['n_domains']} & "
                     f"{r['distinct_winners']} & {r['max_gap']:.3f} \\\\")
        vals = [per[t] for t in ORDER if t in per]
        if vals:
            L.append("\\cmidrule(lr){2-5}")
            L.append(f" & mean & {mean(v['avg_best_wins'] for v in vals):.1f}/"
                     f"{mean(v['n_domains'] for v in vals):.0f} & "
                     f"{mean(v['distinct_winners'] for v in vals):.1f} & "
                     f"{mean(v['max_gap'] for v in vals):.3f} \\\\")
            key = "MJ" if name == "MultiJail" else "LG"
            macros[f"dom{key}targets"] = str(len(vals))
            macros[f"dom{key}wins"] = f"{mean(v['avg_best_wins'] for v in vals):.1f}"
            macros[f"dom{key}doms"] = f"{mean(v['n_domains'] for v in vals):.0f}"
            macros[f"dom{key}distinct"] = f"{mean(v['distinct_winners'] for v in vals):.1f}"
            macros[f"dom{key}maxgap"] = f"{max(v['max_gap'] for v in vals):.3f}"
            macros[f"dom{key}meangap"] = f"{mean(v['mean_gap'] for v in vals):.3f}"
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    Path("paper/tab_sel_domain.tex").write_text("\n".join(L))

    DIG = str.maketrans({"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
                         "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"})
    lines = ["% auto-generated by scripts/domain_analysis.py -- do not edit"]
    for k, v in sorted(macros.items()):
        lines.append(f"\\newcommand{{\\pj{k.translate(DIG)}}}{{{v}}}")
    Path("paper/domain_numbers.tex").write_text("\n".join(lines) + "\n")

    for name, per in res.items():
        print(f"=== {name} ===")
        for t in ORDER:
            if t not in per:
                continue
            r = per[t]
            print(f"  {PRETTY[t]:14s} avg-best {r['avg_best']:20s} wins {r['avg_best_wins']}/{r['n_domains']}"
                  f"  distinct {r['distinct_winners']}  max gap {r['max_gap']:.3f}")
    print("\nwrote paper/tab_sel_domain.tex, paper/domain_numbers.tex, results/domain_analysis_20260908.json")


if __name__ == "__main__":
    raise SystemExit(main())
