#!/usr/bin/env python3
"""Generate the LaTeX tables for the adaptive-selector part of the paper directly from the stored
artifacts, so the manuscript can never drift from the numbers on disk.

Reads   results/{mj,lingua}_bandit_full_20260906.json   (arm matrix, oracle, best fixed, budget curve)
        results/bandit_ci_20260907.json                 (bootstrap CIs and paired deltas)
        results/ablation_corr_20260907.json             (arm-family ablation, recon-score Spearman)
        the 90 per-(dir, model) result files            (per-arm scores under the consistent rule)
Writes  paper/tab_sel_main.tex, tab_sel_permodel.tex, tab_sel_budget.tex,
        paper/tab_sel_ablation.tex, tab_sel_corr.tex
        paper/selector_numbers.tex   (macros for every number quoted in the prose)

Scoring follows scripts/arm_scoring.py: arms whose prompt contains the request in clear English are
ungated (their reconstruction gate is vacuous), every hidden-request arm is strictly gated.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_scoring import arm_score  # noqa: E402

R = Path("results"); P = Path("paper")


class PCT(str):
    """percentage, emitted verbatim into the macro file"""
    def __new__(cls, v):
        return super().__new__(cls, f"{100 * v:.0f}\\%")
MODELS = ["qwen25_3b", "qwen25_7b", "qwen25_14b", "qwen25_32b", "llama32_3b_it",
          "llama31_8b_it", "gemma2_2b_it", "gemma2_9b_it", "gemma2_27b"]
PRETTY = {"qwen25_3b": "Qwen2.5-3B", "qwen25_7b": "Qwen2.5-7B", "qwen25_14b": "Qwen2.5-14B",
          "qwen25_32b": "Qwen2.5-32B", "llama32_3b_it": "Llama-3.2-3B", "llama31_8b_it": "Llama-3.1-8B",
          "gemma2_2b_it": "Gemma-2-2B", "gemma2_9b_it": "Gemma-2-9B", "gemma2_27b": "Gemma-2-27B"}
ARM_TEX = {"amount": "amount", "disorder": "disorder", "combo": "composition",
           "triple": "role separation", "single": "single-vector transforms"}
DS = {"MultiJail": dict(seq="mj_sequential_20260906", dis="mj_disorder_20260906", cmb="mj_combo_20260906",
                        meth="mj_method_20260906", tri="mj_triple_20260906",
                        hien="mj_hien_20260908", extra="extra_arms_mj_20260908"),
      "Lingua-SafetyBench": dict(seq="sequential_resource_20260906", dis="disorder_sweep_20260906",
                                 cmb="combo_20260906", meth="method_baselines_v2_20260906",
                                 tri="lingua_triple_20260907",
                                 hien="lingua_hien_20260908", extra="extra_arms_lg_20260908")}
BUDGETS = ["1", "2", "3", "4", "6"]


def arm_raw(ds, t):
    """{arm: stored record}, so a table can show recon and unsafe beside the verified score."""
    d = DS[ds]; out = {}
    for s_ in json.load(open(R / d["seq"] / f"{t}.json"))["steps"]:
        out[f"amt_n{s_['n']}"] = s_
    j = json.load(open(R / d["dis"] / f"{t}.json"))
    for r in (j.get("rows") or j.get("steps")):
        if r["delta"] != 0.0:
            out[f"dis_{r['delta']}"] = r
    for k, v in json.load(open(R / d["cmb"] / f"{t}.json"))["variants"].items():
        out[f"combo_{k}"] = v
    # match the bandit arm space exactly: role separation is the English-answer pair (hi_en from the
    # hien dir, triple_en from the tri dir); translated comes from the extra-arms dir.
    tri = json.load(open(R / d["tri"] / f"{t}.json"))["variants"]
    hien = json.load(open(R / d["hien"] / f"{t}.json"))["variants"]
    out["tri_hi_en"] = hien["hi_en"]; out["tri_triple_en"] = tri["triple_en"]
    me = json.load(open(R / d["meth"] / f"{t}.json"))["methods"]
    for k in ("aim", "deepinception", "pap"):
        out[f"m_{k}"] = me[k]
    out["m_translated"] = json.load(open(R / d["extra"] / f"{t}.json"))["configs"]["translated"]
    return out


def arm_map(ds, t):
    d = DS[ds]; out = {}
    for s in json.load(open(R / d["seq"] / f"{t}.json"))["steps"]:
        out[f"amt_n{s['n']}"] = arm_score(f"amt_n{s['n']}", s["gated"], s["unsafe"])
    j = json.load(open(R / d["dis"] / f"{t}.json"))
    for r in (j.get("rows") or j.get("steps")):
        if r["delta"] != 0.0:
            out[f"dis_{r['delta']}"] = arm_score(f"dis_{r['delta']}", r["gated"], r["unsafe"])
    for k, v in json.load(open(R / d["cmb"] / f"{t}.json"))["variants"].items():
        out[f"combo_{k}"] = arm_score(f"combo_{k}", v["gated"], v["unsafe"])
    tri = json.load(open(R / d["tri"] / f"{t}.json"))["variants"]
    hien = json.load(open(R / d["hien"] / f"{t}.json"))["variants"]
    out["tri_hi_en"] = arm_score("tri_hi_en", hien["hi_en"]["gated"], hien["hi_en"]["unsafe"])
    out["tri_triple_en"] = arm_score("tri_triple_en", tri["triple_en"]["gated"], tri["triple_en"]["unsafe"])
    me = json.load(open(R / d["meth"] / f"{t}.json"))["methods"]
    for k in ("aim", "deepinception", "pap"):
        out[f"m_{k}"] = arm_score(f"m_{k}", me[k]["gated"], me[k]["unsafe"])
    tr = json.load(open(R / d["extra"] / f"{t}.json"))["configs"]["translated"]
    out["m_translated"] = arm_score("m_translated", tr["gated"], tr["unsafe"])
    return out


def arm_label(a):
    if a.startswith("amt_n"):
        return f"amount $n{{=}}{a.split('n')[1]}$"
    if a.startswith("dis_"):
        return f"disorder $\\delta{{=}}{a.split('_')[1]}$"
    return {"combo_ours": "combination", "combo_ours_persona": "combination+persona",
            "combo_ours_incept": "combination+fiction", "combo_incept_only": "fiction only",
            "tri_hi_wl": "role-split (answer LR)", "tri_triple": "role-split+persona",
            "tri_hi_en": "role-split", "tri_triple_en": "role-split+persona",
            "m_aim": "AIM", "m_deepinception": "DeepInception", "m_pap": "PAP",
            "m_translated": "translation"}.get(a, a.replace("_", "\\_"))


def write(name, body):
    (P / name).write_text(body)
    print("wrote", P / name)


def main():
    A = {ds: {t: arm_map(ds, t) for t in MODELS} for ds in DS}
    RAW = {ds: {t: arm_raw(ds, t) for t in MODELS} for ds in DS}
    B = {ds: json.load(open(R / f"{'mj' if ds == 'MultiJail' else 'lingua'}_bandit_full_20260906.json")) for ds in DS}
    CI = json.load(open(R / "bandit_ci_20260907.json"))
    CIK = {"MultiJail": "multijail", "Lingua-SafetyBench": "lingua"}
    ABL = json.load(open(R / "ablation_corr_20260907.json"))
    names = list(A["MultiJail"][MODELS[0]])
    M = {}   # prose macros

    # ---------- main comparison ----------
    # prompt perplexity under a reference LM (scripts/ppl_evasion.py, ppl_singlevector.py); a stealth
    # metric. Only the single-vector attacks have a single fixed prompt template; meta-rows vary.
    ppl_row = {"m_pap": "\\pjpplpap{}", "m_aim": "\\pjpplaim{}", "m_deepinception": "\\pjppldeepinception{}"}
    rows = []
    for arm, lbl in (("m_pap", "PAP \\citep{zeng2024pap}"), ("m_aim", "AIM (persona)"),
                     ("m_deepinception", "DeepInception \\citep{li2023deepinception}")):
        rows.append((lbl, [mean(A[ds][t][arm] for t in MODELS) for ds in DS], ppl_row[arm]))
        for ds in DS:
            M[f"{arm}{'MJ' if ds=='MultiJail' else 'LG'}"] = mean(A[ds][t][arm] for t in MODELS)
    fixed = {ds: B[ds]["fixed_arm"] for ds in DS}
    rows.append((f"best fixed configuration ({arm_label(fixed['MultiJail'])})",
                 [B[ds]["fixed"] for ds in DS], "--"))
    rob = json.load(open(R / "robustness_20260908.json")) if (R / "robustness_20260908.json").exists() else None
    if rob:
        rows.append(("published attacks, adaptive (3 queries)",
                     [rob[ds]["three_published_attacks_adaptive"] for ds in DS], "--"))
    def civ(ds, key):   # "[lo, hi]" 95% bootstrap interval for a CI-dict key
        lo, hi = CI[CIK[ds]][key]["ci95"]
        return f"[{lo:.3f}, {hi:.3f}]"
    v3 = [B[ds]["budget"]["3"]["benign-recon"] for ds in DS]                       # selector @ fixed 3-query
    av = [CI[CIK[ds]]["adaptive[benign-recon]"]["mean"] for ds in DS]              # difficulty-adaptive (headline)
    qv = [CI[CIK[ds]]["mean_queries"] for ds in DS]
    body = ["\\begin{table}[t]", "\\centering", "\\small",
            "\\caption{Main results: verified attack success (reconstruction-gated ASR) averaged over the "
            "nine-model panel under one consistent gate rule (Section~\\ref{sec:gate}), with 95\\% "
            "panel-bootstrap intervals on the ceiling, the best fixed configuration and the adaptive "
            "selector, and the paired per-target differences at the foot. Single-vector baselines receive "
            "the baseline-favourable ungated score; the selector uses no harmful supervision. \\emph{PPL} "
            "is median prompt perplexity under Qwen2.5-7B (a stealth metric): our configurations sit in "
            "the low-perplexity band of the natural-language attacks (Table~\\ref{tab:encoding}), so a "
            "filter tuned to high-perplexity obfuscations does not flag them.}",
            "\\label{tab:sel_main}",
            "\\begin{tabular}{lccc}", "\\toprule",
            "Strategy & MultiJail & Lingua-SafetyBench & PPL \\\\", "\\midrule"]
    for lbl, vals, ppl in rows[:3]:                                               # PAP, AIM, DeepInception
        body.append(f"{lbl} & {vals[0]:.3f} & {vals[1]:.3f} & {ppl} \\\\")
    body.append("\\midrule")
    body.append(f"{rows[3][0]} & {B['MultiJail']['fixed']:.3f} {civ('MultiJail','fixed')} & "
                f"{B['Lingua-SafetyBench']['fixed']:.3f} {civ('Lingua-SafetyBench','fixed')} & -- \\\\")
    if len(rows) > 4:
        body.append(f"{rows[4][0]} & {rows[4][1][0]:.3f} & {rows[4][1][1]:.3f} & -- \\\\")
    body.append("\\midrule")
    body.append(f"selector, fixed 3-query & {v3[0]:.3f} & {v3[1]:.3f} & \\pjpplours{{}} \\\\")
    body.append(f"\\textbf{{selector, difficulty-adaptive}} & \\textbf{{{av[0]:.3f}}} {civ('MultiJail','adaptive[benign-recon]')} & "
                f"\\textbf{{{av[1]:.3f}}} {civ('Lingua-SafetyBench','adaptive[benign-recon]')} & \\pjpplours{{}} \\\\")
    body.append(f"oracle (best arm per model) & {B['MultiJail']['oracle']:.3f} {civ('MultiJail','oracle')} & "
                f"{B['Lingua-SafetyBench']['oracle']:.3f} {civ('Lingua-SafetyBench','oracle')} & -- \\\\")
    body.append("\\midrule")
    d1, d2 = CI["multijail"]["paired_delta[benign-recon]"], CI["lingua"]["paired_delta[benign-recon]"]
    body.append(f"\\quad paired $\\Delta$ vs.\\ best fixed & {d1['mean']:+.3f} [{d1['ci95'][0]:+.3f}, {d1['ci95'][1]:+.3f}] "
                f"& {d2['mean']:+.3f} [{d2['ci95'][0]:+.3f}, {d2['ci95'][1]:+.3f}] & {d1['n_improved']}/{d1['n']}, {d2['n_improved']}/{d2['n']} \\\\")
    kb = "paired_vs_published[benign-recon]"
    if kb in CI["multijail"]:
        b1, b2 = CI["multijail"][kb], CI["lingua"][kb]
        body.append(f"\\quad paired $\\Delta$ vs.\\ published-3 & {b1['mean']:+.3f} [{b1['ci95'][0]:+.3f}, {b1['ci95'][1]:+.3f}] "
                    f"& {b2['mean']:+.3f} [{b2['ci95'][0]:+.3f}, {b2['ci95'][1]:+.3f}] & {b1['n_improved']}/{b1['n']}, {b2['n_improved']}/{b2['n']} \\\\")
        for key, b in (("MJ", b1), ("LG", b2)):
            M[f"vsbase{key}"] = b["mean"]; M[f"vsbaselo{key}"] = b["ci95"][0]; M[f"vsbasehi{key}"] = b["ci95"][1]
    for k, d in (("MJ", d1), ("LG", d2)):
        M[f"delta{k}"] = d["mean"]; M[f"deltalo{k}"] = d["ci95"][0]; M[f"deltahi{k}"] = d["ci95"][1]
    body.append("\\multicolumn{4}{l}{\\footnotesize Adaptive budget: floor 3, cap 8, stop at 0.5; mean queries "
                + f"{qv[0]:.1f} (MultiJail), {qv[1]:.1f} (Lingua). Last column of $\\Delta$ rows: targets improved. "
                + "Bootstrap: 2{,}000 panel draws, 300 for the selector.} \\\\")
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    write("tab_sel_main.tex", "\n".join(body))

    for ds in DS:
        k = "MJ" if ds == "MultiJail" else "LG"
        M[f"oracle{k}"] = B[ds]["oracle"]; M[f"fixed{k}"] = B[ds]["fixed"]
        for b in BUDGETS:
            M[f"ad{b}{k}"] = B[ds]["budget"][b]["benign-recon"]
            M[f"flat{b}{k}"] = B[ds]["budget"][b]["flat"]
            M[f"loo{b}{k}"] = B[ds]["budget"][b]["loo-data"]
        M[f"ratio{k}"] = PCT(B[ds]["budget"]["3"]["benign-recon"] / B[ds]["oracle"])
        M[f"ratiosix{k}"] = PCT(B[ds]["budget"]["6"]["benign-recon"] / B[ds]["oracle"])
        M[f"adapt{k}"] = CI[CIK[ds]]["adaptive[benign-recon]"]["mean"]
        M[f"qbar{k}"] = f"{CI[CIK[ds]]['mean_queries']:.1f}"
        M[f"ratioadapt{k}"] = PCT(CI[CIK[ds]]["adaptive[benign-recon]"]["mean"] / B[ds]["oracle"])

    # ---------- per-model best arm ----------
    body = ["\\begin{table}[t]", "\\centering", "\\small",
            "\\caption{The best configuration is model-specific: it changes across the panel in both "
            "collections, so no fixed attack can be optimal everywhere. Scores follow the gate rule of "
            "Section~\\ref{sec:gate}: hidden-request configurations are reconstruction-gated, clear-text "
            "ones (AIM, DeepInception, PAP, fiction only) are ungated.}",
            "\\label{tab:sel_permodel}",
            "\\begin{tabular}{lcccc}", "\\toprule",
            "\\multirow{2}{*}{Target} & \\multicolumn{2}{c}{MultiJail} & \\multicolumn{2}{c}{Lingua-SafetyBench} \\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
            " & best configuration & ASR & best configuration & ASR \\\\", "\\midrule"]
    # Ties and near-ties matter: with 40-64 items a gap below the cell's standard error is not a
    # real winner. We mark cells whose runner-up is within one standard error, and count distinct
    # winners as a range over how those cells could break.
    N_ITEMS = {"MultiJail": 64, "Lingua-SafetyBench": 40}
    distinct = {ds: set() for ds in DS}
    tied = {ds: set() for ds in DS}
    n_close = 0
    for t in MODELS:
        cells = []
        for ds in DS:
            order = sorted(names, key=lambda a: -A[ds][t][a])
            b, second = order[0], order[1]
            distinct[ds].add(b)
            v = A[ds][t][b]
            se = (max(v * (1 - v), .01) / N_ITEMS[ds]) ** .5
            close = (v - A[ds][t][second]) <= se
            if close:
                tied[ds].add(second); n_close += 1
            mark = "$^{\\dagger}$" if close else ""
            cells += [arm_label(b) + mark, f"{v:.3f}"]
        body.append(f"{PRETTY[t]} & " + " & ".join(cells) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}",
             "\\vspace{2pt}\\footnotesize $^{\\dagger}$ the runner-up is within one standard error of "
             "this cell, so the winner is not resolved by these item counts.",
             "\\end{table}", ""]
    write("tab_sel_permodel.tex", "\n".join(body))
    M["distinctMJ"] = len(distinct["MultiJail"]); M["distinctLG"] = len(distinct["Lingua-SafetyBench"])
    M["distinctALL"] = len(distinct["MultiJail"] | distinct["Lingua-SafetyBench"])
    # how far the count could move if every marked tie broke the other way
    M["distinctALLlo"] = len({b for ds in DS for b in distinct[ds]} - {b for ds in DS for b in tied[ds]})
    M["distinctALLhi"] = len({b for ds in DS for b in (distinct[ds] | tied[ds])})
    M["ntied"] = n_close
    ours = {ds: sum(1 for t in MODELS
                    if max(names, key=lambda a: A[ds][t][a]).startswith(("amt", "dis", "tri"))
                    or max(names, key=lambda a: A[ds][t][a]) in ("combo_ours", "combo_ours_persona", "combo_ours_incept"))
            for ds in DS}
    M["oursMJ"] = ours["MultiJail"]; M["oursLG"] = ours["Lingua-SafetyBench"]

    # ---------- budget curve ----------
    body = ["\\begin{table}[t]", "\\centering", "\\small",
            "\\caption{Query budget against verified ASR for three priors. The benign prior uses only "
            "reconstruction rates observable with harmless puzzles; \\emph{cross-target} uses harmful "
            "results from the other eight models, which a real attacker would not have. Oracle and "
            "best-fixed reference values are in Table~\\ref{tab:sel_main}.}",
            "\\label{tab:sel_budget}",
            "\\begin{tabular}{lccc|ccc}", "\\toprule",
            "& \\multicolumn{3}{c|}{MultiJail} & \\multicolumn{3}{c}{Lingua-SafetyBench} \\\\",
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
            "Budget & uninformed & benign & cross-target & uninformed & benign & cross-target \\\\", "\\midrule"]
    for b in BUDGETS:
        c = []
        for ds in DS:
            r = B[ds]["budget"][b]
            c += [f"{r['flat']:.3f}", f"\\textbf{{{r['benign-recon']:.3f}}}", f"{r['loo-data']:.3f}"]
        body.append(f"{b} & " + " & ".join(c) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    write("tab_sel_budget.tex", "\n".join(body))

    # ---------- bootstrap CIs + paired deltas are now folded into the merged main table
    # (tab_sel_main); here we only (re)compute the prose macros they feed. ----------
    d1, d2 = CI["multijail"]["paired_delta[benign-recon]"], CI["lingua"]["paired_delta[benign-recon]"]
    k1 = "paired_vs_published[benign-recon]"
    if k1 in CI["multijail"]:
        b1, b2 = CI["multijail"][k1], CI["lingua"][k1]
        for key, b in (("MJ", b1), ("LG", b2)):
            M[f"vsbase{key}"] = b["mean"]; M[f"vsbaselo{key}"] = b["ci95"][0]; M[f"vsbasehi{key}"] = b["ci95"][1]
    for k, d in (("MJ", d1), ("LG", d2)):
        M[f"delta{k}"] = d["mean"]; M[f"deltalo{k}"] = d["ci95"][0]; M[f"deltahi{k}"] = d["ci95"][1]
        M[f"nimp{k}"] = d["n_improved"]

    # ---------- ablation + correlation ----------
    abl = ABL["ablation_cumulative_oracle"]
    # order: amount, +disorder, +composition, +role, +single-vector, +translated.
    # single-vector is the second-to-last step; translation is the last.
    M["singleaddMJ"] = abl["MultiJail"][-2] - abl["MultiJail"][-3]
    M["singleaddLG"] = abl["Lingua-SafetyBench"][-2] - abl["Lingua-SafetyBench"][-3]
    M["transaddMJ"] = abl["MultiJail"][-1] - abl["MultiJail"][-2]
    M["transaddLG"] = abl["Lingua-SafetyBench"][-1] - abl["Lingua-SafetyBench"][-2]
    M["disaddMJ"] = abl["MultiJail"][1] - abl["MultiJail"][0]
    M["disaddLG"] = abl["Lingua-SafetyBench"][1] - abl["Lingua-SafetyBench"][0]
    M["comboaddMJ"] = abl["MultiJail"][2] - abl["MultiJail"][1]
    M["comboaddLG"] = abl["Lingua-SafetyBench"][2] - abl["Lingua-SafetyBench"][1]
    M["triaddMJ"] = abl["MultiJail"][3] - abl["MultiJail"][2]
    M["triaddLG"] = abl["Lingua-SafetyBench"][3] - abl["Lingua-SafetyBench"][2]
    fams = ABL["ablation_cumulative_oracle"]["families_order"]
    body = ["\\begin{table}[t]", "\\centering", "\\small",
            "\\caption{Cumulative oracle as configuration families are added. Combination and role "
            "separation carry the arm space; neither collection is saturated by the amount axis alone.}",
            "\\label{tab:sel_ablation}",
            "\\begin{tabular}{lcc}", "\\toprule",
            "Configuration families available & MultiJail & Lingua-SafetyBench \\\\", "\\midrule"]
    prev = {ds: None for ds in DS}
    for i, f in enumerate(fams):
        cells = []
        for ds, key in (("MultiJail", "MultiJail"), ("Lingua-SafetyBench", "Lingua-SafetyBench")):
            v = ABL["ablation_cumulative_oracle"][key][i]
            cells.append(f"{v:.3f}" + (f" ($+${v-prev[ds]:.3f})" if prev[ds] is not None else ""))
            prev[ds] = v
        label = ("amount" if i == 0 else f"$+$ {ARM_TEX.get(f, f)}")
        body.append(f"{label} & " + " & ".join(cells) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    write("tab_sel_ablation.tex", "\n".join(body))

    # Correlation between the HARMLESS probe and verified ASR, with the harmful-run column beside it.
    # The two must not be conflated: the harmful column shares a factor with the target (verified ASR
    # is recon AND unsafe), which is most of why it looks stronger.
    from scipy.stats import spearmanr
    PROBE = {"MultiJail": "results/benign_arms_mj_20260907",
             "Lingua-SafetyBench": "results/benign_arms_20260907"}
    rho_b, rho_h = {}, {}
    for ds in DS:
        rho_b[ds], rho_h[ds] = {}, {}
        arms = list(A[ds][MODELS[0]])
        for t in MODELS:
            g = [A[ds][t][a] for a in arms]
            rho_h[ds][t] = float(spearmanr([RAW[ds][t][a]["recon"] for a in arms], g).statistic)
            f = Path(PROBE[ds]) / f"{t}.json"
            if f.exists():
                ar = json.loads(f.read_text())["arm_recon"]
                # the probe measured the old arm names; map new ones (see benign_prior aliases).
                al = {"tri_hi_en": "tri_hi_wl", "tri_triple_en": "tri_triple_en"}
                av = [1.0 if a == "m_translated" else ar[al.get(a, a)] for a in arms]
                rho_b[ds][t] = float(spearmanr(av, g).statistic)
    body = ["\\begin{table}[t]", "\\centering", "\\small",
            "\\caption{Spearman correlation with verified ASR across the 23 configurations. The "
            "harmless probe ranks configurations on the smaller targets and loses its grip on the "
            "larger, more strongly aligned member of each family, which bounds what a warm start can "
            "buy. The right-hand pair is the reconstruction column of the harmful runs, which is not "
            "an admissible prior and is shown only because it shares a factor with the target and so "
            "looks stronger.}",
            "\\label{tab:sel_corr}",
            "\\begin{tabular}{lcc|cc}", "\\toprule",
            "& \\multicolumn{2}{c|}{harmless probe} & \\multicolumn{2}{c}{harmful-run recon (reference)} \\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
            "Target & MultiJail & Lingua & MultiJail & Lingua \\\\", "\\midrule"]
    def cell(d, t):
        return f"{d[t]:+.2f}" if t in d else "--"
    for t in MODELS:
        vals = [rho_b["MultiJail"], rho_b["Lingua-SafetyBench"], rho_h["MultiJail"], rho_h["Lingua-SafetyBench"]]
        bold = t in rho_b["Lingua-SafetyBench"] and rho_b["Lingua-SafetyBench"][t] < 0.15
        row = " & ".join(cell(d, t) for d in vals)
        body.append((f"\\textbf{{{PRETTY[t]}}}" if bold else PRETTY[t]) + " & " + row + " \\\\")
    means = []
    for d in (rho_b["MultiJail"], rho_b["Lingua-SafetyBench"], rho_h["MultiJail"], rho_h["Lingua-SafetyBench"]):
        means.append(f"{mean(d.values()):+.2f}" if d else "--")
    body += ["\\midrule", "mean & " + " & ".join(means) + " \\\\",
             "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    write("tab_sel_corr.tex", "\n".join(body))
    for ds, key in (("MultiJail", "MJ"), ("Lingua-SafetyBench", "LG")):
        if rho_b[ds]:
            M[f"rhoben{key}"] = mean(rho_b[ds].values())
        M[f"rhoharm{key}"] = mean(rho_h[ds].values())

    # ---------- metric decomposition: our metric vs the numbers usually reported ----------
    body = ["\\begin{table}[t]", "\\centering", "\\small",
            "\\caption{How verified ASR relates to the quantities usually reported. \\emph{Raw} is the "
            "unsafe-answer rate alone, which most jailbreak papers call ASR; \\emph{recon} is the "
            "reconstruction rate; \\emph{verified} requires both in the same response. The gap is what "
            "a raw number over-counts, and it is largest for a fixed configuration.}",
            "\\label{tab:sel_metrics}",
            "\\begin{tabular}{lcccc}", "\\toprule",
            "& \\multicolumn{2}{c}{MultiJail} & \\multicolumn{2}{c}{Lingua-SafetyBench} \\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
            "Configuration & raw / recon & verified & raw / recon & verified \\\\", "\\midrule"]
    for lbl, pickf in (("selected per target (oracle)", lambda ds, t: max(names, key=lambda a: A[ds][t][a])),
                       ("best fixed configuration", lambda ds, t: B[ds]["fixed_arm"])):
        cells = []
        for ds in DS:
            picks = {t: pickf(ds, t) for t in MODELS}
            u = mean(RAW[ds][t][picks[t]]["unsafe"] for t in MODELS)
            r = mean(RAW[ds][t][picks[t]]["recon"] for t in MODELS)
            v = mean(A[ds][t][picks[t]] for t in MODELS)
            cells += [f"{u:.3f} / {r:.3f}", f"{v:.3f}"]
            M[("oracle" if "oracle" in lbl else "fixed") + "raw" + ("MJ" if ds == "MultiJail" else "LG")] = u
        body.append(f"{lbl} & " + " & ".join(cells) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    write("tab_sel_metrics.tex", "\n".join(body))

    # ---------- prose macros ----------
    DIG = str.maketrans({"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
                         "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"})
    lines = ["% auto-generated by scripts/make_selector_tables.py -- do not edit"]
    for k, v in sorted(M.items()):
        val = str(v) if isinstance(v, (PCT, str)) else (f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append(f"\\newcommand{{\\pj{k.replace('_', '').translate(DIG)}}}{{{val}}}")
    write("selector_numbers.tex", "\n".join(lines) + "\n")
    print("\nmacros:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sorted(M.items())})


if __name__ == "__main__":
    raise SystemExit(main())
