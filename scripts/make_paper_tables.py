#!/usr/bin/env python3
"""Emit ICLR LaTeX result tables from the aggregate result JSONs.

Reads results/*.json and writes paper/tab_*.tex. Re-run after paper_main finishes
to replace the pending cells in the main comparison table with CSRT / multi-target
numbers. Only aggregate numbers are read; no controlled text is touched.
"""
from __future__ import annotations
import json
from pathlib import Path

R = Path("results")
P = Path("paper")


def load(name):
    f = R / name
    return json.load(f.open()) if f.exists() else None


def fmt(x, d=3):
    return f"{x:.{d}f}" if isinstance(x, (int, float)) else r"\pending"


def curve_table():
    cm = load("qwen_interleaving_curve_matrix.json")
    s = load("qwen_interleaving_curve_summary.json")
    if not cm:
        return
    base = s["conditions"]["english_direct"]["gated_asr"]
    fin = s["conditions"].get("translated_direct_Finnish", {}).get("gated_asr")
    loads = cm["language_loads"]
    rows = []
    for n in loads:
        o = cm["matrix"]["ordered"].get(str(n), {})
        sh = cm["matrix"]["shuffled"].get(str(n), {})
        rows.append(
            f"{n} & {fmt(o.get('gated_asr'))} & {fmt(o.get('semantic_recon_rate'),2)} & "
            f"{fmt(sh.get('gated_asr'))} & {fmt(sh.get('semantic_recon_rate'),2)} \\\\")
    body = "\n".join(rows)
    tex = r"""\begin{table}[t]
\centering
\caption{Language-load sweep on Lingua-SafetyBench (Qwen2.5-7B, full 2{,}879).
Reconstruction-gated ASR and reconstruction rate for ordered vs.\ shuffled
interleaving. Reference gated ASR: english\_direct $=%s$, best translation
(Finnish) $=%s$. Ordered peaks at $n{=}4$; over-dispersion collapses
reconstruction.}
\label{tab:curve}
\begin{tabular}{r cc cc}
\toprule
& \multicolumn{2}{c}{Ordered} & \multicolumn{2}{c}{Shuffled} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
$n$ (languages) & gated ASR & recon & gated ASR & recon \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (fmt(base), fmt(fin), body)
    (P / "tab_curve.tex").write_text(tex)
    print("wrote tab_curve.tex")


def main_table():
    """Prefer the paper_main head-to-head (with CSRT) once available; else the
    current full-2879 vLLM run (no CSRT yet) with CSRT rows marked pending."""
    mc = load("paper_lingua_qwen_method_comparison_test.json") \
        or load("paper_lingua_qwen_method_comparison.json")
    order = [
        ("english\\_direct", "english_direct"),
        ("Translation (best) \\citep{yong2023lowresource}", "__best_trans__"),
        ("CSRT $k{=}2$ \\citep{yoo2024csrt}", "csrt_k2"),
        ("CSRT (all spans)", "csrt_all"),
        ("CSRT-MT $n{=}3$", "csrt_mt_n3"),
        ("No-game $n{=}4$ (ablation)", "nogame_ordered_n4"),
        ("\\textbf{Slot} $k{=}2$ (ours)", "slot_k2"),
        ("\\textbf{Interleave} $n{=}4$ ordered (ours)", "interleave_ordered_n4"),
        ("\\textbf{Interleave} $n{=}10$ ordered (ours)", "interleave_ordered_n10"),
    ]
    if mc:
        conds = mc["conditions"]
        best = mc["method_families"].get("best_translation")
        def cell(key):
            k = best if key == "__best_trans__" else key
            v = conds.get(k)
            if not v:
                return r"\pending & \pending & \pending & \pending"
            return (f"{fmt(v['gated_asr'])} & {fmt(v.get('gated_asr_mdjudge'))} & "
                    f"{fmt(v.get('mean_hr'),2)} & {fmt(v['semantic_recon_rate'],2)}")
        note = "held-out test split; CSRT matched to slot on identical spans"
    else:
        # fallback: current full-2879 vLLM gated summary (no CSRT / MD / HR yet)
        s = load("full_textdom_textmodel_vllm_summary.json")
        conds = s["conditions"] if s else {}
        fin = max((c for c in conds if c.startswith("translated_direct_")),
                  key=lambda c: conds[c]["gated_asr"], default=None)
        def cell(key):
            k = fin if key == "__best_trans__" else key
            v = conds.get(k)
            if not v:
                return r"\pending & \pending & \pending & \pending"
            return (f"{fmt(v['gated_asr'])} & \\pending & \\pending & "
                    f"{fmt(v['semantic_recon_rate'],2)}")
        note = ("current full-2879 run; CSRT/MD-Judge/HR columns pending the "
                "full-scale head-to-head")
    rows = "\n".join(f"{lab} & {cell(key)} \\\\" for lab, key in order)
    tex = r"""\begin{table}[t]
\centering
\caption{Attack-method comparison on Lingua-SafetyBench (Qwen2.5-7B).
Reconstruction-gated ASR under the primary judge (Qwen3Guard) and cross judge
(MD-Judge), mean Harmfulness Rating (HR, 0--5), and reconstruction rate. %s.}
\label{tab:main}
\begin{tabular}{l cccc}
\toprule
Method & gated ASR & gated (MD) & HR & recon \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (note, rows)
    (P / "tab_main.tex").write_text(tex)
    print("wrote tab_main.tex (source: %s)" % ("paper_main" if mc else "fallback full-2879"))


def attaq_table():
    s = load("attaq_full_summary.json")
    if not s:
        (P / "tab_attaq.tex").write_text("% attaq pending\n")
        return
    conds = s["conditions"]
    keys = [("english\\_direct", "english_direct"),
            ("Translation (Slovenian)", "translated_direct_Slovenian"),
            ("Translation (Czech)", "translated_direct_Czech"),
            ("\\textbf{Interleave} $n{=}4$ (ours)", "interleave_ordered_n4"),
            ("\\textbf{Interleave} $n{=}7$ (ours)", "interleave_ordered_n7")]
    rows = []
    for lab, k in keys:
        v = conds.get(k)
        if not v:
            continue
        p = v.get("paired_gated_vs_baseline") or {}
        d = p.get("difference")
        rows.append(f"{lab} & {fmt(v['raw_asr'])} & {fmt(v['semantic_recon_rate'],2)} & "
                    f"{fmt(v['gated_asr'])} & {fmt(d) if d is not None else '--'} \\\\")
    tex = r"""\begin{table}[t]
\centering
\caption{External generalization on AttaQ (Qwen2.5-7B, full 1{,}402). PolyJigsaw
lifts gated ASR over direct English; low-resource translation is also strong on
these blunt prompts (see Sec.~5.4).}
\label{tab:attaq}
\begin{tabular}{l cccc}
\toprule
Method & raw ASR & recon & gated ASR & $\Delta$ vs.\ english \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(rows)
    (P / "tab_attaq.tex").write_text(tex)
    print("wrote tab_attaq.tex")


if __name__ == "__main__":
    curve_table(); main_table(); attaq_table()
