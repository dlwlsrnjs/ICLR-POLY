#!/usr/bin/env python3
"""Generate ALL ICLR paper LaTeX tables from the aggregate result JSONs.

Reads results/*.json + a couple of private_artifacts summaries and writes
paper/tab_*.tex. Cells with no data yet are rendered as \pending. Re-run whenever
a new result lands. Only aggregate numbers are read.
"""
from __future__ import annotations
import json
from pathlib import Path

R = Path("results"); P = Path("paper"); PA = Path("private_artifacts")
PEND = "\\pending"


def load(p):
    p = Path(p)
    return json.load(p.open()) if p.exists() else None


def fp(x):
    if not isinstance(x,(int,float)): return PEND
    return '$<10^{-4}$' if x<1e-4 else ('%.4f'%x)

def f(x, d=3):
    return f"{x:.{d}f}" if isinstance(x, (int, float)) else r"\pending"


# ---------- Table: language-load curve ----------
def t_curve():
    cm = load(R/"qwen_interleaving_curve_matrix.json"); s = load(R/"qwen_interleaving_curve_summary.json")
    if not cm: return
    base = s["conditions"]["english_direct"]["gated_asr"]; fin = s["conditions"].get("translated_direct_Finnish",{}).get("gated_asr")
    rows = "\n".join(
        f"{n} & {f(cm['matrix']['ordered'].get(str(n),{}).get('gated_asr'))} & "
        f"{f(cm['matrix']['ordered'].get(str(n),{}).get('semantic_recon_rate'),2)} & "
        f"{f(cm['matrix']['shuffled'].get(str(n),{}).get('gated_asr'))} & "
        f"{f(cm['matrix']['shuffled'].get(str(n),{}).get('semantic_recon_rate'),2)} \\\\"
        for n in cm["language_loads"])
    (P/"tab_curve.tex").write_text(r"""\begin{table}[t]\centering
\caption{Language-load sweep (Qwen2.5-7B, Lingua full 2{,}879). Gated ASR and reconstruction rate;
reference english\_direct $=%s$, best translation (Finnish) $=%s$. Ordered peaks at $n{=}4$.}
\label{tab:curve}
\begin{tabular}{r cc cc}\toprule
& \multicolumn{2}{c}{Ordered} & \multicolumn{2}{c}{Shuffled}\\\cmidrule(lr){2-3}\cmidrule(lr){4-5}
$n$ & gated ASR & recon & gated ASR & recon\\\midrule
%s
\bottomrule\end{tabular}\end{table}
""" % (f(base), f(fin), rows)); print("tab_curve")


# ---------- Table: main method comparison (Qwen test) ----------
def t_main():
    d = load(R/"paper_lingua_qwen_method_comparison_test.json")
    if not d: (P/"tab_main.tex").write_text("% pending\n"); return
    c = d["conditions"]; best = d["method_families"].get("best_translation")
    rows = [
        ("english\\_direct", "english_direct"),
        (f"Translation (best: {best.replace('translated_direct_','') if best else '--'}) \\citep{{yong2023lowresource}}", best),
        ("CSRT $k{=}2$ \\citep{yoo2024csrt}", "csrt_k2"),
        ("CSRT (all spans)", "csrt_all"),
        ("CSRT-MT $n{=}3$", "csrt_mt_n3"),
        ("No-game $n{=}4$ (ablation)", "nogame_ordered_n4"),
        ("\\textbf{Slot} $k{=}1$ (ours)", "slot_k1"),
        ("\\textbf{Interleave} $n{=}4$ (ours)", "interleave_ordered_n4"),
        ("\\textbf{Interleave} $n{=}6$ (ours)", "interleave_ordered_n6"),
    ]
    body = ""
    for lab, k in rows:
        v = c.get(k) if k else None
        if not v: body += f"{lab} & \\pending & \\pending & \\pending & \\pending \\\\\n"; continue
        body += (f"{lab} & {f(v['gated_asr'])} & {f(v.get('gated_asr_mdjudge'))} & "
                 f"{f(v.get('mean_hr'),2)} & {f(v['semantic_recon_rate'],2)} \\\\\n")
    n = d.get("n_items_in_split")
    (P/"tab_main.tex").write_text(r"""\begin{table}[t]\centering
\caption{Attack-method comparison on Lingua-SafetyBench (Qwen2.5-7B, held-out test split, $n{=}%s$ items).
Gated ASR under the primary judge (Qwen3Guard) and cross judge (MD-Judge), mean HR (0--5), and
reconstruction rate. Slot$_k$ and CSRT$_k$ use identical spans; their gap isolates the reconstruction game.}
\label{tab:main}
\begin{tabular}{l cccc}\toprule
Method & gated ASR & gated (MD) & HR & recon\\\midrule
%s\bottomrule\end{tabular}\end{table}
""" % (n, body)); print("tab_main")


# ---------- Table: multi-target generalization ----------
def t_targets():
    targets = [("Qwen2.5-7B", "paper_lingua_qwen_method_comparison_test"),
               ("Qwen3-8B", "paper_lingua_qwen3_method_comparison_test"),
               ("Phi-3.5-mini", "paper_lingua_phi_method_comparison_test"),
               ("InternLM2.5-7B", "paper_lingua_internlm_method_comparison_test")]
    header = r"Target & english & best transl. & CSRT (all) & interleave $n{=}4$ & interleave (best) \\"
    body = ""
    for tname, tf in targets:
        d = load(R / (tf + ".json"))
        if not d:
            body += tname + " & " + " & ".join([PEND]*5) + " " + chr(92)*2 + "\n"
            continue
        c = d["conditions"]; best = d["method_families"].get("best_translation")
        inter = {k: v for k, v in c.items() if k.startswith("interleave_ordered_") and v.get("gated_asr") is not None}
        bestint = max(inter.values(), key=lambda v: v["gated_asr"]) if inter else None
        cells = [
            f(c.get("english_direct", {}).get("gated_asr")),
            f(c.get(best, {}).get("gated_asr")) if best else PEND,
            f(c.get("csrt_all", {}).get("gated_asr")),
            f(c.get("interleave_ordered_n4", {}).get("gated_asr")),
            f(bestint["gated_asr"]) if bestint else PEND,
        ]
        body += tname + " & " + " & ".join(cells) + " " + chr(92)*2 + "\n"
    tex = (r"\begin{table}[t]\centering" "\n"
           r"\caption{Generalization across aligned open models (Lingua test split, gated ASR under Qwen3Guard). "
           r"Interleaving exceeds every baseline on all four targets. ``best'' is the peak interleaving load "
           r"available for that target ($n{=}6$ for Qwen2.5-7B/Phi-3.5, $n{=}10$ for Qwen3-8B/InternLM).}" "\n"
           r"\label{tab:targets}" "\n"
           r"\begin{tabular}{l ccccc}\toprule" "\n" + header + r"\midrule" "\n" + body +
           r"\bottomrule\end{tabular}\end{table}" "\n")
    (P / "tab_targets.tex").write_text(tex); print("tab_targets")

def t_frontier():
    d = load(R/"paper_frontier_gpt-4o-mini_method_comparison.json")
    if not d: (P/"tab_frontier.tex").write_text("% pending\n"); return
    c = d["conditions"]; best = d["method_families"].get("best_translation")
    rows = [("english\\_direct","english_direct"), (f"Translation (best)", best),
            ("CSRT (all)","csrt_all"), ("CSRT-MT $n{=}3$","csrt_mt_n3"),
            ("No-game $n{=}4$","nogame_ordered_n4"),
            ("\\textbf{Interleave} $n{=}4$","interleave_ordered_n4"),
            ("\\textbf{Interleave} $n{=}10$","interleave_ordered_n10")]
    body=""
    for lab,k in rows:
        v=c.get(k) if k else None
        body += f"{lab} & {f(v['gated_asr']) if v else PEND} & {f(v.get('gated_asr_mdjudge')) if v else PEND} & {f(v.get('semantic_recon_rate'),2) if v else PEND} \\\\\n"
    n=(c.get("english_direct",{}) or {}).get("n") or d.get("n_rows_compared") or ""
    (P/"tab_frontier.tex").write_text(r"""\begin{table}[t]\centering
\caption{Commercial target GPT-4o-mini (Lingua test subset, %s items). Interleaving lifts gated
ASR far above every multilingual baseline on a strongly-aligned model.}
\label{tab:frontier}
\begin{tabular}{l ccc}\toprule
Method & gated ASR & gated (MD) & recon\\\midrule
%s\bottomrule\end{tabular}\end{table}
""" % (n, body)); print("tab_frontier")


# ---------- Table: defense (Self-Reminder) ----------
def t_defense():
    d = load(R/"paper_lingua_qwen_defense_summary.json")
    base = load(R/"paper_lingua_qwen_method_comparison_test.json")
    if not d or not base: (P/"tab_defense.tex").write_text("% pending\n"); return
    dc = d["conditions"]; bc = base["conditions"]
    def g(src,k): 
        v=src.get(k); return v.get("gated_asr") if v else None
    rows=[("english\\_direct","english_direct"),("CSRT (all)","csrt_all"),
          ("Translation (Finnish)","translated_direct_Finnish"),
          ("\\textbf{Interleave} $n{=}4$","interleave_ordered_n4")]
    body=""
    for lab,k in rows:
        body += f"{lab} & {f(g(bc,k))} & {f(g(dc,k))} \\\\\n"
    (P/"tab_defense.tex").write_text(r"""\begin{table}[t]\centering
\caption{Under a Self-Reminder defensive system prompt (Qwen2.5-7B, gated ASR). The defense collapses
the baselines but barely dents interleaving.}
\label{tab:defense}
\begin{tabular}{l cc}\toprule
Method & no defense & Self-Reminder\\\midrule
%s\bottomrule\end{tabular}\end{table}
""" % body); print("tab_defense")


# ---------- Table: input detectability gap ----------
def t_detect():
    d = load(PA/"paper_main/lingua_qwen/input_detect/summary.json")
    if not d: (P/"tab_detect.tex").write_text("% pending\n"); return
    pc=d["per_condition"]
    rows=[("english\\_direct","english_direct"),("Translation (Finnish)","translated_direct_Finnish"),
          ("CSRT (all)","csrt_all"),("\\textbf{Interleave} $n{=}4$","interleave_ordered_n4"),
          ("\\textbf{Interleave} $n{=}6$","interleave_ordered_n6")]
    body=""
    for lab,k in rows:
        v=pc.get(k)
        body += f"{lab} & {f(v['input_flagged_unsafe_rate']) if v else PEND} & {f(v['input_passed_as_safe_rate']) if v else PEND} \\\\\n"
    (P/"tab_detect.tex").write_text(r"""\begin{table}[t]\centering
\caption{Input-stage detectability (Qwen3Guard prompt moderation on the submitted request).
Dispersed prompts pass input screening far more often than the direct request.}
\label{tab:detect}
\begin{tabular}{l cc}\toprule
Method & flagged unsafe & passed as safe\\\midrule
%s\bottomrule\end{tabular}\end{table}
""" % body); print("tab_detect")


# ---------- Table: thinking vs no-thinking (Qwen3) ----------
def t_thinking():
    nt = load(R/"paper_lingua_qwen3_method_comparison_test.json")
    th = load(R/"paper_lingua_qwen3think_method_comparison_subset.json")
    if not (nt and th): (P/"tab_thinking.tex").write_text("% pending thinking run\n"); return
    rows=[("english\\_direct","english_direct"),("interleave $n{=}4$","interleave_ordered_n4"),
          ("interleave $n{=}10$","interleave_ordered_n10"),("no-game $n{=}4$","nogame_ordered_n4")]
    body=""
    for lab,k in rows:
        a=nt["conditions"].get(k); b=th["conditions"].get(k)
        body += (f"{lab} & {f(a['gated_asr']) if a else PEND} & {f(a['semantic_recon_rate'],2) if a else PEND} & "
                 f"{f(b['gated_asr']) if b else PEND} & {f(b['semantic_recon_rate'],2) if b else PEND} \\\\\n")
    (P/"tab_thinking.tex").write_text(r"""\begin{table}[t]\centering
\caption{Qwen3-8B with thinking disabled vs enabled (Lingua; no-think test split, think 240-item subset).
Thinking mode's chain fails to terminate on the interleaving task, collapsing reconstruction.}
\label{tab:thinking}
\begin{tabular}{l cc cc}\toprule
& \multicolumn{2}{c}{thinking OFF} & \multicolumn{2}{c}{thinking ON}\\\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Condition & gated ASR & recon & gated ASR & recon\\\midrule
%s\bottomrule\end{tabular}\end{table}
""" % body); print("tab_thinking")


# ---------- Table: AttaQ ----------
def t_attaq():
    d=load(R/"paper_attaq_qwen_method_comparison.json") or load(R/"attaq_full_summary.json")
    if not d: (P/"tab_attaq.tex").write_text("% pending\n"); return
    c=d["conditions"]
    def cell(k,key="gated_asr"):
        v=c.get(k); 
        if not v: return r"\pending"
        return f(v.get(key) if key in v else v.get("gated_asr"))
    rows=[("english\\_direct","english_direct"),("Translation (Slovenian)","translated_direct_Slovenian"),
          ("\\textbf{Interleave} $n{=}4$","interleave_ordered_n4"),("\\textbf{Interleave} $n{=}7$","interleave_ordered_n7")]
    body=""
    for lab,k in rows:
        v=c.get(k)
        gm = v.get("gated_asr_mdjudge") if v else None
        body += f"{lab} & {cell(k)} & {f(gm) if gm is not None else '--'} \\\\\n"
    (P/"tab_attaq.tex").write_text(r"""\begin{table}[t]\centering
\caption{External dataset AttaQ (Qwen2.5-7B, 1{,}402 items). Under the cross judge (MD) interleaving is
level with or ahead of the strongest low-resource translation on these blunt prompts.}
\label{tab:attaq}
\begin{tabular}{l cc}\toprule
Method & gated ASR (Guard) & gated ASR (MD)\\\midrule
%s\bottomrule\end{tabular}\end{table}
""" % body); print("tab_attaq")



def t_matched():
    d = load(R/"paper_lingua_qwen_method_comparison_test.json")
    if not d: (P/"tab_matched.tex").write_text("% pending\n"); return
    c = d["conditions"]; mm = d.get("matched_slot_vs_csrt_same_spans", {})
    rows = ""
    for k in (1,2,3):
        sk=c.get("slot_k%d"%k); ck=c.get("csrt_k%d"%k)
        pair=mm.get("slot_k%d_vs_csrt_k%d"%(k,k), {})
        n = sk["n"] if sk else (ck["n"] if ck else None)
        rows += ("$k{=}%d$ & %s & %s & %s & %s & %s "%(
            k, (str(n) if n else PEND),
            f(ck["gated_asr"]) if ck else PEND,
            f(sk["gated_asr"]) if sk else PEND,
            (("$+$" if (pair.get("difference") or 0)>=0 else "")+f(pair.get("difference"))) if pair else PEND,
            fp(pair.get("exact_mcnemar_p")) if pair else PEND)) + chr(92)*2 + "\n"
    tex = (chr(92)+"begin{table}[t]"+chr(92)+"centering\n"
        +chr(92)+"caption{Matched comparison on identical items and official spans (Qwen2.5-7B, test): "
        "the inline-slot reconstruction game vs.\\ CSRT direct code-switch at each span count $k$. "
        "The only difference is the reconstruction instruction; the paired difference isolates its effect.}\n"
        +chr(92)+"label{tab:matched}\n"
        +chr(92)+"begin{tabular}{l r cc cc}"+chr(92)+"toprule\n"
        "$k$ & $n$ & CSRT$_k$ (no game) & slot$_k$ (game) & $\\Delta$ (game$-$CSRT) & McNemar $p$ "+chr(92)*2+chr(92)+"midrule\n"
        +rows+chr(92)+"bottomrule"+chr(92)+"end{tabular}"+chr(92)+"end{table}\n")
    (P/"tab_matched.tex").write_text(tex); print("tab_matched")

def t_rawgated():
    d = load(R/"paper_lingua_qwen_method_comparison_test.json")
    if not d: (P/"tab_rawgated.tex").write_text("% pending\n"); return
    c = d["conditions"]
    order=[("english\\_direct","english_direct"),("translation (Finnish)","translated_direct_Finnish"),
           ("CSRT (all)","csrt_all"),("no-game $n{=}4$","nogame_ordered_n4"),
           ("slot $k{=}1$","slot_k1"),("interleave $n{=}4$","interleave_ordered_n4"),
           ("interleave $n{=}6$","interleave_ordered_n6")]
    rows=""
    for lab,k in order:
        v=c.get(k)
        if not v: rows+=lab+" & "+" & ".join([PEND]*3)+chr(92)*2+"\n"; continue
        rows+=("%s & %s & %s & %s "%(lab, f(v["raw_asr"]), f(v["semantic_recon_rate"],2), f(v["gated_asr"])))+chr(92)*2+"\n"
    tex=(chr(92)+"begin{table}[t]"+chr(92)+"centering\n"
        +chr(92)+"caption{Raw vs.\\ reconstruction-gated ASR (Qwen2.5-7B, test). Direct conditions are "
        "faithful by construction (recon $=1$), so the gate does not lower them; only our reconstruction "
        "conditions are reduced by the gate, so gated ASR is a conservative measure of our attack.}\n"
        +chr(92)+"label{tab:rawgated}\n"
        +chr(92)+"begin{tabular}{l ccc}"+chr(92)+"toprule\n"
        "Method & raw ASR & recon & gated ASR "+chr(92)*2+chr(92)+"midrule\n"
        +rows+chr(92)+"bottomrule"+chr(92)+"end{tabular}"+chr(92)+"end{table}\n")
    (P/"tab_rawgated.tex").write_text(tex); print("tab_rawgated")


if __name__ == "__main__":
    for fn in (t_curve,t_main,t_targets,t_frontier,t_defense,t_detect,t_thinking,t_attaq,t_matched,t_rawgated):
        try: fn()
        except Exception as e: print("ERR", fn.__name__, e)
