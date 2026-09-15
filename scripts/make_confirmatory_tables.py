#!/usr/bin/env python3
"""Generate the LaTeX tables and macros for the 17-model confirmatory selector experiments.

Reads the distilled summaries and family-clustered statistics in
experiments_suite/exp06_confirmatory_selector/results (see scripts/distill_confirmatory_summaries.py)
and writes, so the manuscript can never drift from the numbers on disk:

    paper/confirmatory_numbers.tex     macros for every number quoted in the prose
    paper/tab_confirm_strict.tex       strict item-held-out additive BO
    paper/tab_confirm_mfid.tex         equal-item-cost multi-fidelity allocation (negative result)
    paper/tab_confirm_selector.tex     offline/online learned selector
    paper/tab_confirm_endtoend.tex     aggregate replay against search-baseline adaptations

Panel: 17 models, 7 families, MultiJail + Lingua-SafetyBench, 160 arms, Qwen3Guard unsafe ASR.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "experiments_suite/exp06_confirmatory_selector/results"
P = ROOT / "paper"

BUDGET_WORD = {0: "Zero", 2: "Two", 4: "Four", 8: "Eight", 16: "Sixteen"}
macros: dict[str, str] = {}


def put(name: str, value, digits: int = 3) -> str:
    text = f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    macros[name] = text
    return text


def signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def load(name: str) -> dict:
    return json.loads((R / name).read_text(encoding="utf-8"))


def by_method(summary, key="family_macro_test_asr", budget_key="budget"):
    table: dict[tuple[str, int], float] = {}
    for row in summary:
        method = row.get("method", row.get("kernel"))
        budget = int(row.get(budget_key, row.get("equivalent_full_arm_budget", -1)))
        table[(method, budget)] = float(row[key])
    return table


def fmt(value):
    return "--" if value is None else f"{value:.4f}"


def row(label: str, values, bold=()):
    cells = []
    for budget, value in values:
        text = fmt(value)
        if budget in bold:
            text = f"\\textbf{{{text}}}"
        cells.append(text)
    return f"{label} & " + " & ".join(cells) + " \\\\"


# ---------------------------------------------------------------- strict item-held-out
strict = load("itemheldout_advanced_bo_posterior_summary.json")
S = by_method(strict["summary"])
# The posterior run re-scored every deterministic method but replayed random search with fewer
# repetitions (16) than the earlier run (64 for the first five splits, 32 afterwards).  Both agree
# exactly on every deterministic method, so the random baseline is taken from the better-estimated
# run to avoid flattering the comparison.
random_source = load("itemheldout_advanced_bo_summary.json")
for (method, budget), value in by_method(random_source["summary"]).items():
    if method == "random_probe_best_observed":
        S[(method, budget)] = value
RANDOM_REPS = random_source["protocol"].get("random_repetitions")
BUD = [2, 4, 8, 16]
STRICT_ROWS = [
    ("random_probe_best_observed", "Random, best observed", "Rand"),
    ("best_fixed_other_families", "Fixed best from other families", "Fixed"),
    ("structured_gp_crossfamily_incumbent", "Structured GP, best observed", "GPobs"),
    ("structured_gp_crossfamily_posterior", "Structured GP, posterior", "GPpost"),
    ("advanced_nested_meanstd", "Advanced mean-std, best observed", "MSobs"),
    ("advanced_nested_meanstd_posterior", "Advanced mean-std, posterior", "MSpost"),
    ("advanced_nested_cvar25", "Advanced CVaR25, best observed", "CVobs"),
    ("advanced_nested_cvar25_posterior", "Advanced CVaR25, posterior", "CVpost"),
    ("advanced_nested_minimax", "Advanced minimax", "MM"),
]
best_at = {b: max(S.get((m, b), 0.0) for m, _, _ in STRICT_ROWS) for b in BUD}
lines = []
for method, label, code in STRICT_ROWS:
    values = [(b, S.get((method, b))) for b in BUD]
    for b, value in values:
        if value is not None:
            put(f"\\pjcfStrict{code}B{BUDGET_WORD[b]}", value, 4)
    lines.append(row(label, values, bold={b for b, v in values if v is not None and v == best_at[b]}))
oracle = S.get(("oracle_test_upper_bound", -1))
if oracle is None:
    oracle = max(v for (m, _), v in S.items() if m == "oracle_test_upper_bound")
put("\\pjcfStrictOracle", oracle, 4)
put("\\pjcfStrictSeeds", len(strict["protocol"]["split_seeds"]))
print(f"random baseline repetitions: {RANDOM_REPS}")
put("\\pjcfModels", len(strict["protocol"]["models"]))
put("\\pjcfFamilies", len(strict["protocol"]["families"]))
put("\\pjcfArms", strict["protocol"]["arms"])
lines.append("\\midrule")
lines.append(f"Test oracle upper bound & \\multicolumn{{3}}{{c}}{{--}} & {oracle:.4f} \\\\")

(P / "tab_confirm_strict.tex").write_text(f"""\\begin{{table}}[t]
\\centering
\\small
\\caption{{\\textbf{{Strict item-held-out transfer on the 17-model panel.}} Family-macro Qwen3Guard
unsafe ASR on target \\emph{{test}} items, over {len(strict['protocol']['split_seeds'])} independent item splits and seven
leave-one-family-out folds. Within each benchmark half the items train the source prior, a quarter
calibrate the target, and a quarter are held out for scoring; target test outcomes never reach the
prior, kernel, tuning, arm queries or recommendation rule. Columns are the number of target arm
evaluations. The oracle is the best test arm in hindsight, an upper bound rather than an achievable
target. Deterministic methods are scored by the posterior-recommendation run; random search is
reported from the companion run that replays it with more repetitions, which the two runs otherwise
match exactly.}}
\\label{{tab:confirm_strict}}
\\begin{{tabular}}{{lcccc}}
\\toprule
Method & $B{{=}}2$ & $B{{=}}4$ & $B{{=}}8$ & $B{{=}}16$ \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""", encoding="utf-8")

# ---------------------------------------------------------------- multi-fidelity
mfid = load("multifidelity_itemcost_bo_summary.json")
M = by_method(mfid["summary"], budget_key="equivalent_full_arm_budget")
MFID_ROWS = [
    ("full_fidelity_gp", "Full-fidelity structured GP", "Full"),
    ("multifidelity_graph_kernel_ucb", "Multi-fidelity graph kernel-UCB", "MF"),
    ("prior_seeded_successive_halving_top32", "Successive halving, top-32", "SH"),
    ("random_full_fidelity", "Random full-fidelity", "Rand"),
]
best_mf = {b: max(M.get((m, b), 0.0) for m, _, _ in MFID_ROWS) for b in BUD}
mlines = []
for method, label, code in MFID_ROWS:
    values = [(b, M.get((method, b))) for b in BUD]
    for b, value in values:
        if value is not None:
            put(f"\\pjcfMfid{code}B{BUDGET_WORD[b]}", value, 4)
    mlines.append(row(label, values, bold={b for b, v in values if v is not None and v == best_mf[b]}))

(P / "tab_confirm_mfid.tex").write_text(f"""\\begin{{table}}[t]
\\centering
\\small
\\caption{{\\textbf{{Equal-item-cost multi-fidelity allocation: a negative result.}} One cost unit is one
target response judged on one calibration item, so partial arm measurements and full arm evaluations
are charged the same way; columns give the cost in full-arm-evaluation equivalents. Family-macro
unsafe ASR on held-out test items. At the 10--16 item calibration sizes available here, spending the
budget on fewer, fully measured arms beats measuring more arms shallowly.}}
\\label{{tab:confirm_mfid}}
\\begin{{tabular}}{{lcccc}}
\\toprule
Method & $B{{=}}2$ & $B{{=}}4$ & $B{{=}}8$ & $B{{=}}16$ \\\\
\\midrule
{chr(10).join(mlines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""", encoding="utf-8")

# ---------------------------------------------------------------- learned selector
sel = load("selector_gpu_familybalanced_3splits_robustness_20260914.json")
SEL = {(r["method"], int(r["budget"])): float(r["equal_split_family_macro_test_asr"])
       for r in sel["overall"]}
SEL_BUD = [0, 2, 4, 8, 16]
SEL_ROWS = [
    ("best_fixed_other_families", "Family-balanced fixed prior", "Fixed"),
    ("random_probe_best_observed", "Random, best observed", "Rand"),
    ("supervised_context", "Supervised context selector", "Sup"),
    ("ppo_no_context", "PPO without context", "NoCtx"),
    ("ppo_no_feedback", "PPO without feedback", "NoFb"),
    ("ppo_context", "PPO context and feedback", "PPO"),
    ("ppo_query_best_observed", "PPO queries, best-observed incumbent", "PPOobs"),
]
best_sel = {b: max(SEL.get((m, b), 0.0) for m, _, _ in SEL_ROWS) for b in SEL_BUD}
slines = []
for method, label, code in SEL_ROWS:
    values = [(b, SEL.get((method, b))) for b in SEL_BUD]
    for b, value in values:
        if value is not None:
            put(f"\\pjcfSel{code}B{BUDGET_WORD[b]}", value, 4)
    slines.append(row(label, values, bold={b for b, v in values if v is not None and v == best_sel[b]}))
put("\\pjcfSelSplits", len(sel["protocol"]["item_split_seeds"]))

(P / "tab_confirm_selector.tex").write_text(f"""\\begin{{table}}[t]
\\centering
\\small
\\caption{{\\textbf{{Learned contextual selector, family-held-out.}} A policy over arms is initialised by
supervised ranking and trained with PPO on source families only; its context is built from harmless
FLORES reconstruction and FalseReject over-refusal probes, the queried-arm mask, observed calibration
outcomes and the remaining budget. Entries average repetitions, then models, datasets and training
seeds, then the seven held-out families, giving the {len(sel['protocol']['item_split_seeds'])} item splits equal weight. Learning helps at the
cold start and through its ablations; it does not dominate random search or the structured GP once
enough target queries are available.}}
\\label{{tab:confirm_selector}}
\\begin{{tabular}}{{lccccc}}
\\toprule
Method & $B{{=}}0$ & $B{{=}}2$ & $B{{=}}4$ & $B{{=}}8$ & $B{{=}}16$ \\\\
\\midrule
{chr(10).join(slines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""", encoding="utf-8")

# ---------------------------------------------------------------- end-to-end replay
e2e = load("pilot_end_to_end_search_baselines_l40s17_b16_20260914.json")
E = {r["method"]: r for r in e2e["summary"]}
E2E_ROWS = [
    ("manual_library_lofo_ranker", "Handcrafted attack library, rank selector", "Manual"),
    ("uniform_random_C", "Uniform random over the arm space", "Rand"),
    ("evolutionary_fuzzer_C", "Evolutionary fuzzer (GPTFuzzer-style)", "Fuzz"),
    ("tree_pruning_C", "Tree pruning (TAP-style)", "Tree"),
    ("map_elites_C", "Quality-diversity archive (MAP-Elites-style)", "QD"),
    ("flat_gp_ucb_C", "Additive GP-UCB, no prior", "Flat"),
    ("poly_bo_C", "POLY warm start and additive GP-UCB", "Poly"),
    ("poly_warm_tree_refinement_C", "POLY warm start and tree refinement", "PolyTree"),
]
cols = ["asr_at_4", "asr_at_8", "asr_at_budget", "auc"]
# LaTeX control sequences are letters only, so budgets become words in macro names.
COL_CODE = {"asr_at_4": "AtFour", "asr_at_8": "AtEight", "asr_at_budget": "AtBudget", "auc": "Auc"}
best_e2e = {c: max(float(E[m][c]) for m, _, _ in E2E_ROWS if c in E.get(m, {})) for c in cols}
elines = []
for method, label, code in E2E_ROWS:
    record = E.get(method, {})
    cells = []
    for column in cols:
        value = record.get(column)
        if value is None:
            cells.append("--")
            continue
        put(f"\\pjcfEnd{code}{COL_CODE[column]}", float(value), 4)
        text = f"{float(value):.4f}"
        cells.append(f"\\textbf{{{text}}}" if float(value) == best_e2e[column] else text)
    elines.append(f"{label} & " + " & ".join(cells) + " \\\\")

(P / "tab_confirm_endtoend.tex").write_text(f"""\\begin{{table}}[t]
\\centering
\\small
\\caption{{\\textbf{{Aggregate replay against search-strategy adaptations.}} Family-macro unsafe ASR over
the full benchmark, best observed arm within budget. The three search adaptations are discrete
re-implementations over the same arm space, not native reproductions of the published systems, and
attacker-side generation cost is not represented; the handcrafted library searches its own four-arm
pool. This is an upper-resource diagnostic and is reported separately from the strict item-held-out
result in Table~\\ref{{tab:confirm_strict}}.}}
\\label{{tab:confirm_endtoend}}
\\begin{{tabular}}{{lcccc}}
\\toprule
Method & ASR@4 & ASR@8 & ASR@16 & curve AUC \\\\
\\midrule
{chr(10).join(elines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""", encoding="utf-8")

# ---------------------------------------------------------------- family-clustered statistics
stats = load("confirmatory_stats_final_20260914.json")
WANTED = {
    ("strict_itemheldout", "advanced_nested_meanstd_posterior", "structured_gp_crossfamily_posterior", 2): "StrictMSpostBTwo",
    ("strict_itemheldout", "advanced_nested_cvar25_posterior", "structured_gp_crossfamily_posterior", 16): "StrictCVpostBSixteen",
    ("equal_item_cost_multifidelity", "multifidelity_graph_kernel_ucb", "full_fidelity_gp", 8): "MfidBEight",
    ("equal_item_cost_multifidelity", "multifidelity_graph_kernel_ucb", "full_fidelity_gp", 16): "MfidBSixteen",
    ("gpu_familyloo_selector", "ppo_context", "random_probe_best_observed", 2): "SelVsRandBTwo",
    ("gpu_familyloo_selector", "ppo_context", "ppo_no_context", 2): "SelCtxBTwo",
    ("gpu_familyloo_selector", "ppo_context", "ppo_no_context", 8): "SelCtxBEight",
    ("gpu_familyloo_selector", "ppo_context", "ppo_no_context", 16): "SelCtxBSixteen",
    ("gpu_familyloo_selector", "ppo_context", "ppo_no_feedback", 8): "SelFbBEight",
    ("gpu_familyloo_selector", "ppo_context", "ppo_no_feedback", 16): "SelFbBSixteen",
}
found = 0
for record in stats["comparisons"]:
    key = (record["experiment"], record["treatment"], record["control"], int(record["budget"]))
    code = WANTED.get(key)
    if code is None:
        continue
    found += 1
    low, high = record["family_cluster_bootstrap_95ci"]
    macros[f"\\pjcfD{code}"] = signed(record["paired_difference"])
    macros[f"\\pjcfD{code}lo"] = signed(low)
    macros[f"\\pjcfD{code}hi"] = signed(high)
    macros[f"\\pjcfD{code}p"] = f"{record['exact_family_signflip_p_two_sided']:.4f}"
put("\\pjcfStatClusters", len(stats["comparisons"][0]["families"]))
put("\\pjcfStatBoot", "20{,}000")

comparison = load("advanced_additive_bo_comparison_20260914.json")
put("\\pjcfAggOracle", float(comparison["oracle_family_macro"]), 4)
put("\\pjcfAggPrimaryAtSixteen", float(comparison["primary"]["asr_at_16"]), 4)
put("\\pjcfAggPreviousAtSixteen", float(comparison["previous_best"]["asr_at_16"]), 4)
put("\\pjcfAggGapClosed", 100 * float(comparison["primary"]["oracle_gap_closed_from_shared_first_arm"]), 1)

lines = ["% auto-generated by scripts/make_confirmatory_tables.py -- do not edit"]
lines += [f"\\newcommand{{{name}}}{{{value}}}" for name, value in sorted(macros.items())]
(P / "confirmatory_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"macros: {len(macros)}  statistics matched: {found}/{len(WANTED)}")
for name in ("tab_confirm_strict.tex", "tab_confirm_mfid.tex", "tab_confirm_selector.tex",
             "tab_confirm_endtoend.tex", "confirmatory_numbers.tex"):
    print(f"  wrote paper/{name} ({(P / name).stat().st_size} bytes)")
