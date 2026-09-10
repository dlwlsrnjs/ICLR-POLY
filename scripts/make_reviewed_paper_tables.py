#!/usr/bin/env python3
"""Regenerate manuscript-review tables from existing aggregate evidence only.

No target generation, judge calls, model training, or configuration search occurs.
Run from the repository root. Also called after the legacy table generator so that
its old captions cannot silently restore withdrawn claims.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
RESULTS = ROOT / 'results'
SOURCES = {}


def load(name):
    path = RESULTS / name
    data = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def fmt(value, digits=3):
    return f'{value:.{digits}f}' if isinstance(value, (int, float)) else r'\pending'


def table(filename, caption, label, spec, header, rows):
    body = '\n'.join(' & '.join(map(str, row)) + r' \\' for row in rows)
    text = (r'\begin{table}[t]\centering' + '\n' + r'\small' + '\n'
            + r'\setlength{\tabcolsep}{3.5pt}' + '\n'
            + '\\caption{' + caption + '}\n\\label{' + label + '}\n'
            + '\\begin{tabular}{' + spec + '}\\toprule\n'
            + header + r' \\\midrule' + '\n' + body
            + '\n' + r'\bottomrule\end{tabular}\end{table}' + '\n')
    (PAPER / filename).write_text(text)


def caption(filename, replacement):
    path = PAPER / filename
    text = path.read_text()
    start = text.index(r'\caption{') + len(r'\caption{')
    depth = 1
    end = start
    while depth:
        if text[end] == '{' and text[end - 1] != '\\':
            depth += 1
        elif text[end] == '}' and text[end - 1] != '\\':
            depth -= 1
        end += 1
    path.write_text(text[:start] + replacement + text[end - 1:])


def generate_reviewed_tables():
    main = load('paper_lingua_qwen_method_comparison_test.json')
    dev = load('paper_lingua_qwen_method_comparison_dev.json')
    # The named main translation is fixed from the development aggregate.
    translation = dev['method_families']['best_translation']
    assert translation == 'translated_direct_Finnish'
    rows = []
    for name, key in [('English direct', 'english_direct'), ('Finnish (dev-selected)', translation),
                      (r'CSRT $k=2$', 'csrt_k2'), ('CSRT all spans', 'csrt_all'),
                      (r'CSRT-MT $n=3$', 'csrt_mt_n3'), (r'No-game $n=4$', 'nogame_ordered_n4'),
                      (r'Slot $k=1$', 'slot_k1'), (r'Interleave $n=4$', 'interleave_ordered_n4'),
                      (r'Interleave $n=6$', 'interleave_ordered_n6')]:
        value = main['conditions'][key]
        lo, hi = value['gated_asr_wilson95']
        interval = fmt(value['gated_asr']) + r' {\scriptsize [' + f'{lo:.2f}, {hi:.2f}' + ']}'
        rows.append([name, value['n'], interval, fmt(value.get('gated_asr_mdjudge')),
                     fmt(value.get('mean_hr'), 2), fmt(value['semantic_recon_rate'], 2)])
    table('tab_main.tex',
          r'Qwen2.5-7B on the study test split. $N$ is the number of valid primary-judge rows for each condition, not a common denominator across all methods. The primary event includes Unsafe and Controversial labels; gated ASR includes its stored Wilson 95\% interval. MD is the cross-judge estimate, HR is mean harmfulness, and $R$ is reconstruction. Paired differences in the text use shared item IDs. Direct-condition $R=1$ is a scoring convention.',
          'tab:main', 'l r c c c c', r'Condition & $N$ & gated ASR [95\% CI] & MD & HR & $R$', rows)

    rows = []
    for name, source in [('Qwen2.5-7B', 'qwen'), ('Qwen3-8B', 'qwen3'),
                         ('Phi-3.5-mini', 'phi'), ('InternLM2.5-7B', 'internlm')]:
        data = load(f'paper_lingua_{source}_method_comparison_test.json')
        conditions = data['conditions']
        trans = conditions[data['method_families']['best_translation']]
        # Preserve historical test maxima, explicitly labeled as descriptive.
        selected = max((v for k, v in conditions.items() if k.startswith('interleave_ordered_')),
                       key=lambda v: v['gated_asr'])
        rows.append([name, fmt(conditions['english_direct']['gated_asr']),
                     fmt(trans['gated_asr']), fmt(trans.get('gated_asr_mdjudge')),
                     fmt(selected['gated_asr']), fmt(selected.get('gated_asr_mdjudge'))])
    table('tab_targets.tex',
          r'Cross-target descriptive comparison. Translation and ordered-interleaving columns select the highest primary-judge rate within each evaluated test set, then show that same condition under MD-Judge. These are test maxima, not validated performance of a development-selected or adaptive policy. Judge differences do not establish accuracy; secondary denominators are not retained in every source aggregate.',
          'tab:targets', 'l c cc cc',
          r'Target & English & Trans. Guard & Trans. MD & Interl. Guard & Interl. MD', rows)

    data = load('paper_regate_strict.json')
    rows = []
    for name, key in [('English direct', 'english_direct'), ('Finnish translation', 'translated_direct_Finnish'),
                      ('CSRT all', 'csrt_all'), (r'No-game ordered $n=4$', 'nogame_ordered_n4'),
                      (r'Slot $k=1$', 'slot_k1'), (r'Interleave ordered $n=4$', 'interleave_ordered_n4'),
                      (r'Interleave ordered $n=6$', 'interleave_ordered_n6'),
                      (r'Interleave shuffled $n=4$', 'interleave_shuffled_n4'),
                      (r'Interleave shuffled $n=6$', 'interleave_shuffled_n6')]:
        rows.append([name, fmt(data['primary_gate_qwen25'][key]), fmt(data['strict_gate_gpt4omini'][key])])
    table('tab_gateval.tex',
          r'Reconstruction-judge sensitivity on the full-set analysis, distinct from the main held-out test table. Direct conditions retain the conventional unit gate. GPT-4o-mini labels 24,836 rows; a separate GPT-4o sample has 1,916 valid pairs out of 2,000 planned. Larger changes for shuffled conditions preclude a claim that every ranking is preserved. This is model-judge agreement, not human validation.',
          'tab:gateval', 'l cc', 'Condition & Qwen gate & GPT-4o-mini gate', rows)

    data = load('paper_english_game_control.json')
    rows = []
    for name, key in [('English direct', 'english_direct'), ('English-only game', 'english_game'),
                      (r'Interleave ordered $n=4$', 'interleave_n4')]:
        v = data[key]
        rows.append([name, fmt(v['raw']), fmt(v.get('recon', 1)), fmt(v['gated']), fmt(v['raw'] - v['gated']), fmt(v['comply'])])
    table('tab_control.tex',
          r'English-only framing control on Qwen2.5-7B. Here $U$ denotes the primary Unsafe-or-Controversial event. $U-RU$ is the share receiving that label without a passing gate, computed from rounded aggregate rates. It includes missing or malformed reconstructions; it is not a human usefulness judgment. Direct English uses $R=1$ by convention. The conditions change the complete task framing and structure, so this comparison does not identify an internal safety mechanism. Source aggregate does not retain per-condition $N$.',
          'tab:control', 'l ccccc', r'Condition & raw $U$ & $R$ & joint $RU$ & $U-RU$ & $P(U\mid R)$', rows)

    five = load('paper_panel_theory_aligned5.json')
    six = load('paper_panel_theory_all6.json')
    rows = [['Five-model subset', five['n_models'], fmt(five['b_of_C_corr']), fmt(five['k'])],
            ['Six models including Mistral', six['n_models'], fmt(six['b_of_C_corr']), fmt(six['k'])]]
    table('tab_panel.tex',
          r'Sensitivity of the historical exploratory panel fit to model inclusion. The fitted reconstruction slope $k$ reaches its 0.05 lower bound in both cases. Correlations are descriptive and do not validate a predictive law. Historical BO query counts are omitted because their counting and observation-reuse implementation is invalid.',
          'tab:panel', 'l r cc', r'Panel & Models & capability--slope correlation & fitted $k$', rows)

    table('tab_defense2.tex',
          r'Defense and detector observations from different study subsets. Perplexity AUC compares interleaving with direct harmful English, not benign inputs; a low AUC indicates score-direction reversal. The output filter is a retrospective MD-Judge screen against Qwen3Guard-defined successes. Benign utility and deployment false-positive rates were not measured.',
          'tab:defense2', 'l l l', 'Intervention/statistic & Stage & Reported result', [
              ['Self-Reminder-style prompt', 'pre-generation', r'gated $0.693\to0.692$'],
              ['Full-sequence perplexity', 'input', 'AUC 0.016 (reversed direction)'],
              ['Windowed-max perplexity', 'input', 'AUC 0.980 (different sample)'],
              ['Paraphrase then answer', 'pre-generation', r'gated $0.693\to0.408$'],
              ['MD-Judge output screen', 'post-generation', r'gated $0.700\to0.079$']])

    corrections = {
        'tab_sota_cross.tex': r'Supplementary descriptive results for selected obfuscation implementations. Qwen uses 1,727 test items; GPT-4o baseline records use 250 items. The GPT-4o PolyJigsaw value comes from a separate structural panel whose item-ID intersection with these baselines was not reverified. Different settings and populations prevent a paired superiority or adaptive-policy claim. These observations do not establish general capability regimes.',
        'tab_sota.tex': r'Selected obfuscation implementations on Qwen2.5-7B, 1,727 test items. Reconstruction-gated scoring and an English refusal-dictionary score measure different events. Character flipping has gated ASR 0.000; word flipping has 0.020. These implementations do not reproduce all components of the published methods, and no human usefulness labels are implied.',
        'tab_litcompare.tex': r'Summary of recorded baseline implementations on the Qwen test collection. Row populations differ (notably span-CSRT, $N=649$); see the main table for coverage. The Base64 and payload-splitting rows are representatives, not reproductions of every cited method. Original-paper ASR values are not used for cross-study ranking.',
        'tab_frontier2.tex': r'Original commercial-model comparison on 200-item subsets, with primary-judge Wilson intervals and MD-Judge scores. These fixed displayed conditions differ from the later 250-item structural panel. Rankings depend on the safety judge; results concern the evaluated model versions, not all current commercial models.',
        'tab_frontier.tex': r'GPT-4o-mini on the original 200-item subset. Displayed conditions and primary-judge Wilson intervals are descriptive; the highest displayed rate is not a validated adaptive-policy estimate.',
        'tab_mechanism.tex': r'Full-set ordered Qwen sweep ($N=2,879$). The factorization $P(R\cap U)=P(R)P(U\mid R)$ is a probability identity on the same population. Conditional rates use the reconstructed subset for each load. Direct English uses the conventional unit reconstruction gate. This decomposition does not identify an internal causal mechanism.',
        'tab_matched.tex': r'Matched span comparison on the Qwen test split. Each pair uses the same eligible items and spans, while changing the reconstruction framing and requested output. $N$ varies with span count. The paired difference describes that complete prompt intervention, not a separately identified internal reconstruction process.',
        'tab_encoding.tex': r'Tested Base64 and payload-splitting variants on the Qwen test collection under reconstruction-gated scoring. Results concern these implementations and decoding settings, not general decoding ability or all encoding-based methods.',
    }
    corrections.update({
        'tab_defense.tex': r'Self-Reminder-style system-instruction comparison on Qwen2.5-7B. The displayed rates change little for ordered interleaving and more for direct English and translation. These marginal estimates do not by themselves establish paired significance or equivalence. The results concern the recorded intervention and sample.',
        'tab_attaq.tex': r'External AttaQ collection (Qwen2.5-7B, 1,402 items). Translation has the higher primary-judge estimate, while the cross-judge ranking differs. This is external evidence of joint failures, not universal superiority.',
        'tab_safetyjudge.tex': r'Three model safety judges on a 1,488-row stratified sample using the same reconstruction gate. Agreement with GPT-4o is 0.88 for Qwen3Guard and 0.83 for MD-Judge. Method rankings and agreement concern this selected sample; no human ground truth or general judge accuracy is established.',
        'tab_difficulty.tex': r'Category-level reconstruction and gated-ASR estimates for the recorded Qwen interleaving study. Conditional unsafe rates divide joint successes by reconstruction successes within each category. Category comparisons are descriptive and do not identify a causal safety mechanism.',
        'tab_scenario.tex': r'Per-scenario gated rates on the full Lingua collection for Qwen2.5-7B. These full-set descriptive estimates overlap the study development and test partitions.',
    })
    corrections.update({
        'tab_languages.tex': r'The ten study languages in the fixed order used by the language-load sweep. Script and family labels describe the inventory; the study does not isolate their causal effects or validate this order as optimal.',
        'tab_detect.tex': r'Input moderation on the recorded submitted requests. The classifier flags 75.7\% of ordered interleaved inputs and accepts 24.3\%. These harmful-input results do not supply benign false-positive rates or support a claim that the detector usually misses the construction.',
        'tab_mtrobust.tex': r'Translation-source sensitivity on the Qwen2.5-7B test study. Released benchmark translations are compared with NLLB-200 machine translations. The observed rates describe these sources and configurations; they do not establish independence from translation quality.',
        'tab_thinking.tex': r'Qwen3-8B reasoning-mode observations. Thinking OFF uses the test split; thinking ON uses a separate 240-item subset with a 5,120-token budget. Population and decoding differences prevent interpreting these rates as paired causal effects of thinking. Low reconstruction can include termination or formatting failures.',
        'tab_langpairs.tex': r'English plus one other language ($n=2$, ordered) on the Qwen2.5-7B test study. Gated rates vary from 0.595 to 0.687 across the evaluated pairs. Reconstruction and unsafe responding both vary; these observations do not isolate script distance, language family, or an optimal language order.',
        'tab_rawgated.tex': r'Raw unsafe and reconstruction-gated rates. Direct conditions use a unit reconstruction gate by scoring convention, not measured semantic fidelity. The conjunction is evaluated on each response; raw and conditional unsafe rates have different denominators.',
    })
    for filename, replacement in corrections.items():
        caption(filename, replacement)
    path = PAPER / 'tab_sota_cross.tex'
    text = path.read_text().replace('Qwen2.5-7B (weak)', 'Qwen2.5-7B').replace('GPT-4o (strong)', 'GPT-4o')
    text = text.replace('PolyJigsaw (adaptive difficulty, ours)', 'PolyJigsaw (separate panel)')
    path.write_text(text)
    path = PAPER / 'tab_litcompare.tex'
    path.write_text(path.read_text().replace('Character obfuscation / encoding', 'Base64 encoding').replace('Payload splitting / decomposition', 'Payload splitting'))
    path = PAPER / 'tab_litcompare.tex'
    text = path.read_text().replace(r'\begin{tabular}{l l c}', r'\begin{tabular}{p{0.38\linewidth} p{0.38\linewidth} c}')
    text = text.replace('gated ASR (our data)', 'gated ASR').replace(r'\citep{kang2024exploiting}, cf.\ \citep{li2024drattack}, \citep{ding2024renellm}', r'\citep{kang2024exploiting}')
    text = text.replace(r'cf.\ \citep{liu2024flipattack}', 'Base64 variant')
    path.write_text(text)
    provenance = {'scope': 'Existing aggregates used by reviewed table regeneration; no new experiment or raw-data rejudging.',
                  'script': 'scripts/make_reviewed_paper_tables.py', 'source_sha256': SOURCES,
                  'limitations': ['Secondary judge denominators are absent in some aggregates.',
                                  'Cross-model maxima are explicitly descriptive.',
                                  'Historical structural-panel item matching remains unverified.']}
    (PAPER / 'reviewed_table_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('reviewed manuscript tables regenerated')


if __name__ == '__main__':
    generate_reviewed_tables()
