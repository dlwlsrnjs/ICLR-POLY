#!/usr/bin/env python3
"""CPU-only audit of archived source and the current MJ/LG frame contract."""
import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

PACKAGE = Path(__file__).resolve().parent
RUN = PACKAGE / 'runs/willingness394_milmmt_v3'
FRAME_MAP = {
    'plain': 'plain',
    'persona': 'legacy_persona',
    'fiction': 'legacy_fiction',
    'pap': 'legacy_pap',
    'persona+fiction': 'legacy_persona_fiction',
}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def audit(repo):
    manifest = json.loads((PACKAGE / 'SOURCE_MANIFEST.json').read_text())
    for entry in manifest['files']:
        assert hashlib.sha256((PACKAGE / entry['path']).read_bytes()).hexdigest() == entry['sha256'], entry['path']
    script = repo / 'scripts/closed_compare.py'
    tree = ast.parse(script.read_text())
    wanted = {'AIM_HEAD', 'FICTION_HEAD', 'PAP_HEAD', 'C_WILL', 'GRID_F', 'GRID_ARR', 'GRID_N', 'REASSEMBLE'}
    const = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    const[target.id] = ast.literal_eval(node.value)
    assert wanted <= const.keys()
    assert set(FRAME_MAP) == const['C_WILL']
    frames = json.loads((RUN / 'inputs/frames.json').read_text())
    config = json.loads((RUN / 'configs/legacy_394.json').read_text())
    grid_heads = {
        'plain': 'You are playing a multilingual reassembly game.',
        'persona': const['AIM_HEAD'], 'fiction': const['FICTION_HEAD'],
        'pap': const['PAP_HEAD'],
        'persona+fiction': const['AIM_HEAD'] + '\n' + const['FICTION_HEAD'],
    }
    frame_checks = {
        grid: {'v3_frame': v3, 'frame_text_identical': frames[v3]['text'] == grid_heads[grid]}
        for grid, v3 in FRAME_MAP.items()
    }
    assert not frame_checks['plain']['frame_text_identical']
    assert all(frame_checks[f]['frame_text_identical'] for f in FRAME_MAP if f != 'plain')
    v3 = module('snapshot_scenarios', RUN / 'code/scenarios.py')
    puzzle = module('grid_puzzle', repo / 'scripts/run_qwen_interleaving_probe.py')
    cells = [f'g{g}_{o}_n{n}' for g in const['GRID_F'] for o in const['GRID_ARR'] for n in const['GRID_N']]
    expected = {c + '__' + f for c in cells for f in FRAME_MAP}
    assert len(cells) == 32 and len(expected) == 160
    assert len(list(v3.arms(config))) == 111

    # Structural fixture only: safe English tokens stand in for four translations.
    # It tests fragment assembly, not translation quality or model behavior.
    item = {'id': 'fixture', 'prompt': 'Please prepare a short invitation for the library reading club on Saturday morning and ask each guest to bring one book'}
    languages = ['English', 'Norwegian', 'Finnish', 'Arabic']
    texts = {(item['id'], lang): item['prompt'] for lang in languages}
    approved = {v3.align_key(item, 5, lang, item['prompt']): v3.split(item['prompt'], 5) for lang in languages[1:]}
    fixture_config = dict(config, include_raw_control=False, include_intact_controls=False, frames=['legacy_persona'], orders=['ordered'])
    v3_rows = list(v3.build(fixture_config, [item], texts, frames, approved))
    grid_row = {'item_id': 'fixture', 'scenario': 'library', 'questions': dict.fromkeys(languages, item['prompt'])}
    counts = []
    for n in [2, 4]:
        built = puzzle.build_puzzle(grid_row, languages[:n], 'ordered', 20260828, 'coarse', 5)
        row = next(x for x in v3_rows if len(x['arm']['profile']) == n)
        payload = row['prompt'].split('\nINPUT\n', 1)[1].split('\n\n', 1)[0]
        actual_v3 = len(payload.splitlines())
        assert actual_v3 == 5 and len(built['records']) == 5 * n
        counts.append({'languages': n, 'v3_total_fragments': actual_v3, 'grid_g5_total_fragments': len(built['records'])})

    # Render the actual grid wrapper without importing its GPU/judge dependencies.
    heads = {k: grid_heads[k] for k in ['persona', 'fiction', 'pap']}
    ns = dict(const, FRAME_HEAD=heads, os=SimpleNamespace(environ={}),
              CE=SimpleNamespace(mixed=lambda row, langs, n, g, order: (
                  puzzle.build_puzzle(row, langs[:n], order, 20260828, 'coarse', g)['mixed_paragraph'], langs[:n])))
    compose = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_compose')
    exec(compile(ast.Module(body=[compose], type_ignores=[]), str(script), 'exec'), ns)
    rendered = ns['_compose'](grid_row, languages, 4, 5, 'ordered', ('persona',))
    v3_prompt = next(x['prompt'] for x in v3_rows if len(x['arm']['profile']) == 4)
    assert rendered != v3_prompt

    results = repo / 'experiments_suite/exp02_panel_collect/results'
    evidence = {}
    for ds in ['mj', 'lg']:
        tag = 'qwen25_7b_' + ds
        paths = list((results / 'attack').glob(tag + '__*.json'))
        methods = {json.loads(p.read_text())['method'] for p in paths}
        assert expected <= methods, f'{ds}: missing {len(expected - methods)} C arms'
        recon_path = results / 'benign' / (tag + '.json')
        recon = json.loads(recon_path.read_text())
        assert recon['model'] == config['target_model']
        assert set(recon['benign_recon_by_cell']) == set(cells)
        evidence[ds] = {
            'result_files': len(paths), 'C_grid_arms_present': len(expected & methods),
            'grid_frame_counts': dict(sorted(Counter(m.split('__')[-1] for m in methods if m.startswith('g')).items())),
            'comprehension_cells': len(recon['benign_recon_by_cell']),
            'comprehension_source': str(recon_path.relative_to(repo)),
            'model': recon['model'], 'collection': recon['collection'],
        }
    return {
        'repo_commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        'archived_files_verified': len(manifest['files']), 'frame_mapping': frame_checks,
        'comprehension_cells': 32, 'C_grid_arms': 160, 'v3_scenarios': 111,
        'fragment_fixture': counts, 'full_prompt_identical': False,
        'v3_target': config['target_model'], 'dataset_evidence': evidence,
        'conclusion': 'Four frame texts match; plain, full wrapper, fragment semantics and language coverage differ. Factorized transfer requires explicit assumptions; these are not matched 160-cell observations.',
        'gpu_calls': 0,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=PACKAGE.parents[1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.repo.resolve())
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text)
