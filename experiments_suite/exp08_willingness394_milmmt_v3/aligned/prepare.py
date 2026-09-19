#!/usr/bin/env python3
"""Build five frame jobs per eligible benign item at a frozen comprehension cell."""
import argparse
import json
from pathlib import Path
import re
from grid_contract import FRAMES, GridContract, digest


def build(config, items, translations, repo):
    if config.get('system_message') is not None or config.get('strong_reconstruction_prompt') is not False:
        raise ValueError('Only the default user-only panel contract is supported')
    if config['frames'] != FRAMES: raise ValueError('Frame order differs from canonical five frames')
    ids = [x['id'] for x in items]
    if len(ids) != config['expected_items'] or len(set(ids)) != len(ids):
        raise ValueError('Wrong corpus size or duplicate item IDs')
    if any(not isinstance(x.get('prompt'), str) or not x['prompt'].strip() for x in items):
        raise ValueError('Empty source request')
    match = re.fullmatch(r'g(3|5|8|12)_(ordered|shuffled)_n(2|4|6|8)', config['cell'])
    if not match: raise ValueError('Invalid C-space cell')
    g, order, n = int(match[1]), match[2], int(match[3])
    languages = config['languages']
    if n != len(languages) or languages[0] != 'English' or len(set(languages)) != n:
        raise ValueError('Language profile does not match cell')
    contract = GridContract(repo, config['renderer_sha256'])
    bank = {}
    originals = {x['id']: x['prompt'] for x in items}
    for x in translations:
        key = (x['id'], x['language'])
        if x['id'] not in originals: raise ValueError('Translation belongs to a different item bank')
        if key in bank: raise ValueError('Duplicate translation key')
        expected_qa = 'original_english' if x['language'] == 'English' else 'qwen32_semantic_check_not_human_gold'
        if x.get('qa') != expected_qa or not str(x.get('text', '')).strip():
            raise ValueError('Translation is not explicitly QA-accepted')
        if x['language'] == 'English' and x['text'] != originals[x['id']]:
            raise ValueError('English translation entry differs from frozen original')
        if 'original' in x and x['original'] != originals[x['id']]:
            raise ValueError('Translation source differs from frozen original')
        bank[key] = x['text']
    jobs, blocked = [], []
    for item in items:
        questions = {'English': item['prompt']}
        missing = [l for l in languages[1:] if (item['id'], l) not in bank]
        if missing:
            blocked.append({'id': item['id'], 'missing_languages': missing}); continue
        questions.update({l: bank[(item['id'], l)] for l in languages[1:]})
        row = {'item_id': item['id'], 'scenario': 'Benign Control', 'questions': questions}
        payloads = set()
        for frame in FRAMES:
            rendered = contract.render(row, languages, g, order, frame)
            payloads.add(rendered['payload_sha256'])
            job = {'id': item['id'], 'dataset': config['dataset'], 'cell': config['cell'],
                   'frame': frame, 'arm': config['cell']+'__'+frame, 'original_request': item['prompt'], **rendered}
            job['key'] = digest({'config': config, 'job': job})
            jobs.append(job)
        if len(payloads) != 1: raise AssertionError('Frames changed the puzzle payload')
    return jobs, blocked


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--items', type=Path, required=True)
    p.add_argument('--translations', type=Path, required=True, help='QA-accepted full sentences, same schema as original accepted_translations.json')
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    config, items, translations = [json.loads(f.read_text()) for f in [a.config, a.items, a.translations]]
    jobs, blocked = build(config, items, translations, a.repo)
    manifest = {'schema': 'benign_fixed_cell_v1', 'config': config, 'config_sha256': digest(config),
                'items_sha256': digest(items), 'translations_sha256': digest(translations), 'jobs_sha256': digest(jobs),
                'eligible_items': len(jobs)//5, 'candidate_items': len(items), 'jobs': len(jobs),
                'blocked_items': len(blocked), 'same_items_for_all_five_frames': True,
                'missing_translation_policy': 'exclude item from all five frames, never substitute another language'}
    a.out.mkdir(parents=True, exist_ok=True)
    mp = a.out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise SystemExit('Run inputs changed; choose a new --out directory')
    (a.out/'jobs.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in jobs))
    (a.out/'blocked.json').write_text(json.dumps(blocked, indent=2)+'\n')
    mp.write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k: manifest[k] for k in ['eligible_items','candidate_items','jobs','blocked_items']}, indent=2))
