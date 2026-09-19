#!/usr/bin/env python3
"""Connect the confirmed 331-item over-refusal bank without inventing translation QA."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from grid_contract import digest

BANK_SHA = 'ec7f615c515cc00ea1924466b78233b56598b6a783902d4e957269ec390577de'


def connect(source, model_file, out):
    source, out = Path(source), Path(out)
    raw = (source/'inputs/items.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != BANK_SHA:
        raise ValueError('Not the confirmed curated bank: source hash differs')
    rows = json.loads(raw)
    if len(rows) != 331 or len({r['item_id'] for r in rows}) != 331:
        raise ValueError('Expected 331 unique source items')
    if any(r['bare_refuse_count'] < 5 or r['bare_refuse_models'] != 9 for r in rows):
        raise ValueError('Screening evidence does not match the selected bank')
    items = [dict(r, id=r['item_id'], prompt=r['english_original']) for r in rows]
    models = json.loads(Path(model_file).read_text())
    if len(models) != 17 or len({m['tag'] for m in models}) != 17:
        raise ValueError('Expected 17 distinct panel models')
    originals = {r['id']: r['prompt'] for r in items}
    translations = [json.loads(l) for l in (source/'translations/translations.jsonl').read_text().splitlines() if l.strip()]
    bank = {}
    for t in translations:
        key = (t['item_id'], t['language'])
        if key in bank or originals.get(key[0]) != t['english_original']:
            raise ValueError('Duplicate or mismatched translation source')
        bank[key] = t
    # Existing legacy judgments omit termination and original/backtranslation hashes.
    # Preserve them for review, but never manufacture the stronger QA contract.
    accepted = [dict(id=r['id'], language='English', text=r['prompt'], qa='original_english') for r in items]
    pending = []
    for r in items:
        for language in ('Norwegian', 'Bengali'):
            t = bank.get((r['id'], language))
            pending.append(dict(id=r['id'], language=language, original=r['prompt'],
                                state='needs_semantic_qa' if t else 'needs_translation',
                                translation_evidence=t))
    config_dir = Path(__file__).parent/'configs'
    files = {'items.json': items, 'accepted_translations.json': accepted,
             'translation_queue.json': pending, 'models.json': models}
    for dataset in ('mj', 'lg'):
        base = json.loads((config_dir/f'{dataset}_qwen7b.json').read_text())
        for m in models:
            config = dict(base, expected_items=331, corpus='overrefusal331_min5of9',
                          corpus_sha256=digest(items), target_model=m['model'],
                          target_revision=m['revision'], model_tag=m['tag'])
            files[f'configs/{dataset}_{m["tag"]}.json'] = config
    manifest = dict(schema='curated_bank_connection_v1', source_items_sha256=BANK_SHA,
                    source=str(source.resolve()), rows=331,
                    by_dataset=dict(Counter(r['source_dataset'] for r in rows)),
                    screening='bare_refuse_count >= 5 among 9 screened models',
                    user_confirmed_count=331, previous_count_311_corrected=True,
                    cell='g3_ordered_n2', models=17, profiles=['mj','lg'],
                    planned_jobs_per_profile=331*5*17,
                    translation_queue=dict(Counter(r['state'] for r in pending)),
                    accepted_nonenglish_translations=0,
                    prior_g5_ordered_n4_responses_reused=False,
                    artifacts={name:digest(data) for name,data in files.items()})
    files['connection_manifest.json'] = manifest
    # Refuse changed output before writing any file.
    for name, data in files.items():
        path = out/name
        if path.exists() and json.loads(path.read_text()) != data:
            raise ValueError(f'Changed artifact; choose a fresh output directory: {name}')
    for name, data in files.items():
        path = out/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(connect(a.source, a.models, a.out), indent=2))
