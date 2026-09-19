#!/usr/bin/env python3
"""CPU structural audit; this does not certify translation meaning or model reconstruction."""
import argparse
import json
from pathlib import Path
from grid_contract import GridContract, FRAMES, digest
from prepare import build


def audit(inputs, repo):
    items = json.loads((inputs/'items.json').read_text())
    queue = json.loads((inputs/'translation_queue.json').read_text())
    accepted = json.loads((inputs/'accepted_translations.json').read_text())
    config = json.loads((inputs/'configs/lg_qwen25_7b.json').read_text())
    contract = GridContract(repo, config['renderer_sha256'])
    translated = {r['id']:r['translation_evidence'] for r in queue if r['language']=='Norwegian'}
    count = 0
    for item in items:
        t = translated[item['id']]
        if not t: continue
        row = dict(item_id=item['id'], scenario='Benign Control',
                   questions={'English':item['prompt'], 'Norwegian':t['translated']})
        payloads = set()
        for frame in FRAMES:
            result = contract.render(row, config['languages'], 3, 'ordered', frame)
            payloads.add(result['payload_sha256'])
            byid = {r['id']:r for r in result['fragment_records']}
            for lang, ids in result['gold_ids'].items():
                rebuilt = ' '.join(byid[i]['text'] for i in ids)
                if rebuilt.split() != row['questions'][lang].split():
                    raise ValueError(f'Fragment loss/reordering: {item["id"]}, {lang}')
            count += 1
        if len(payloads)!=1: raise ValueError('Frame changed fragment payload')
    configs = sorted((inputs/'configs').glob('*.json'))
    status = {}
    for path in configs:
        c = json.loads(path.read_text())
        jobs, blocked = build(c, items, accepted, repo)
        if jobs or len(blocked)!=331:
            raise ValueError('Unreviewed translations must not enter production')
        status[path.stem] = dict(eligible_items=0, blocked_items=len(blocked))
    return dict(schema='curated_bank_cpu_audit_v1', items=331, items_sha256=digest(items),
                lg_structural_prompts_checked=count, frame_payload_invariance=True,
                english_norwegian_roundtrip='equal after whitespace normalization',
                production_configs_checked=len(configs), production_status=status,
                semantic_translation_qa='not run', target_model_inference='not run',
                willingness_covariance='not estimable: no new target responses',
                legacy_responses_used=False)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
    p.add_argument('--out', type=Path, required=True)
    a=p.parse_args()
    result=audit(a.inputs,a.repo)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='production_status'},indent=2))
