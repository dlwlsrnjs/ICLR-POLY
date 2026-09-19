#!/usr/bin/env python3
"""Recover QA-accepted MiLMMT translations by exact original text, never ID alone."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recover(inputs, previous, out):
    items = json.loads((inputs/'items.json').read_text())
    originals = {r['prompt']:r['id'] for r in items}
    if len(originals)!=len(items): raise ValueError('Ambiguous duplicate originals')
    paths = {k:previous/v for k,v in {
        'accepted':'inputs/accepted_translations.json',
        'translations':'outputs/translations.jsonl',
        'qa':'outputs/translation_qa.jsonl',
        'metadata':'outputs/translation_qa.metadata.json'}.items()}
    accepted = json.loads(paths['accepted'].read_text())
    accepted_keys = {(r['id'],r['language'],r['text']) for r in accepted if r['qa']=='qwen32_semantic_check_not_human_gold'}
    translations = [json.loads(l) for l in paths['translations'].read_text().splitlines() if l.strip()]
    qs = [json.loads(l) for l in paths['qa'].read_text().splitlines() if l.strip()]
    qa = {(r['id'],r['language']):r for r in qs}
    if len(qa)!=len(qs): raise ValueError('Duplicate QA identity')
    metadata = json.loads(paths['metadata'].read_text())
    if metadata['revision']!='5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd':
        raise ValueError('Unexpected semantic judge revision')
    output = [dict(id=r['id'],language='English',text=r['prompt'],qa='original_english') for r in items]
    matches, reused, evidence = [], [], []
    for t in translations:
        if t['original'] not in originals or t['language'] not in ('Norwegian','Bengali'): continue
        matches.append(t['language'])
        if (t['id'],t['language'],t['translated']) not in accepted_keys: continue
        q=qa[(t['id'],t['language'])]
        if not (t['forward_terminated'] is True and t['back_terminated'] is True
                and q['finish_reason']=='stop' and q['parsed'].get('equivalent') is True
                and all(t[k] in q['prompt'] for k in ['original','translated','backtranslation'])):
            raise ValueError('Accepted translation lacks matching completed QA')
        if 'MiLMMT' not in t.get('translation_model',''): raise ValueError('Different translator family')
        item_id=originals[t['original']]
        output.append(dict(id=item_id,language=t['language'],text=t['translated'],original=t['original'],
                           qa='qwen32_semantic_check_not_human_gold',source_id=t['id'],
                           source_file_sha256=sha(paths['translations'])))
        evidence.append(dict(id=item_id,translation=t,judgment=q,judge_metadata=metadata))
        reused.append(t['language'])
    keys=[(r['id'],r['language']) for r in output]
    if len(keys)!=len(set(keys)): raise ValueError('Ambiguous matching translations')
    report=dict(schema='existing_translation_reuse_v1',items=len(items),matching_originals=dict(Counter(matches)),
                reused_nonenglish=dict(Counter(reused)),source_files={str(v):sha(v) for v in paths.values()},
                policy='exact original + exact accepted translation + completed matching Qwen32 QA; MiLMMT only',
                reused_ids=[r['id'] for r in output if r['language']!='English'])
    files={'accepted_translations.json':output,'reuse_evidence.json':evidence,'reuse_report.json':report}
    for name,data in files.items():
        p=out/name
        if p.exists() and json.loads(p.read_text())!=data: raise ValueError('Changed reuse output; use a new directory')
    out.mkdir(parents=True,exist_ok=True)
    for name,data in files.items(): (out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(recover(a.inputs,a.previous,a.out),indent=2))
