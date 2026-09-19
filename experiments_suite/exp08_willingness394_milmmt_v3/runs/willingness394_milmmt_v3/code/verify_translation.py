import json,hashlib,py_compile
from pathlib import Path
from collections import Counter
R=Path(__file__).resolve().parents[1]
def read(p):return [json.loads(l) for l in p.open() if l.strip()]
items=json.loads((R/'inputs/items.json').read_text());originals={x['id']:x['prompt'] for x in items};assert len(originals)==394
olditems=json.loads((R.parent/'willingness394_scenarios_v2/inputs/items.json').read_text());assert items==olditems
meta=json.loads((R/'outputs/translation.metadata.json').read_text());assert meta['model']=='xiaomi-research/MiLMMT-46-12B-v1.0' and meta['vllm']=='0.8.5' and meta['gpu']==1
assert meta['protocol_sha256']==hashlib.sha256((R/'code/translate.py').read_bytes()).hexdigest()
rows=read(R/'outputs/translations.jsonl');assert len(rows)==len({x['key'] for x in rows})==1182
expected={x['id']+'|'+l for x in items for l in ['Norwegian','Finnish','Arabic']};assert {x['key'] for x in rows}==expected
for x in rows:
 assert x['original']==originals[x['id']] and x['translation_model']==meta['model'] and x['model_revision']==meta['revision']
 assert x['forward_prompt']==meta['prompt_template'].format(source='English',target=x['language'],text=x['original'])
 assert x['back_prompt']==meta['prompt_template'].format(source=x['language'],target='English',text=x['translated'])
accepted=json.loads((R/'inputs/accepted_translations.json').read_text());q={x['key']:x for x in read(R/'outputs/translation_qa.jsonl')};by={x['key']:x for x in rows}
for x in accepted:
 if x['language']=='English':assert x['text']==originals[x['id']];continue
 k=x['id']+'|'+x['language'];assert x['text']==by[k]['translated'] and q[k]['parsed']['equivalent'] is True and q[k]['finish_reason']=='stop' and by[k]['forward_terminated'] and by[k]['back_terminated']
for p in (R/'code').glob('*.py'):py_compile.compile(str(p),doraise=True)
r={'passed':True,'same_394_originals':True,'translations_checked':len(rows),'accepted_checked':len(accepted),'model_revision':meta['revision'],'forward_finish_reasons':dict(Counter(x['forward_finish_reason'] for x in rows)),'back_finish_reasons':dict(Counter(x['back_finish_reason'] for x in rows)),'human_translation_quality_validated':False}
(R/'outputs/verification.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
