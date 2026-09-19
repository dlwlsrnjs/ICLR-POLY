import json,hashlib
from collections import Counter
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def read(p):return [json.loads(l) for l in p.open() if l.strip()]
items=json.loads((R/'inputs/items.json').read_text());ts=read(R/'outputs/translations.jsonl');qs=read(R/'outputs/translation_qa.jsonl')
expected={x['id']+'|'+l for x in items for l in ['Norwegian','Finnish','Arabic']};t={x['key']:x for x in ts};q={x['key']:x for x in qs}
assert len(ts)==len(qs)==len(expected)==1182 and set(t)==set(q)==expected
accepted=[{'id':x['id'],'language':'English','text':x['prompt'],'qa':'original_english'} for x in items];excluded=[]
for k,x in t.items():
 y=q[k];valid=x['forward_terminated'] and x['back_terminated'] and y['finish_reason']=='stop' and y.get('parsed',{}).get('equivalent') is True
 if valid:accepted.append({'id':x['id'],'language':x['language'],'text':x['translated'],'qa':'qwen32_semantic_check_not_human_gold','translation_model':x['translation_model'],'translation_revision':x['model_revision']})
 else:excluded.append({'key':k,'forward_terminated':x['forward_terminated'],'back_terminated':x['back_terminated'],'qa':y.get('parsed',{}),'qa_finish_reason':y['finish_reason']})
(R/'inputs/accepted_translations.json').write_text(json.dumps(accepted,ensure_ascii=False,indent=2))
(R/'outputs/translation_exclusions.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2))
old=json.loads((R.parent/'willingness394_scenarios_v2/inputs/accepted_translations.json').read_text());oldkeys={(x['id'],x['language']) for x in old};newkeys={(x['id'],x['language']) for x in accepted}
report={'source_items':394,'translations':len(ts),'accepted':len(accepted)-394,'excluded':len(excluded),'accepted_by_language':dict(Counter(x['language'] for x in accepted)),'nllb_previous_accepted_by_language':dict(Counter(x['language'] for x in old)),'newly_accepted_vs_nllb':len(newkeys-oldkeys),'no_longer_accepted_vs_nllb':len(oldkeys-newkeys),'human_gold':False,'quality_note':'Automatic pass counts (new rubric also explicitly checks target language), not proof of higher translation quality. Compare common source items. No fallback to NLLB.'}
(R/'outputs/translation_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
