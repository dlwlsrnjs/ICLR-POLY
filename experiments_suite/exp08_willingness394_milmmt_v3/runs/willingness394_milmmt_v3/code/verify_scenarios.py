import json,copy,py_compile
from collections import defaultdict,Counter
from scenarios import R,load,build,arms,align_key,read,norm,digest
# Synthetic, alignment-known four-language fixture: structural checks only.
c,_,_,frames=load(R/'configs/expanded_394.json');c=copy.deepcopy(c)
c.update(frames=['plain','persona_archivist'],fragment_counts=[5],languages=['English','Norwegian','Finnish','Arabic'],mixed_profiles=[['English','Norwegian','Finnish','Arabic']],include_raw_control=False,include_intact_controls=False,monolingual_puzzles=False)
item={'id':'fixture','prompt':'one two three four five'}
text={('fixture','English'):item['prompt'],('fixture','Norwegian'):'en to tre fire fem',('fixture','Finnish'):'yksi kaksi kolme neljä viisi',('fixture','Arabic'):'واحد اثنان ثلاثة أربعة خمسة'}
approved={align_key(item,5,l,t):t.split() for (id,l),t in text.items() if l!='English'}
rows=list(build(c,[item],text,frames,approved));assert len(rows)==6 and all(x['ready'] for x in rows)
payloads=defaultdict(set)
for x in rows:payloads[x['order']].add(x['payload_sha256'])
assert all(len(v)==1 for v in payloads.values()) and len({next(iter(v)) for v in payloads.values()})==3
assert all(not x['ready'] for x in build(c,[item],text,frames,{}))
short={'id':'short','prompt':'one two'};assert all(x.get('reason')=='source_too_short' for x in build(c,[short],{},frames,{}))
base=rows[0]['key'];changed=copy.deepcopy(c);changed['target_revision']='changed';assert next(build(changed,[item],text,frames,approved))['key']!=base
c,items,texts,frames=load(R/'configs/legacy_394.json');assert len(list(arms(c)))==111
jobs=read(R/'outputs'/c['name']/'jobs.jsonl');assert len(jobs)==len({x['key'] for x in jobs})
groups=defaultdict(set)
for x in jobs:
 arm=x['arm'];group=(x['id'],arm['kind'],tuple(arm['profile']),arm['n'],arm['order'],arm['rotation']);groups[group].add(x['payload_sha256'])
assert all(len(v)==1 for v in groups.values())
responses=[]
for p in (R/'outputs'/c['name']).glob('responses.shard*.jsonl'):responses+=read(p)
by={x['key']:x for x in jobs}
for x in responses:
 assert x['key'] in by and x['prompt']==by[x['key']]['prompt'] and x['system']==by[x['key']]['system']
for p in (R/'code').glob('*.py'):py_compile.compile(str(p),doraise=True)
report={'passed':True,'structural_four_language_fixture':True,'unapproved_alignment_blocked':True,'short_items_blocked':True,'model_revision_changes_key':True,'frame_payload_equality':True,'ready_jobs_checked':len(jobs),'live_responses':len(responses),'live_profiles':dict(Counter(x['language'] for x in responses)),'live_finish_reasons':dict(Counter(x['finish_reason'] for x in responses)),'semantic_label_accuracy_validated':False}
(R/'outputs/scenario_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
