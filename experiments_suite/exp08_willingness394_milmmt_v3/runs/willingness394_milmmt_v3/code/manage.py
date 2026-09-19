import argparse,json,hashlib
from pathlib import Path
from collections import Counter
from scenarios import R,load,arms,build,banks,alignment_jobs,digest
p=argparse.ArgumentParser();p.add_argument('command',choices=['plan','build']);p.add_argument('--config',required=True);a=p.parse_args()
c,items,texts,frames=load(a.config);out=R/'outputs'/c['name'];out.mkdir(exist_ok=True)
counts=Counter();byarm=Counter();ready=0;fp=(out/'jobs.jsonl').open('w') if a.command=='build' else None
for x in build(c,items,texts,frames,banks()):
 if x['ready']:
  ready+=1;byarm[x['arm_id']]+=1
  if fp:fp.write(json.dumps(x,ensure_ascii=False)+'\n')
 else:counts[x['reason']]+=1
if fp:fp.close()
catalog=[x|{'arm_id':digest(x)[:16],'ready_items':byarm[digest(x)[:16]]} for x in arms(c)]
(out/'arms.json').write_text(json.dumps(catalog,indent=2))
report={'config':c,'config_sha256':digest(c),'items':len(items),'arms':len(catalog),'candidate_jobs':len(catalog)*len(items),'ready_jobs':ready,'blocked_jobs':dict(counts),'alignment_tasks':sum(1 for _ in alignment_jobs(c,items,texts)),'built':a.command=='build','human_gold':False}
(out/'plan.json').write_text(json.dumps(report,indent=2))
if a.command=='build':
 report['files']=[{'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [out/'jobs.jsonl',R/'inputs/items.json',R/'inputs/accepted_translations.json',R/'inputs/frames.json',R/'code/scenarios.py'] + [p for p in [R/'outputs/alignment.jsonl',R/'outputs/alignment_qa.jsonl'] if p.exists()]]
 (out/'build_manifest.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='config'},indent=2))
