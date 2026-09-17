"""Offline ownership/coverage audit. Does not load models or send queries."""
import json,sys
from collections import Counter
from pathlib import Path

def audit(plan):
 models=plan['models'];settings=plan['settings'];ds=plan['datasets']
 assert len(models)==17 and len({x['model'] for x in models})==17 and len({x['tag'] for x in models})==17,'model duplicates or omissions'
 expected={f'g{f}_{o}_n{n}__{w}' for f in [3,5,8,12] for o in ['ordered','shuffled'] for n in [2,4,6,8] for w in ['plain','persona','fiction','pap','persona+fiction']}
 assert len(settings)==160 and set(settings)==expected,'settings differ from frozen 160'
 assert set(ds)=={'mj','lg'}
 assert ds['mj']['items']==315 and ds['mj']['positions_inclusive']==[0,314]
 assert ds['lg']['items']==250 and ds['lg']['positions_inclusive']==[0,249]
 assert {x['owner'] for x in models}=={'primary_large','l40s_small'}
 units=[]
 for x in models:
  for d,meta in ds.items():units.append({'owner':x['owner'],'model':x['model'],'tag':x['tag'],'dataset':d,'first_position':0,'last_position':meta['items']-1,'settings':160,'responses':meta['items']*160})
 assert len({(x['model'],x['dataset']) for x in units})==34
 counts=Counter()
 for u in units:counts[u['owner']]+=u['responses']
 assert sum(counts.values())==1536800
 return {'status':plan['status'],'units':units,'responses_by_owner':dict(counts),'total_responses':sum(counts.values()),'coverage_valid':True,'activation_verified':False}
if __name__=='__main__':
 p=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).with_name('work_plan.json')
 print(json.dumps(audit(json.loads(p.read_text())),indent=2))
