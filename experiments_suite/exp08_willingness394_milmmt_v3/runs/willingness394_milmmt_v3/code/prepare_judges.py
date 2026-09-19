import argparse,json,shutil
from pathlib import Path
from scenarios import R,load,read,digest
p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();c,items,_,_=load(a.config);folder=R/'outputs'/c['name'];assert json.loads((folder/'build_manifest.json').read_text())['config_sha256']==digest(c),'config differs from built jobs';originals={x['id']:x['prompt'] for x in items};rows={}
for path in sorted(folder.glob('responses.shard*.jsonl')):
 for x in read(path):
  if x['key'] in rows:assert rows[x['key']]==x,'conflicting duplicate response'
  rows[x['key']]=x
assert rows,'No responses collected'
jroot=folder/'judge'
for sub in ['inputs','outputs','code']:(jroot/sub).mkdir(parents=True,exist_ok=True)
old=R.parent/'wildguard_labeling_v1/code'
for name in ['batch_label.py','label.py']:
 dst=jroot/'code'/name
 if dst.exists():assert dst.read_bytes()==(old/name).read_bytes(),'judge source changed; use separate run'
 else:shutil.copyfile(old/name,dst)
prior={x['key']:x for x in read(jroot/'inputs/jobs.jsonl')};new=[]
for x in rows.values():
 y={'key':x['key'],'original_request':originals[x['id']],'response':x['response'],'finish_reason':x['finish_reason']}
 if y['key'] in prior:assert prior[y['key']]==y,'judge input changed at same key'
 new.append(y)
assert set(prior)<=set(rows),'previous judge inputs removed'
(jroot/'inputs/jobs.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in new))
print(jroot.resolve())
