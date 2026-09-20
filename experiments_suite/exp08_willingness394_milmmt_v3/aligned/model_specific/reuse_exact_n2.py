#!/usr/bin/env python3
"""Reuse only byte-identical rendered n2 inputs with identical model settings."""
import argparse,hashlib,json
from pathlib import Path
from analyze import write

def reuse(old,new):
 audit=[]
 for src in sorted((old/'calibration').glob('*/responses.jsonl')):
  tag=src.parent.name;dest=new/'calibration'/tag
  if (dest/'responses.jsonl').exists():raise ValueError('Destination already has responses')
  man=json.loads((dest/'manifest.json').read_text());jobs=[json.loads(l) for l in (dest/'jobs.jsonl').open()];lookup={(x['id'],x['cell']):x for x in jobs};rows=[]
  oldman=json.loads((src.parent/'manifest.json').read_text());om=oldman['config'];nm=man['config']
  for k in ['target_model','target_revision','sampling','max_model_len','system_message','strong_reconstruction_prompt','renderer_sha256']:assert om[k]==nm[k]
  for x in map(json.loads,src.open()):
   if x['cell']!='g3_ordered_n2':continue
   job=lookup[x['id'],x['cell']]
   for k in ['prompt','messages','original_request','fragment_records','gold_ids','payload_sha256','languages']:assert x[k]==job[k],(tag,k)
   assert hashlib.sha256(x['response'].encode()).hexdigest()==x['response_sha256']
   rows.append({**x,**job,'reused_from':{'path':str(src),'key':x['key'],'source_config_sha256':oldman['config_sha256'],'reason':'Exact same model revision, sampling and rendered English+Arabic input.'}})
  if not rows:continue
  (dest/'responses.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
  meta=json.loads((src.parent/'collection_metadata.json').read_text());meta.update(config_sha256=man['config_sha256'],jobs_sha256=man['jobs_sha256']);write(dest/'collection_metadata.json',meta)
  audit.append(dict(model=tag,reused_n2_responses=len(rows),source_file=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),source_metadata_sha256=hashlib.sha256((src.parent/'collection_metadata.json').read_bytes()).hexdigest()))
 write(new/'REUSE_AUDIT.json',dict(models=audit,total_reused=sum(x['reused_n2_responses'] for x in audit),all_n4_plus_recomputed=True))
 print('reused',sum(x['reused_n2_responses'] for x in audit))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--new',type=Path,required=True);a=p.parse_args();reuse(a.old,a.new)
