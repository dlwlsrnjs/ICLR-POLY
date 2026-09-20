#!/usr/bin/env python3
"""Run exactly one stage with the unchanged bb69581 judge implementation."""
import argparse,gc,hashlib,importlib.util,json,os,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--stage',choices=['reconstruction','fulfillment'],required=True);p.add_argument('--batch',type=int,default=16);p.add_argument('--gpu',default='0');p.add_argument('--code',type=Path,default=Path(__file__).with_name('qwen32_dual_judge.py'));a=p.parse_args()
 os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 spec=importlib.util.spec_from_file_location('pinned_dual_judge',a.code);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 assert a.model.resolve().name==mod.REVISION
 source=list(mod.rows(a.run/'responses.jsonl'));assert len(source)==int(json.loads((a.run/'manifest.json').read_text())['jobs'])
 expected={x['key']:x['response_sha256'] for x in source};assert len(expected)==len(source)
 out=a.run/'judge_qwen32';out.mkdir(exist_ok=True)
 protocol={'schema':'qwen32_dual_judge/v1','model':str(a.model.resolve()),'revision':mod.REVISION,'responses_sha256':mod.sha(a.run/'responses.jsonl'),'temperature':0,'seed':20260920,'reconstruction_rubric_sha256':hashlib.sha256(mod.RECON_SYSTEM.encode()).hexdigest(),'fulfillment_rubric_sha256':hashlib.sha256(mod.FULFILL_SYSTEM.encode()).hexdigest()}
 pf=out/'protocol.json'
 if pf.exists():
  old=json.loads(pf.read_text());assert {k:v for k,v in old.items() if k!='model'}=={k:v for k,v in protocol.items() if k!='model'}
  if old['model']!=protocol['model']:
   backup=out/'protocol.before_stage_handoff.json'
   if not backup.exists():backup.write_bytes(pf.read_bytes())
 pf.write_text(json.dumps(protocol,indent=2)+'\n')
 target=out/f'{a.stage}.jsonl'
 def coverage():
  good=set()
  for x in list(mod.rows(target)) if target.exists() else []:
   assert expected.get(x['key'])==x['source_response_sha256']
   typed=isinstance(x.get('equivalent'),bool) if a.stage=='reconstruction' else x.get('label') in ['full','partial','none','uncertain']
   if x.get('valid') is True and typed:good.add(x['key'])
  return len(good)
 if coverage()==len(source):print(json.dumps({'stage':a.stage,'complete':True,'already_complete':True}),flush=True);return
 tokenizer=mod.AutoTokenizer.from_pretrained(a.model,local_files_only=True);backend=mod.TransformersBackend(a.model,tokenizer)
 import torch
 batch=a.batch
 for attempt in range(1,4):
  try:mod.run_stage(a.stage,source,target,tokenizer,backend,batch)
  except torch.cuda.OutOfMemoryError:
   if batch<=1:raise
   batch=max(1,batch//2);print(json.dumps({'event':'oom_batch_reduced','batch':batch}),flush=True)
  gc.collect();torch.cuda.empty_cache()
  n=coverage();result={'stage':a.stage,'valid_unique':n,'total':len(source),'remaining':len(source)-n,'complete':n==len(source),'attempt':attempt,'batch':batch,'time':time.time()}
  (out/f'{a.stage}.stage_status.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
  if result['complete']:return
 raise SystemExit(3)
if __name__=='__main__':main()
