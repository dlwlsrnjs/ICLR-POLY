#!/usr/bin/env python3
"""Run each model's paired 331 probe, then its five-frame prior if confirmed."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from analyze import write
from transition331 import prepare,analyze_model,heldout_summary
from continue_collection import alive

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--bank-run',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--aux-cache',type=Path,required=True);a=p.parse_args()
 root=Path(__file__).resolve().parent;run=a.run;bank=a.bank_run;run.mkdir(parents=True,exist_ok=True)
 lock=(run/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def command(stage,args):
  write(run/'status.json',dict(stage=stage,command=list(map(str,args))))
  with (run/(stage+'.log')).open('a') as log:
   subprocess.run([sys.executable]+list(map(str,args)),env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',VLLM_USE_V1='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
 qwen7=a.cache/'models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28';wg=a.aux_cache/'models--allenai--wildguard/snapshots/cbba4823f3e8020e5a74a5e29bf85072def6f2ff'
 try:
  qa=json.loads((bank/'qa_extra_process.json').read_text());write(run/'status.json',dict(stage='waiting_translation_qa',pid=qa['pid']))
  while alive(qa['pid'],'qa_languages.py'):time.sleep(10)
  if not (bank/'extra_qa/qa/summary.json').exists():raise RuntimeError('Translation QA did not complete; see bank-run/qa_extra.log')
  command('export_translation_bank',[root/'export_bank.py','--run',bank])
  prepare(root,run,a.repo,bank)
  plan=json.loads((run/'TRANSITION_PLAN.json').read_text());out=[]
  for m in sorted(plan['models'],key=lambda m:(m['model']!='qwen25_7b',m['model'])):
   tag=m['model'];probe=run/'probes'/tag;cfg=json.loads((probe/'manifest.json').read_text())['config'];model=a.cache/('models--'+cfg['target_model'].replace('/','--'))/'snapshots'/cfg['target_revision']
   try:
    command(tag+'_probe_collect',[root.parent/'collect.py','--run',probe,'--gpu','0','--model-path',model,'--memory-utilization','.90','--batch-size','16'])
    command(tag+'_probe_reconstruction',[root/'judge_local.py','reconstruction','--run',probe,'--repo',a.repo,'--model-path',qwen7,'--gpu','0'])
    command(tag+'_probe_refusal',[root/'judge_local.py','wildguard','--run',probe,'--repo',a.repo,'--model-path',wg,'--gpu','0'])
    result=analyze_model(root,run,tag);out.append(result);write(run/'model_results.json',out)
    if result['status']!='confirmed_transition_refusal':continue
    dest=run/'willingness'/tag
    command(tag+'_willingness_prepare',[root.parent/'prepare.py','--config',run/'frozen_configs'/f'{tag}.json','--items',run/'inputs/items.json','--translations',run/'inputs/accepted_translations.json','--repo',a.repo,'--out',dest])
    command(tag+'_willingness_collect',[root.parent/'collect.py','--run',dest,'--gpu','0','--model-path',model,'--memory-utilization','.90','--batch-size','16'])
    command(tag+'_willingness_reconstruction',[root/'judge_local.py','reconstruction','--run',dest,'--repo',a.repo,'--model-path',qwen7,'--gpu','0'])
    command(tag+'_willingness_refusal',[root/'judge_local.py','wildguard','--run',dest,'--repo',a.repo,'--model-path',wg,'--gpu','0'])
    command(tag+'_willingness_summary',[root.parent/'summarize.py','--run',dest]);heldout_summary(dest)
    result['willingness_status']=json.loads((dest/'willingness_prior.json').read_text())['status'];result['validation_willingness_status']=json.loads((dest/'willingness_prior_validation.json').read_text())['status'];write(run/'model_results.json',out)
   except Exception as e:
    out.append(dict(model=tag,status='execution_failed',error=str(e)));write(run/'model_results.json',out)
  command('panel_summary',[root/'panel_summary.py','--run',run])
  write(run/'status.json',dict(stage='finished',models_evaluated=len({x['model'] for x in out}),confirmed=sum(x.get('status')=='confirmed_transition_refusal' for x in out),execution_failures=sum(x.get('status')=='execution_failed' for x in out),note='Models without confirmed transitions do not receive fabricated anchors. See model_results.json.'))
 except Exception as e:
  write(run/'status.json',dict(stage='failed',error=str(e)));raise
if __name__=='__main__':main()
