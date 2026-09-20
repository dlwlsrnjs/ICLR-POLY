#!/usr/bin/env python3
"""Continue translation QA, then collect and annotate the frozen five-frame priors."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from analyze import write

def alive(pid,marker):
 try:return marker in Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0',' ')
 except (FileNotFoundError,ProcessLookupError):return False

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--cache',type=Path,default=Path('/home/ubuntu/342/jinkwon/hf_cache/hub'));p.add_argument('--aux-cache',type=Path,required=True);a=p.parse_args()
 root=Path(__file__).resolve().parent;run=a.run;repo=a.repo;py=sys.executable
 lock=(run/'continuation.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 processes=json.loads((run/'processes.json').read_text());qwen32=a.cache/'models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd';qwen7=a.cache/'models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28'
 def command(stage,args,gpu='1',v1='0'):
  write(run/'continuation_status.json',dict(stage=stage,command=list(map(str,args))))
  with (run/(stage+'.log')).open('a') as log:
   subprocess.run([py]+list(map(str,args)),env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,VLLM_USE_V1=v1,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
 try:
  write(run/'continuation_status.json',dict(stage='waiting_existing_translation_qa'))
  while alive(processes['translation_qa']['pid'],'qa_existing.py'):time.sleep(10)
  # Resume safely also handles postprocessing failure after all judgments were saved.
  if not (run/'qa/summary.json').exists():
   command('qa_existing_resume',[root/'qa_existing.py','--run',run,'--source',root.parent/'shared_en_ar/data/translations.jsonl','--model',qwen32,'--gpu','1'])
  required=['Chinese','Bengali','Thai','Korean']
  accepted=json.loads((run/'qa/accepted_translations.json').read_text())
  if required:
   model=a.aux_cache/'models--xiaomi-research--MiLMMT-46-12B-v1.0/snapshots/a27dbbb37142ff076990820a1c9f0827beb5d6ea'
   command('translate_extra',[root/'translate_extra.py','--run',run/'extra_translations','--items',run/'inputs/items.json','--model',model,'--languages',*required,'--gpu','1'],v1='1')
   extra=run/'extra_qa';(extra/'inputs').mkdir(parents=True,exist_ok=True);(extra/'inputs/items.json').write_bytes((run/'inputs/items.json').read_bytes())
   write(run/'continuation_status.json',dict(stage='waiting_calibration_before_gpu0_qa'))
   while alive(processes['calibration']['pid'],'calibrate.py'):time.sleep(10)
   command('qa_extra',[root/'qa_languages.py','--run',extra,'--source',run/'extra_translations/translations.jsonl','--model',qwen32,'--gpu','0'],gpu='0')
   accepted += [x for x in json.loads((extra/'qa/accepted_translations.json').read_text()) if x['language']!='English']
  write(run/'accepted_translations.json',accepted)
  # Preserve all generated translations, including failed QA, for all 331 items.
  source_paths=[root.parent/'shared_en_ar/data/translations.jsonl',run/'extra_translations/translations.jsonl']
  bank=[json.loads(line) for path in source_paths for line in path.read_text().splitlines() if line.strip()]
  items=json.loads((run/'inputs/items.json').read_text());originals={x['id']:x['prompt'] for x in items}
  expected={(item,lang) for item in originals for lang in ['Arabic','Norwegian','Finnish','Chinese','Bengali','Thai','Korean']}
  if len(bank)!=2317 or {(x['item_id'],x['language']) for x in bank}!=expected:raise RuntimeError('Full 331 x 7 bank coverage mismatch')
  if any(x['english_original']!=originals[x['item_id']] for x in bank):raise RuntimeError('English original changed')
  (run/'translations_7languages.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in bank))
  judgments=[json.loads(line) for path in [run/'qa/judgments.jsonl',run/'extra_qa/qa/judgments.jsonl'] for line in path.read_text().splitlines()]
  failed=[dict(item_id=x['item_id'],language=x['source']['language'],source_sha256=x['source_sha256'],reason=x.get('parsed',{}).get('reason')) for x in judgments if not x['pass_qa']]
  write(run/'retranslation_required.json',failed)
  write(run/'translation_bank_summary.json',dict(items=331,languages_including_english=8,foreign_pairs=len(bank),english_unchanged=True,qa_pass=sum(x['pass_qa'] for x in judgments),qa_fail=len(failed),failed_pairs_preserved=True))
  write(run/'continuation_status.json',dict(stage='waiting_frozen_anchors'))
  while alive(processes['calibration']['pid'],'calibrate.py'):time.sleep(10)
  if not (run/'anchor_decisions.json').exists():raise RuntimeError('Calibration did not produce frozen anchors; inspect calibration.log')
  decisions=json.loads((run/'anchor_decisions.json').read_text())['models']
  frozen=[x for x in decisions if x['status']=='frozen']
  failures=[dict(model=x['model'],stage='anchor_not_frozen') for x in decisions if x['status']!='frozen'];completed=[]
  for x in frozen:
   tag=x['model'];cfg=run/'frozen_configs'/f'{tag}.json';c=json.loads(cfg.read_text());dest=run/'willingness'/tag
   try:
    command(tag+'_prepare',[root.parent/'prepare.py','--config',cfg,'--items',run/'inputs/items.json','--translations',run/'accepted_translations.json','--repo',repo,'--out',dest],gpu='0')
    manifest=json.loads((dest/'manifest.json').read_text())
    if manifest['eligible_items']<c['minimum_common_items']:raise RuntimeError('Too few translation-QA-accepted items')
    model=a.cache/('models--'+c['target_model'].replace('/','--'))/'snapshots'/c['target_revision']
    command(tag+'_collect',[root.parent/'collect.py','--run',dest,'--gpu','0','--model-path',model,'--memory-utilization','.90','--batch-size','16'],gpu='0')
    command(tag+'_reconstruction',[root/'judge_local.py','reconstruction','--run',dest,'--repo',repo,'--model-path',qwen7,'--gpu','0'],gpu='0')
    wildguard=a.aux_cache/'models--allenai--wildguard/snapshots/cbba4823f3e8020e5a74a5e29bf85072def6f2ff'
    command(tag+'_wildguard',[root/'judge_local.py','wildguard','--run',dest,'--repo',repo,'--model-path',wildguard,'--gpu','0'],gpu='0',v1='1')
    command(tag+'_summarize',[root.parent/'summarize.py','--run',dest],gpu='0')
    summary=json.loads((dest/'willingness_prior.json').read_text())
    if summary['status']!='ready':raise RuntimeError('Collected but insufficient common reconstructed items')
    completed.append(tag)
   except Exception as e:
    failures.append(dict(model=tag,stage='willingness',error=str(e)))
   write(run/'collection_progress.json',dict(completed=completed,failures=failures,total_models=17))
  command('panel_summary',[root/'panel_summary.py','--run',run],gpu='0')
  write(run/'continuation_status.json',dict(stage='finished' if not failures else 'finished_with_failures',completed=completed,failures=failures))
 except Exception as e:
  write(run/'continuation_status.json',dict(stage='failed',error=str(e)));raise
if __name__=='__main__':main()
