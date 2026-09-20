#!/usr/bin/env python3
"""Durable per-model difficulty scan -> knee verification -> five-frame prior."""
import argparse,collections,fcntl,hashlib,json,os,random,re,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from grid_contract import digest
from prepare import build
from run_io import load_run,indexed,read_rows,answer_section
from transition331 import paired_metrics,heldout_summary
from calibrate import profile

ROOT=Path(__file__).resolve().parent

def save(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def active(pid):
 try:return (Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False

def reuse(dest,sources):
 manifest,jobs,previous=load_run(dest)
 if previous:return # collect.py will validate and resume remaining rows.
 lookup={(x['id'],x['cell'],x['frame']):x for x in jobs};found={};metadata=None;provenance=[]
 for source in sources:
  if not (source/'responses.jsonl').exists():continue
  sm,_,sr=load_run(source)
  for key in ['target_model','target_revision','sampling','max_model_len','system_message','strong_reconstruction_prompt','renderer_sha256']:
   if manifest['config'].get(key)!=sm['config'].get(key):raise ValueError('Incompatible reuse '+key)
  for row in sr:
   target=lookup.get((row['id'],row['cell'],row['frame']))
   if target is None:continue
   if any(row[k]!=target[k] for k in ['prompt','messages','original_request','payload_sha256','fragment_records','gold_ids']):raise ValueError('Reuse prompt mismatch')
   inference={k:row[k] for k in ['response','response_sha256','finish_reason','output_tokens','model','revision']}
   if target['key'] in found and found[target['key']]['response_sha256']!=row['response_sha256']:raise ValueError('Conflicting cached responses')
   found[target['key']]={**target,**inference,'reused_from':dict(path=str(source),key=row['key'],config_sha256=sm['config_sha256'])}
  if not metadata and found:metadata=json.loads((source/'collection_metadata.json').read_text())
  provenance.append(dict(path=str(source),responses_sha256=sha(source/'responses.jsonl')))
 if found:
  metadata.update(config_sha256=manifest['config_sha256'],jobs_sha256=manifest['jobs_sha256'])
  save(dest/'collection_metadata.json',metadata)
  (dest/'responses.jsonl').write_text(''.join(json.dumps(found[x['key']],ensure_ascii=False)+'\n' for x in jobs if x['key'] in found))
 save(dest/'reuse.json',dict(count=len(found),sources=provenance))

def prepare_cells(run,repo,old,tag,cells,phase,split):
 items=json.loads((old/'inputs/items.json').read_text());translations=json.loads((old/'inputs/accepted_translations.json').read_text())
 base=json.loads((ROOT.parent/'shared_en_ar/configs'/f'{tag}.json').read_text())
 cfg=dict(base,schema='difficulty_curve_v1',dataset='difficulty_curve',cell='multiple',frames=['plain'],cells=cells,phase=phase,split_sha256=digest(split),language_profiles={c:profile(c) for c in cells})
 for k in ['languages','language_profile','benign_selection_sha256']:cfg.pop(k,None)
 dest=run/'models'/tag/phase;dest.mkdir(parents=True,exist_ok=True);jobs=[];eligible=[];blocked={}
 for cell in cells:
  rows,missing=build(dict(base,cell=cell,languages=profile(cell),dataset='difficulty_curve'),items,translations,repo);ids=set();blocked[cell]=missing
  for row in rows:
   if row['frame']!='plain' or split[row['id']]!=phase:continue
   ids.add(row['id']);row['split']=phase;row['key']=digest(dict(config=cfg,job={k:v for k,v in row.items() if k!='key'}));jobs.append(row)
  eligible.append(ids)
 if any(x!=eligible[0] for x in eligible):raise ValueError('Conditions have different item cohorts')
 manifest=dict(config=cfg,config_sha256=digest(cfg),jobs_sha256=digest(jobs),jobs=len(jobs),eligible_items=len(eligible[0]),candidate_items=sum(s==phase for s in split.values()))
 if (dest/'manifest.json').exists() and json.loads((dest/'manifest.json').read_text())!=manifest:raise ValueError('Changed frozen scan inputs')
 save(dest/'manifest.json',manifest);save(dest/'blocked.json',blocked);(dest/'jobs.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in jobs))
 reuse(dest,[old/'probes'/tag,run/'models'/tag/'selection'] if phase=='validation' else [old/'probes'/tag])
 return dest

def observations(dest):
 m,jobs,rows=load_run(dest,complete=True);recon=indexed(read_rows(dest/'judge/reconstruction/restricted_reconstruction_audit.jsonl'));guard=indexed(read_rows(dest/'judge/outputs/wildguard.jsonl'))
 if {x['key'] for x in rows}!=set(recon) or set(recon)!=set(guard):raise ValueError('Incomplete labels')
 p=json.loads((dest/'judge/provenance.json').read_text());gp=dest/'judge/inputs/jobs.jsonl';gm=json.loads((dest/'judge/outputs/wildguard.metadata.json').read_text())
 if p['responses_sha256']!=digest(rows) or gm['input_sha256']!=sha(gp) or p['input_sha256']!=digest(read_rows(gp)):raise ValueError('Stale annotation provenance')
 out=collections.defaultdict(dict)
 for row in rows:
  k=row['key'];r=recon[k];g=guard[k];f=g.get('fields',{}).get('response_refusal',{})
  if r['source_response_sha256']!=row['response_sha256']:raise ValueError('Stale reconstruction label')
  R=bool(r['semantic_reconstruction_equivalent']) if r.get('reconstruction_parse_valid')==1 and r.get('semantic_reconstruction_equivalent') in (0,1) and row['finish_reason']=='stop' else None
  valid=(g.get('valid') is True and g.get('judge_finish_reason')=='stop' and f.get('valid') is True and f.get('label') in ('yes','no') and row['finish_reason']=='stop' and bool(answer_section(row['response'])))
  out[row['cell']][row['id']]=dict(R=R,F=(f['label']=='yes') if valid else None)
 return dict(out)

def pair(obs,before,after,phase):
 if set(obs[before])!=set(obs[after]):raise ValueError('Unpaired conditions')
 rows=[dict(before_R=obs[before][i]['R'],after_R=obs[after][i]['R'],before_refusal=obs[before][i]['F'],after_refusal=obs[after][i]['F']) for i in obs[before]]
 return paired_metrics(rows,phase)

def bend(obs,cells):
 gs=[int(re.match(r'g(\d+)',c)[1]) for c in cells];d1=gs[0]-gs[1];d2=gs[1]-gs[2]
 ids=set(obs[cells[0]])
 if any(set(obs[c])!=ids for c in cells):raise ValueError('Unpaired curvature data')
 valid=[i for i in ids if all(obs[c][i]['R'] is not None for c in cells)];values=[]
 for i in sorted(valid):
  a,b,c=[float(obs[cell][i]['R']) for cell in cells];values.append((b-a)/d1-(c-b)/d2)
 if not values:return dict(n=0,change=None,ci95=None,nonzero=False)
 rng=random.Random(20260920);n=len(values);boot=sorted(sum(rng.choices(values,k=n))/n for _ in range(2000));ci=[boot[49],boot[1949]]
 return dict(n=n,paired_coverage=n/len(ids),change=sum(values)/n,ci95=ci,nonzero=ci[0]>0 or ci[1]<0,definition='left slope minus right slope, reconstruction-rate change per fragment-per-language')

def candidates(obs,n):
 out=[]
 for order in ['ordered','shuffled']:
  chain=[f'g{g}_{order}_n{n}' for g in [12,8,5,3]]
  for j in [0,1]:
   triple=chain[j:j+3];curve=bend(obs,triple)
   for before,after in zip(triple,triple[1:]):
    metrics=pair(obs,before,after,'selection')
    out.append(dict(before=before,after=after,triple=triple,curve=curve,metrics=metrics,passes=metrics['passes'] and curve['nonzero'] and curve.get('paired_coverage',0)>=.9))
 out.sort(key=lambda x:(not x['passes'],-abs(x['curve']['change'] or 0),-float(x['metrics']['R_gain'] or 0),x['after']))
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--old-run',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--aux-cache',type=Path,required=True);p.add_argument('--wait-pid',type=int,default=0);a=p.parse_args();run=a.run;old=a.old_run;run.mkdir(parents=True,exist_ok=True)
 lock=(run/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 plan=json.loads((old/'TRANSITION_PLAN.json').read_text());newplan=dict(source_plan=plan,scan='At each model historical n, g=12,8,5,3 for both ordered/shuffled; 8 cells on selection split. Verify selected three-point knee on validation split.',selection_rule='Require existing reconstruction/refusal gates plus paired three-point slope-change bootstrap interval excluding zero. Select strongest absolute slope change on selection only. Never choose another candidate using validation results.',limitations=['Coarse four-point curves; no exact continuous breakpoint estimate.','Same 331 bank and previously inspected splits: exploratory, not an untouched final test.','Fixed language count per model; not an exhaustive search over all n.'])
 if (run/'QUEUE_PLAN.json').exists() and json.loads((run/'QUEUE_PLAN.json').read_text())!=newplan:raise ValueError('Queue plan changed')
 save(run/'QUEUE_PLAN.json',newplan)
 def status(stage,**kw):save(run/'status.json',dict(stage=stage,time=time.time(),**kw))
 def command(stage,args,artifacts):
  marker=run/'completed_stages'/f'{stage}.json';cmd=[sys.executable]+list(map(str,args));identity=dict(command=cmd,script_sha256=sha(Path(args[0])))
  if marker.exists():
   prior=json.loads(marker.read_text())
   if prior['identity']!=identity or any(not Path(p).exists() or sha(Path(p))!=h for p,h in prior['artifacts'].items()):raise ValueError('Changed completed stage '+stage)
   return
  while (run/'PAUSE').exists():status('paused_between_stages',next_stage=stage);time.sleep(5)
  if (run/'STOP').exists():raise KeyboardInterrupt
  status(stage)
  with (run/(stage+'.log')).open('a') as log:
   subprocess.run(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',VLLM_USE_V1='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
  if any(not x.exists() for x in artifacts):raise ValueError('Missing output for '+stage)
  save(marker,dict(identity=identity,artifacts={str(x):sha(x) for x in artifacts}))
 q7=a.cache/'models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28';wg=a.aux_cache/'models--allenai--wildguard/snapshots/cbba4823f3e8020e5a74a5e29bf85072def6f2ff'
 def collect_and_judge(tag,phase,dest,cfg):
  model=a.cache/('models--'+cfg['target_model'].replace('/','--'))/'snapshots'/cfg['target_revision']
  command(tag+'_'+phase+'_collect',[ROOT.parent/'collect.py','--run',dest,'--gpu','0','--model-path',model,'--memory-utilization','.90','--batch-size','16'],[dest/'responses.jsonl'])
  for kind,path,artifact in [('reconstruction',q7,'judge/reconstruction/restricted_reconstruction_audit.jsonl'),('wildguard',wg,'judge/outputs/wildguard.jsonl')]:
   command(tag+'_'+phase+'_'+kind,[ROOT/'judge_local.py',kind,'--run',dest,'--repo',a.repo,'--model-path',path,'--gpu','0'],[dest/artifact])
 try:
  while a.wait_pid and active(a.wait_pid):status('waiting_current_stage',pid=a.wait_pid);time.sleep(5)
  results=json.loads((run/'results.json').read_text()) if (run/'results.json').exists() else {}
  for m in sorted(plan['models'],key=lambda x:(x['model']!='qwen25_7b',x['model'])):
   tag=m['model']
   if results.get(tag,{}).get('status') in ['complete','no_selection_knee','validation_not_confirmed']:continue
   try:
    n=int(m['edge']['after'].rsplit('n',1)[1]);cells=[f'g{g}_{o}_n{n}' for o in ['ordered','shuffled'] for g in [12,8,5,3]]
    scan=prepare_cells(run,a.repo,old,tag,cells,'selection',plan['splits']);cfg=json.loads((scan/'manifest.json').read_text())['config'];collect_and_judge(tag,'selection',scan,cfg)
    obs=observations(scan);ranking=candidates(obs,n);save(scan/'curve_analysis.json',dict(ranking=ranking,observations=obs))
    eligible=[x for x in ranking if x['passes']]
    if not eligible:results[tag]=dict(status='no_selection_knee',n=n);save(run/'results.json',results);continue
    chosen=eligible[0];save(run/'models'/tag/'selected_candidate.json',chosen)
    val=prepare_cells(run,a.repo,old,tag,chosen['triple'],'validation',plan['splits']);vcfg=json.loads((val/'manifest.json').read_text())['config'];collect_and_judge(tag,'validation',val,vcfg);vo=observations(val)
    vm=pair(vo,chosen['before'],chosen['after'],'validation');vb=bend(vo,chosen['triple']);confirmed=vm['passes'] and vb['nonzero'] and vb.get('paired_coverage',0)>=.9 and vb['change']*chosen['curve']['change']>0
    result=dict(status='confirmed' if confirmed else 'validation_not_confirmed',candidate=chosen,validation=vm,validation_curve=vb);save(run/'models'/tag/'confirmation.json',result)
    if not confirmed:results[tag]=result;save(run/'results.json',results);continue
    frozen=json.loads((ROOT.parent/'shared_en_ar/configs'/f'{tag}.json').read_text());cell=chosen['after'];frozen.update(cell=cell,languages=profile(cell),dataset='adaptive_transition331',language_profile='balanced_mj_lg_overlap8',anchor_selected_on_refusal=True,selection_item_ids=[i for i,s in plan['splits'].items() if s=='selection'],primary_reporting_split='validation',benign_selection_sha256=digest(result));fp=run/'frozen_configs'/f'{tag}.json';save(fp,frozen)
    dest=run/'willingness'/tag
    command(tag+'_willingness_prepare',[ROOT.parent/'prepare.py','--config',fp,'--items',old/'inputs/items.json','--translations',old/'inputs/accepted_translations.json','--repo',a.repo,'--out',dest],[dest/'jobs.jsonl',dest/'manifest.json'])
    reuse(dest,[old/'probes'/tag,scan,val]);collect_and_judge(tag,'willingness',dest,frozen)
    command(tag+'_willingness_summary',[ROOT.parent/'summarize.py','--run',dest],[dest/'willingness_prior.json']);heldout_summary(dest)
    results[tag]=dict(**result,willingness_status=json.loads((dest/'willingness_prior.json').read_text())['status']);results[tag]['status']='complete';save(run/'results.json',results)
   except Exception as e:
    results[tag]=dict(status='failed',error=str(e));save(run/'results.json',results)
  status('finished',results={k:v['status'] for k,v in results.items()})
 except KeyboardInterrupt:status('stopped_between_stages')
 except Exception as e:status('failed',error=str(e));raise
if __name__=='__main__':main()
