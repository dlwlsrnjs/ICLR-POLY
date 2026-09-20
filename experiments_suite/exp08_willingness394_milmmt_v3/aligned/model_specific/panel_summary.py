#!/usr/bin/env python3
"""Descriptive five-frame covariance; never an anchor selection objective."""
import argparse,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from grid_contract import FRAMES
from run_io import load_run,indexed,read_rows,answer_section
from analyze import write

def matrices(rows):
 n=len(rows)
 if n<2:return dict(n=n,covariance=None,correlation=None)
 means=[sum(x[j] for x in rows)/n for j in range(5)]
 cov=[[sum((x[i]-means[i])*(x[j]-means[j]) for x in rows)/(n-1) for j in range(5)] for i in range(5)]
 corr=[[cov[i][j]/math.sqrt(cov[i][i]*cov[j][j]) if cov[i][i]*cov[j][j]>0 else None for j in range(5)] for i in range(5)]
 return dict(n=n,means=means,covariance=cov,correlation=corr)

def summarize(run):
 models={};vectors=[]
 for path in sorted((run/'willingness').glob('*/willingness_prior.json')):
  s=json.loads(path.read_text());tag=path.parent.name
  if s['status']!='ready':models[tag]=dict(status=s['status']);continue
  _,jobs,responses=load_run(path.parent,complete=True);r=indexed(responses)
  recon=indexed(read_rows(path.parent/'judge/reconstruction/restricted_reconstruction_audit.jsonl'));guard=indexed(read_rows(path.parent/'judge/outputs/wildguard.jsonl'))
  itemframes={}
  for job in jobs:itemframes.setdefault(job['id'],{})[job['frame']]=job['key']
  common=[];binary=[]
  for item,frames in itemframes.items():
   if set(frames)!=set(FRAMES):raise ValueError('Incomplete frame set')
   keys=[frames[f] for f in FRAMES]
   def valid(k):
    field=guard[k].get('fields',{}).get('response_refusal',{})
    return (r[k]['finish_reason']=='stop' and bool(answer_section(r[k]['response'])) and recon[k]['reconstruction_parse_valid']==1 and recon[k]['semantic_reconstruction_equivalent']==1 and guard[k].get('valid') is True and guard[k].get('judge_finish_reason')=='stop' and field.get('valid') is True and field.get('label') in ('yes','no'))
   if all(valid(k) for k in keys):
    common.append(item);binary.append([int(guard[k]['fields']['response_refusal']['label']=='no') for k in keys])
  stats=matrices(binary)
  if len(common)!=s['common_items'] or any(abs(x-y)>1e-12 for x,y in zip(stats['means'],s['willingness_vector'])):raise ValueError('Summary cohort mismatch')
  models[tag]=dict(status='ready',cell=s['fixed_cell'],common_item_ids=common,item_frame_covariance=stats)
  vectors.append(s['willingness_vector'])
 write(run/'panel_summary.json',dict(frames=FRAMES,models=models,ready_models=len(vectors),expected_models=17,
  across_model_mean_vectors=matrices(vectors),interpretation=[
   'Descriptive only: anchors/languages and reconstructed item cohorts can differ across models.',
   'Item-level binary covariance and across-model mean-vector covariance are different estimands.',
   'A correlation around 0.6 is possible; population covariance for Bernoulli indicators is bounded in magnitude by 0.25.',
   'Zero-variance correlations are null, not zero. No correlation target is used to tune collection.',
   'Nonrefusal does not establish useful fulfillment; these labels are model judgments, not human gold.']))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();summarize(a.run)
