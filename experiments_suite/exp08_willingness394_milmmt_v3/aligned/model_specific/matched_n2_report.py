#!/usr/bin/env python3
"""Compare only identical n2 selection items and cells; keep unknown outcomes explicit."""
import argparse,collections,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_io import load_run,read_rows
from grid_contract import digest

def validate_jobs(jobs,contract):
 if {j['cell'] for j in jobs}!=set(contract['cells']):return False
 expected=set(contract['selection_item_ids'])
 for cell in contract['cells']:
  rows=[r for r in jobs if r['cell']==cell]
  if len(rows)!=len(expected) or {r['id'] for r in rows}!=expected or any(r['frame']!='plain' for r in rows):raise ValueError('Mismatched cell/item/frame cohort '+cell)
 return True

def main():
 p=argparse.ArgumentParser();p.add_argument('--contract',type=Path,required=True);p.add_argument('--sources',type=Path,nargs='+',required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();contract=json.loads(a.contract.read_text());sources={};excluded=[]
 for root in a.sources:
  for folder in sorted(root.glob('models/*/selection')):
   if not (folder/'manifest.json').exists():continue
   manifest,jobs,rows=load_run(folder)
   if not validate_jobs(jobs,contract):excluded.append(str(folder));continue
   config=manifest['config']
   if config['sampling']!=contract['sampling'] or config['max_model_len']!=contract['max_model_len']:raise ValueError('Generation protocol mismatch')
   tag=folder.parent.name
   if tag not in contract['models']:raise ValueError('Unknown model')
   analysis=folder/'curve_analysis.json';quality=(len(rows),analysis.exists())
   if tag not in sources or quality>=sources[tag][0]:sources[tag]=(quality,folder,rows,json.loads(analysis.read_text()).get('observations',{}) if analysis.exists() else {})
 labels={r['uid']:r for r in read_rows(a.audit/'judgments.jsonl')};output=[]
 for tag in contract['models']:
  if tag not in sources:output.append(dict(model=tag,status='pending',expected_responses=720));continue
  _,folder,rows,observations=sources[tag];cells=[]
  for cell in contract['cells']:
   subset=[r for r in rows if r['cell']==cell];counts=collections.Counter();R=collections.Counter();F=collections.Counter()
   for row in subset:
    uid=digest({k:row[k] for k in ['model','revision','id','cell','frame','prompt','response_sha256']});label=labels.get(uid)
    counts[label['label'] if label and label.get('valid') else ('invalid_judge' if label else 'pending')]+=1
    obs=observations.get(cell,{}).get(row['id'],{});R[str(obs.get('R'))]+=1
    if obs.get('R') is True:F[str(obs.get('F'))]+=1
   cells.append(dict(cell=cell,expected_items=90,collected=len(subset),reconstruction_counts=dict(R),refusal_given_reconstruction_counts=dict(F),fulfillment_counts=dict(counts)))
  output.append(dict(model=tag,source=str(folder),expected_responses=720,collected=len(rows),cells=cells))
 result=dict(created_at=time.time(),scope='Primary matched comparison ONLY: same 90 items, eight n2 cells, selection split. Validation and n6 excluded.',contract_sha256=digest(contract),models=output,excluded_n6=excluded,unknown_policy='None, invalid_judge and pending are NOT task failures; report coverage explicitly.')
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'matched_models_with_inputs':len(sources),'excluded_sources':len(excluded),'output':str(a.out)}))
if __name__=='__main__':main()
