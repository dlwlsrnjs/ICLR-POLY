#!/usr/bin/env python3
"""Read-only audit of saved grid files; never renders prompts or responses."""
import argparse, collections, hashlib, itertools, json
from pathlib import Path
MODELS=['qwen25_14b','qwen25_32b','gemma2_27b','mistral24b','phi3_medium_14b']
ARMS=[f'g{g}_{o}_n{n}__{f}' for g,o,n,f in itertools.product([3,5,8,12],['ordered','shuffled'],[2,4,6,8],['plain','persona','fiction','pap','persona+fiction'])]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(x) for x in p.open() if x.strip()]
def audit(root,data):
 report={'schema':'mj-lg-checkpoint-audit-v1','expected_arms_per_pair':160,'pairs':{},'inputs':{},'files':{},'errors':[]}
 for ds,rel in [('lg','panel_v2/harm_grid.jsonl'),('mj','multijail_v1/harm_grid.jsonl')]:
  ip=data/rel; expected={str(r['item_id']) for r in rows(ip)}
  assert len(expected)==(250 if ds=='lg' else 315)
  report['inputs'][ds]={'path':rel,'sha256':sha(ip),'items':len(expected)}
  for model in MODELS:
   pair={'raw_complete_arms':[],'judged_current_arms':[],'judged_stale_arms':[],'raw_missing_arms':[],'judged_missing_arms':[],'raw_rows':0,'current_judged_rows':0,'finish_reasons':{},'missing_generation_metadata_rows':0,'null_R':0,'null_U':0}
   reasons=collections.Counter()
   for arm in ARMS:
    name=f'{model}_{ds}__{arm}.jsonl'; rp=root/'raw'/ds/name; jp=root/'judged'/ds/name
    valid_raw=False
    if rp.exists():
     try:
      rr=rows(rp); ids=[str(r['item_id']) for r in rr]
      assert len(ids)==len(expected) and set(ids)==expected
      assert all(all(k in r for k in ['raw_output','answer','reconstruction','prompt_sha256']) for r in rr)
      digest=sha(rp); valid_raw=True; pair['raw_complete_arms'].append(arm);pair['raw_rows']+=len(rr)
      reasons.update(str(r.get('generation_finish_reason','UNRECORDED')) for r in rr)
      pair['missing_generation_metadata_rows']+=sum('generation_finish_reason' not in r for r in rr)
      report['files'][str(rp.relative_to(root))]={'sha256':digest,'bytes':rp.stat().st_size,'rows':len(rr),'valid':True}
     except Exception as e:report['errors'].append({'path':str(rp.relative_to(root)),'error':type(e).__name__})
    if not valid_raw:pair['raw_missing_arms'].append(arm)
    current=False
    if jp.exists():
     try:
      jr=rows(jp); ids=[str(r['item_id']) for r in jr];mp=jp.with_suffix('.jsonl.meta.json'); meta=json.loads(mp.read_text())
      assert len(ids)==len(expected) and set(ids)==expected
      assert all(all(k in r for k in ['R','U','J','reconstruction_judge_raw','guard_judge_raw']) for r in jr)
      current=valid_raw and meta['source_sha256']==digest and meta['rows']==len(expected) and meta['recon_revision']=='a09a35458c702b33eeacc393d103063234e8bc28' and meta['guard_revision']=='4505cb1a6f1864f21f8b27f7daf1b9a1aab6edbb'
      report['files'][str(jp.relative_to(root))]={'sha256':sha(jp),'bytes':jp.stat().st_size,'rows':len(jr),'source_current':current}
      if current:
       pair['judged_current_arms'].append(arm);pair['current_judged_rows']+=len(jr)
       pair['null_R']+=sum(r['R'] is None for r in jr);pair['null_U']+=sum(r['U'] is None for r in jr)
      else:pair['judged_stale_arms'].append(arm)
     except Exception as e:report['errors'].append({'path':str(jp.relative_to(root)),'error':type(e).__name__})
    if not current:pair['judged_missing_arms'].append(arm)
   pair['finish_reasons']=dict(reasons)
   pair['truncation_retry_summary_present']=(root/'raw'/ds/f'{model}_{ds}__truncation_retry_summary.json').exists()
   report['pairs'][f'{model}/{ds}']=pair
 report['totals']={k:sum(p[k] for p in report['pairs'].values()) for k in ['raw_rows','current_judged_rows','null_R','null_U','missing_generation_metadata_rows']}
 report['totals'].update(raw_arms=sum(len(p['raw_complete_arms']) for p in report['pairs'].values()),judged_current_arms=sum(len(p['judged_current_arms']) for p in report['pairs'].values()),expected_arms=1600,expected_rows=452000)
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 r=audit(a.run_root,a.data_dir);a.output.write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['totals']))
 for k,v in r['pairs'].items():print(k,'raw',len(v['raw_complete_arms']),'judged_current',len(v['judged_current_arms']),'stale',len(v['judged_stale_arms']))
 print('errors',r['errors'][:10]);raise SystemExit(bool(r['errors']))
