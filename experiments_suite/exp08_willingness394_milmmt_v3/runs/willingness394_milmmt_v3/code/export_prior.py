import argparse,json,csv,math
from collections import Counter
from scenarios import R,load,read,digest
p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();c,items,_,_=load(a.config);folder=R/'outputs'/c['name']
jobs={x['key']:x for x in read(folder/'jobs.jsonl')};rows={x['key']:x for f in folder.glob('responses.shard*.jsonl') for x in read(f)}
wg={x['key']:x for x in read(folder/'judge/outputs/wildguard.jsonl')};behavior={x['key']:x for x in read(folder/'judge/outputs/behavior.jsonl')}
assert set(rows)==set(jobs)==set(wg)==set(behavior),'incomplete collection or judgment coverage'
summary=json.loads((folder/'prior_observations.json').read_text());groups={}
with (folder/'labeled_responses.jsonl').open('w') as f:
 for key,x in rows.items():
  assert x['prompt']==jobs[key]['prompt'] and x['system']==jobs[key]['system']
  f.write(json.dumps(x|{'wildguard':wg[key],'behavior_candidate':behavior[key],'human_gold':False},ensure_ascii=False)+'\n')
  groups.setdefault(x['arm_id'],[]).append(x)
for arm in summary['arms']:
 rs=groups.get(arm['arm_id'],[]);valid=[wg[x['key']] for x in rs if wg[x['key']].get('valid')]
 arm['guard_invalid_or_missing']=len(rs)-len(valid)
 arm['behavior_invalid_or_missing']=len(rs)-sum(bool(behavior[x['key']].get('valid')) for x in rs)
 arm['eligible_items']=sum(x['arm_id']==arm['arm_id'] for x in jobs.values())
 arm['candidate_items']=len(items)
 arm['collection_fraction_of_eligible']=len(rs)/arm['eligible_items'] if arm['eligible_items'] else None
 for field in ['response_refusal','response_harmfulness']:
  values=[x['fields'][field]['p_yes'] for x in valid]
  assert all(math.isfinite(v) and 0<=v<=1 for v in values)
  arm['mean_uncalibrated_p_'+field]=sum(values)/len(values) if values else None
summary.update(target_revision=c['target_revision'],translation_model=c['translation_model'],translation_revision=c['translation_revision'],config_sha256=digest(c),probabilities_calibrated=False,automatic_behavior_labels_are_candidates=True)
(folder/'prior_observations.json').write_text(json.dumps(summary,indent=2))
with (folder/'prior_observations.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(summary['arms'][0]));w.writeheader()
 for x in summary['arms']:w.writerow(x|{'behavior_candidate_counts':json.dumps(x['behavior_candidate_counts'])})
report={'collected':len(rows),'eligible_jobs':len(jobs),'arms':len(summary['arms']),'guard_valid':sum(bool(x.get('valid')) for x in wg.values()),'behavior_valid':sum(bool(x.get('valid')) for x in behavior.values()),'finish_reasons':dict(Counter(x['finish_reason'] for x in rows.values())),'human_gold':False,'all_eligible_responses_collected':True,'all_judgment_calls_recorded':True}
(folder/'completion_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
