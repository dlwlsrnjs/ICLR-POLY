#!/usr/bin/env python3
"""Audit existing benign priors; ASR is diagnostic and never enters selection."""
import argparse, csv, hashlib, json, math, re
from pathlib import Path

BASE='g3_ordered_n2'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def complexity(c):
 g,o,n=re.fullmatch(r'g(3|5|8|12)_(ordered|shuffled)_n(2|4|6|8)',c).groups()
 return int(g)*int(n), o!='ordered',int(n),int(g),c

def wilson(k,n):
 z=1.959963984540054;p=k/n;den=1+z*z/n
 a=(p+z*z/(2*n))/den;b=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [a-b,a+b]

def audit(root):
 inp=root/'inputs';flo=[json.loads(l) for l in (inp/'flores100.jsonl').read_text().splitlines()];models=sorted({r['model_tag'] for r in flo})
 assert len(models)==17 and len(flo)==544
 out=[];asr=[]
 for model in models:
  f={r['condition'].replace('frag','g',1):r for r in flo if r['model_tag']==model}
  assert len(f)==32 and all(r['n']==100 and r['successes']/100==r['reconstruction_rate'] for r in f.values())
  b={ds:json.loads((inp/f'{model}_{ds}.json').read_text()) for ds in ['mj','lg']}
  for ds,r in b.items():
   assert r['tag']==model+'_'+ds and r['harmful_content'] is False and set(r['benign_recon_by_cell'])==set(f)
  counts={ds:{c:round(v*b[ds]['n_probe_puzzles']) for c,v in b[ds]['benign_recon_by_cell'].items()} for ds in b}
  ns={ds:b[ds]['n_probe_puzzles'] for ds in b}
  def rank(c):
   lowers=[wilson(counts[ds][c],ns[ds])[0] for ds in b]
   return -min(lowers),complexity(c) # simple tie-break avoids maximizing another noisy statistic
  winner=min(f,key=rank)
  bestf=min(f,key=lambda c:(-f[c]['successes'],complexity(c)))
  # Exactly one probe's deficit is a sensitivity analysis, not proof of equivalence.
  near=[c for c in f if all(counts[ds][c]>=counts[ds][winner]-1 for ds in b)]
  stable=min(near,key=complexity)
  shortlist=sorted({BASE,winner,bestf},key=complexity)
  out.append(dict(model=model,historical_minimax_candidate=winner,one_probe_tolerance_candidate=stable,
    flores_best_candidate=bestf,calibration_candidates=shortlist,
    base_flores_rate=f[BASE]['reconstruction_rate'],best_flores_rate=f[bestf]['reconstruction_rate'],
    base_mj_rate=counts['mj'][BASE]/ns['mj'],base_lg_rate=counts['lg'][BASE]/ns['lg'],
    historical_candidate_evidence={ds:dict(k=counts[ds][winner],n=ns[ds],wilson95=wilson(counts[ds][winner],ns[ds])) for ds in b},
    status='candidate_only_requires_profile_matched_calibration'))
  for ds in b:
   matrix=json.loads((inp/f'{ds}_plain_asr.json').read_text());arms=matrix['matrix'][model+'_'+ds]
   for cell in f:
    xs=arms[cell+'__plain'];assert len(xs)==len(matrix['items']) and all(x in (0,1) for x in xs)
    asr.append(dict(model=model,dataset=ds,cell=cell,n=len(xs),verified_successes=sum(xs),verified_asr=sum(xs)/len(xs),
      benign_reconstruction=counts[ds][cell]/ns[ds],selected_historical_candidate=cell==winner))
 sources=json.loads((inp/'provenance.json').read_text())
 for name,h in sources['bundled_sha256'].items(): assert sha(inp/name)==h,name
 write(root/'analysis.json',dict(selection_uses_asr=False,selection_rule='Maximize worst MJ/LG pointwise Wilson lower bound; tie: fewer fragments, ordered, fewer languages.',
  models=out,source_provenance=sources,limitations=[
   'MJ and LG use different language prefixes; historical n2 is English+Bengali / English+Norwegian, not English+Arabic.',
   'FLORES100 uses English+Chinese n2 and another prompt wrapper; it is supporting evidence only.',
   'MJ/LG 24-item priors overlap FLORES100: they are not independent replications.',
   'Pointwise confidence intervals do not adjust for selecting among 32 cells.',
   'One-probe tolerance is a sensitivity analysis, not a noninferiority test.',
   'ASR matrices contain reconstruction-gated success; they cannot alone distinguish refusal from reconstruction failure.',
   'A 331-item over-refusal bank measures benign nonrefusal, not harmful attack success.']))
 with (root/'asr_diagnostic.csv').open('w') as h:
  w=csv.DictWriter(h,fieldnames=list(asr[0]));w.writeheader();w.writerows(asr)
 with (root/'model_candidates.csv').open('w') as h:
  keys=['model','historical_minimax_candidate','one_probe_tolerance_candidate','flores_best_candidate','base_flores_rate','best_flores_rate','base_mj_rate','base_lg_rate']
  w=csv.DictWriter(h,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(out)
 comparisons=[]
 for ds in ['mj','lg']:
  r=[r for r in asr if r['dataset']==ds];lookup={(x['model'],x['cell']):x['verified_asr'] for x in r}
  comparisons.append(dict(dataset=ds,comparison='g3_ordered_n2 versus g12_shuffled_n8, plain frame only',
   lower=sum(lookup[m,BASE]<lookup[m,'g12_shuffled_n8'] for m in models),equal=sum(lookup[m,BASE]==lookup[m,'g12_shuffled_n8'] for m in models),higher=sum(lookup[m,BASE]>lookup[m,'g12_shuffled_n8'] for m in models)))
 write(root/'asr_diagnostic_summary.json',dict(comparisons=comparisons,selection_uses_asr=False,causal_claim=False))
 print(json.dumps(dict(models=len(out),shortlist_cells=sum(len(x['calibration_candidates']) for x in out),asr_comparisons=comparisons)))
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args();audit(a.root)
