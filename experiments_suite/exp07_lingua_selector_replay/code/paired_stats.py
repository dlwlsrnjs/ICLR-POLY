#!/usr/bin/env python3
"""Paired statistics per model, for the paper-9 subset and the full 17, in two arm spaces."""
import json,random,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from run_lg_selector import features, gp_posterior, U0, BETA, SIGMA, CLEAR, G
BEN=Path('/home/ubuntu/342/jinkwon/poly/bucket_0913/L40S-only/experiments_suite/exp02_panel_collect/results/benign')
d=json.loads(Path('item_matrix.json').read_text())
tags,items,matrix=d['tags'],d['items'],d['matrix']
allarms=sorted(matrix[tags[0]])
priors={t:json.loads((BEN/f'{t}.json').read_text())['prior'] for t in tags}
PAPER9=sorted({'qwen25_3b_lg','qwen25_7b_lg','qwen25_14b_lg','qwen25_32b_lg','llama32_3b_it_lg',
 'llama31_8b_it_lg','gemma2_2b_it_lg','gemma2_9b_it_lg','gemma2_27b_lg'})
def space(n):
    grid=[a for a in allarms if G.match(a)]
    if n=='164': keep=grid
    else: keep=[a for a in grid if int(G.match(a).group(1))==3]
    return sorted(keep+[a for a in allarms if a in CLEAR])
def sel(mode,b,arms,F,calt,probe,rng):
    if mode=='random': q=rng.sample(range(len(arms)),min(b,len(arms)))
    else:
        mu0=U0*probe if mode=='ours' else np.zeros(len(arms)); q,ys=[],[]
        for _ in range(b):
            if q: m_,sd=gp_posterior(F[q],F,np.array(ys),mu0[q],mu0)
            else: m_,sd=mu0.copy(),np.full(len(arms),SIGMA)
            acq=m_+BETA*sd
            for i in q: acq[i]=-1e9
            nx=int(np.argmax(acq)); q.append(nx); ys.append(calt[arms[nx]])
    return max(q,key=lambda i:calt[arms[i]])
def signflip(dv):
    k=len(dv); masks=np.arange(1<<k,dtype=np.int64)[:,None]
    s=1-2*((masks>>np.arange(k))&1).astype(np.int8)
    st=(s*dv).mean(1); return float((np.abs(st)>=abs(dv.mean())-1e-12).mean())
for spname in ('164','44'):
    arms=space(spname); F=np.array([features(a) for a in arms])
    for subset,lab in ((PAPER9,'논문 9모델'),(tags,'전체 17모델')):
        per={k:{t:[] for t in subset} for k in ('fixed','ours3','ours8','rand3','aim','deep','oracle')}
        for s in range(30):
            rng=random.Random(1000+s); idx=list(range(len(items))); rng.shuffle(idx)
            calib,test=idx[:len(items)//2],idx[len(items)//2:]
            cal={t:{a:float(np.mean([matrix[t][a][i] for i in calib])) for a in arms} for t in subset}
            tst={t:{a:float(np.mean([matrix[t][a][i] for i in test])) for a in arms} for t in subset}
            for t in subset:
                others=[o for o in subset if o!=t]
                mn={a:float(np.mean([cal[o][a] for o in others])) for a in arms}
                per['fixed'][t].append(tst[t][max(mn,key=mn.get)])
                per['oracle'][t].append(max(tst[t].values()))
                per['aim'][t].append(tst[t]['m_aim']); per['deep'][t].append(tst[t]['m_deepinception'])
                pr=np.array([priors[t].get(a,0.0) for a in arms],float)
                per['ours3'][t].append(tst[t][arms[sel('ours',3,arms,F,cal[t],pr,random.Random(s))]])
                per['ours8'][t].append(tst[t][arms[sel('ours',8,arms,F,cal[t],pr,random.Random(s))]])
                per['rand3'][t].append(tst[t][arms[sel('random',3,arms,F,cal[t],pr,random.Random(s*7+1))]])
        M={k:{t:float(np.mean(v)) for t,v in per[k].items()} for k in per}
        print(f"\n### 공간 {spname} arm, {lab} (n={len(subset)})")
        print("  "+"  ".join(f"{k} {np.mean(list(M[k].values())):.3f}" for k in ('fixed','ours3','ours8','rand3','aim','deep','oracle')))
        for a,b in (('ours3','fixed'),('ours8','fixed'),('ours3','rand3'),('ours8','deep')):
            dv=np.array([M[a][t]-M[b][t] for t in subset])
            rng=np.random.default_rng(0); bs=dv[rng.integers(0,len(dv),size=(20000,len(dv)))].mean(1)
            print(f"  {a:6s} vs {b:6s} delta {dv.mean():+.3f} CI [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}] wins {int((dv>0).sum())}/{len(dv)} p={signflip(dv):.4f}")
