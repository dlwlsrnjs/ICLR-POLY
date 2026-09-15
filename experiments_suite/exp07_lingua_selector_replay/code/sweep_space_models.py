#!/usr/bin/env python3
"""Same Lingua replay, swept over arm-space size and model subset."""
import json,re,random,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from run_lg_selector import features, gp_posterior, U0, BETA, SIGMA, CLEAR, G

BEN=Path('/home/ubuntu/342/jinkwon/poly/bucket_0913/L40S-only/experiments_suite/exp02_panel_collect/results/benign')
d=json.loads(Path('item_matrix.json').read_text())
tags,items,matrix=d['tags'],d['items'],d['matrix']
allarms=sorted(matrix[tags[0]])
priors={t:json.loads((BEN/f'{t}.json').read_text())['prior'] for t in tags}
PAPER9={'qwen25_3b_lg','qwen25_7b_lg','qwen25_14b_lg','qwen25_32b_lg','llama32_3b_it_lg',
        'llama31_8b_it_lg','gemma2_2b_it_lg','gemma2_9b_it_lg','gemma2_27b_lg'}
def space(name):
    grid=[a for a in allarms if G.match(a)]
    if name=='164': keep=grid
    elif name=='44': keep=[a for a in grid if int(G.match(a).group(1))==3]
    elif name=='24': keep=[a for a in grid if int(G.match(a).group(1))==3 and G.match(a).group(2)=='ordered']
    return sorted(keep+[a for a in allarms if a in CLEAR])
def select(mode,budget,arms,F,cal,probe,rng):
    if mode=='random':
        q=rng.sample(range(len(arms)),min(budget,len(arms)))
    else:
        mu0=U0*probe if mode=='ours' else np.zeros(len(arms))
        q,ys=[],[]
        for _ in range(budget):
            if q: mean,sd=gp_posterior(F[q],F,np.array(ys),mu0[q],mu0)
            else: mean,sd=mu0.copy(),np.full(len(arms),SIGMA)
            acq=mean+BETA*sd
            for i in q: acq[i]=-1e9
            nxt=int(np.argmax(acq)); q.append(nxt); ys.append(cal[arms[nxt]])
    return max(q,key=lambda i:cal[arms[i]])
print(f"{'공간':>5} {'모델':>6} {'arm':>4} {'fixed':>7} {'ours@3':>7} {'ours@8':>7} {'rand@3':>7} {'oracle':>7} {'ours3-fixed':>12}")
for sp in ('164','44','24'):
    arms=space(sp); F=np.array([features(a) for a in arms])
    for subset,label in ((tags,'17'),(sorted(PAPER9),'논문9')):
        acc={k:[] for k in ('fixed','ours3','ours8','rand3','oracle')}
        for s in range(30):
            rng=random.Random(1000+s); idx=list(range(len(items))); rng.shuffle(idx)
            calib,test=idx[:len(items)//2],idx[len(items)//2:]
            cal={t:{a:float(np.mean([matrix[t][a][i] for i in calib])) for a in arms} for t in subset}
            tst={t:{a:float(np.mean([matrix[t][a][i] for i in test])) for a in arms} for t in subset}
            for t in subset:
                others=[o for o in subset if o!=t]
                mean={a:float(np.mean([cal[o][a] for o in others])) for a in arms}
                acc['fixed'].append(tst[t][max(mean,key=mean.get)])
                acc['oracle'].append(max(tst[t].values()))
                probe=np.array([priors[t].get(a,0.0) for a in arms],float)
                for b,key in ((3,'ours3'),(8,'ours8')):
                    acc[key].append(tst[t][arms[select('ours',b,arms,F,cal[t],probe,random.Random(s*97+b))]])
                acc['rand3'].append(tst[t][arms[select('random',3,arms,F,cal[t],probe,random.Random(s*31+7))]])
        m={k:float(np.mean(v)) for k,v in acc.items()}
        print(f"{sp:>5} {label:>6} {len(arms):4d} {m['fixed']:7.3f} {m['ours3']:7.3f} {m['ours8']:7.3f} {m['rand3']:7.3f} {m['oracle']:7.3f} {m['ours3']-m['fixed']:+12.3f}")
