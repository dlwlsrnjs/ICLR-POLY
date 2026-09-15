#!/usr/bin/env python3
"""Which arm wins as 'best fixed', and does the paper's difficulty-adaptive budget change things?"""
import json,random,sys,collections
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from run_lg_selector import features, gp_posterior, U0, BETA, SIGMA, CLEAR, G
BEN=Path('/home/ubuntu/342/jinkwon/poly/bucket_0913/L40S-only/experiments_suite/exp02_panel_collect/results/benign')
d=json.loads(Path('item_matrix.json').read_text())
tags,items,matrix=d['tags'],d['items'],d['matrix']
arms=sorted(matrix[tags[0]]); F=np.array([features(a) for a in arms])
priors={t:json.loads((BEN/f'{t}.json').read_text())['prior'] for t in tags}
fixedpick=collections.Counter(); ourspick=collections.Counter()
adapt=[]; fixed=[]; ours8=[]
BMIN,BMAX,TAU=3,8,0.5
for s in range(30):
    rng=random.Random(1000+s); idx=list(range(len(items))); rng.shuffle(idx)
    calib,test=idx[:len(items)//2],idx[len(items)//2:]
    cal={t:{a:float(np.mean([matrix[t][a][i] for i in calib])) for a in arms} for t in tags}
    tst={t:{a:float(np.mean([matrix[t][a][i] for i in test])) for a in arms} for t in tags}
    for t in tags:
        others=[o for o in tags if o!=t]
        mean={a:float(np.mean([cal[o][a] for o in others])) for a in arms}
        fa=max(mean,key=mean.get); fixedpick[fa]+=1; fixed.append(tst[t][fa])
        probe=np.array([priors[t].get(a,0.0) for a in arms],float)
        mu0=U0*probe; q,ys=[],[]
        while True:
            if q: m_,sd=gp_posterior(F[q],F,np.array(ys),mu0[q],mu0)
            else: m_,sd=mu0.copy(),np.full(len(arms),SIGMA)
            acq=m_+BETA*sd
            for i in q: acq[i]=-1e9
            nx=int(np.argmax(acq)); q.append(nx); ys.append(cal[t][arms[nx]])
            if len(q)>=BMIN and (max(ys)>=TAU or len(q)>=BMAX): break
        best=max(q,key=lambda i:cal[t][arms[i]]); adapt.append(tst[t][arms[best]]); ourspick[arms[best]]+=1
        q8,ys8=[],[]
        for _ in range(8):
            if q8: m_,sd=gp_posterior(F[q8],F,np.array(ys8),mu0[q8],mu0)
            else: m_,sd=mu0.copy(),np.full(len(arms),SIGMA)
            acq=m_+BETA*sd
            for i in q8: acq[i]=-1e9
            nx=int(np.argmax(acq)); q8.append(nx); ys8.append(cal[t][arms[nx]])
        ours8.append(tst[t][arms[max(q8,key=lambda i:cal[t][arms[i]])]])
print("best fixed 로 뽑힌 arm (상위):",fixedpick.most_common(5))
print("우리 selector 가 추천한 arm (상위):",ourspick.most_common(8))
print(f"\nfixed {np.mean(fixed):.3f} | ours 난이도적응(3~8) {np.mean(adapt):.3f} | ours@8 {np.mean(ours8):.3f}")
print(f"평균 질의 수(적응): 계산상 최소 {BMIN}")
