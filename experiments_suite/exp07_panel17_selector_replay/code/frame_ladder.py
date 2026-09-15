#!/usr/bin/env python3
"""Add willingness frames one at a time on top of the 32-cell comprehension grid."""
import json,sys,random
from pathlib import Path
import numpy as np
sys.path.insert(0,'.')
from run_lg_selector import features, gp_posterior, U0, BETA, SIGMA, CLEAR, G
BEN=Path('/home/ubuntu/342/jinkwon/poly/bucket_0913/L40S-only/experiments_suite/exp02_panel_collect/results/benign')
def fam(t):
    b=t.rsplit('_',1)[0]
    for f in ('qwen','llama','gemma','mistral','phi','falcon','glm'):
        if b.startswith(f): return f
def signflip(dv):
    k=len(dv); m=np.arange(1<<k,dtype=np.int64)[:,None]
    s=1-2*((m>>np.arange(k))&1).astype(np.int8)
    return float((np.abs((s*dv).mean(1))>=abs(dv.mean())-1e-12).mean())
def sp(a,b):
    ra=np.argsort(np.argsort(a)).astype(float); rb=np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra,rb)[0,1])
LADDER=[('plain만',{'plain'}),
        ('+persona',{'plain','persona'}),
        ('+fiction',{'plain','fiction'}),
        ('+persona,fiction',{'plain','persona','fiction'}),
        ('+pap',{'plain','persona','fiction','pap'}),
        ('+persona+fiction 스택',{'plain','persona','fiction','pap','persona+fiction'})]
for MX,name in (('item_matrix_mj.json','MultiJail'),('item_matrix.json','Lingua')):
    d=json.loads(Path(MX).read_text()); tags,items,matrix=d['tags'],d['items'],d['matrix']
    allarms=sorted(matrix[tags[0]]); fams=sorted({fam(t) for t in tags})
    priors={t:json.loads((BEN/f'{t}.json').read_text())['prior'] for t in tags}
    print(f"\n######## {name} (격자만, 공개공격 제외)")
    print(f"{'프레임':>22} {'arm':>4} {'상관':>7} {'오라클':>7} {'ours@3':>7} {'ours@8':>7} {'rand@8':>7} {'ours3-rand3':>11} {'p':>7}")
    for lab,ws in LADDER:
        arms=sorted([a for a in allarms if G.match(a) and a.split('__',1)[1] in ws])
        F=np.array([features(a) for a in arms])
        rho=[sp(np.array([priors[t].get(a,0.) for a in arms]),
                np.array([float(np.mean(matrix[t][a])) for a in arms])) for t in tags]
        K=['ours3','ours8','rand3','rand8','oracle']
        per={k:{t:[] for t in tags} for k in K}
        for s in range(30):
            rng=random.Random(1000+s); idx=list(range(len(items))); rng.shuffle(idx)
            cb,te=idx[:len(items)//2],idx[len(items)//2:]
            cal={t:{a:float(np.mean([matrix[t][a][i] for i in cb])) for a in arms} for t in tags}
            tst={t:{a:float(np.mean([matrix[t][a][i] for i in te])) for a in arms} for t in tags}
            for t in tags:
                per['oracle'][t].append(max(tst[t].values()))
                pr=np.array([priors[t].get(a,0.) for a in arms],float); mu0=U0*pr
                for B in (3,8):
                    r=random.Random(s*7+B); q=r.sample(range(len(arms)),min(B,len(arms)))
                    per[f'rand{B}'][t].append(tst[t][arms[max(q,key=lambda i:cal[t][arms[i]])]])
                    q,ys=[],[]
                    for _ in range(min(B,len(arms))):
                        if q: m_,sd=gp_posterior(F[q],F,np.array(ys),mu0[q],mu0)
                        else: m_,sd=mu0.copy(),np.full(len(arms),SIGMA)
                        acq=m_+BETA*sd
                        for i in q: acq[i]=-1e9
                        nx=int(np.argmax(acq)); q.append(nx); ys.append(cal[t][arms[nx]])
                    per[f'ours{B}'][t].append(tst[t][arms[max(q,key=lambda i:cal[t][arms[i]])]])
        M={k:{t:float(np.mean(v)) for t,v in per[k].items()} for k in K}
        fvm={k:{f:float(np.mean([M[k][t] for t in tags if fam(t)==f])) for f in fams} for k in K}
        fv={k:float(np.mean(list(fvm[k].values()))) for k in K}
        dv=np.array([fvm['ours3'][f]-fvm['rand3'][f] for f in fams])
        print(f"{lab:>22} {len(arms):4d} {np.mean(rho):+7.3f} {fv['oracle']:7.3f} {fv['ours3']:7.3f} {fv['ours8']:7.3f} {fv['rand8']:7.3f} {dv.mean():+11.3f} {signflip(dv):7.4f}")
