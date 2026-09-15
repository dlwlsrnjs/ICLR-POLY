#!/usr/bin/env python3
"""Inside our own 40-arm grid space (published attacks excluded), does the benign prior still
accelerate search? That is the narrower claim the paper can still make."""
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
for MX,name in (('item_matrix_mj.json','MultiJail'),('item_matrix.json','Lingua')):
    d=json.loads(Path(MX).read_text()); tags,items,matrix=d['tags'],d['items'],d['matrix']
    allarms=sorted(matrix[tags[0]])
    arms=sorted([a for a in allarms if G.match(a) and int(G.match(a).group(1))==3])  # 격자만
    F=np.array([features(a) for a in arms])
    priors={t:json.loads((BEN/f'{t}.json').read_text())['prior'] for t in tags}
    fams=sorted({fam(t) for t in tags})
    K=[('ours',3),('ours',8),('rand',3),('rand',8),('unin',3),('unin',8)]
    per={k:{t:[] for t in tags} for k in K}; orc={t:[] for t in tags}
    for s in range(30):
        rng=random.Random(1000+s); idx=list(range(len(items))); rng.shuffle(idx)
        cb,te=idx[:len(items)//2],idx[len(items)//2:]
        cal={t:{a:float(np.mean([matrix[t][a][i] for i in cb])) for a in arms} for t in tags}
        tst={t:{a:float(np.mean([matrix[t][a][i] for i in te])) for a in arms} for t in tags}
        for t in tags:
            orc[t].append(max(tst[t].values()))
            pr=np.array([priors[t].get(a,0.) for a in arms],float)
            for mode,B in K:
                if mode=='rand':
                    r=random.Random(s*11+B); q=r.sample(range(len(arms)),B)
                else:
                    mu0=U0*pr if mode=='ours' else np.zeros(len(arms))
                    q,ys=[],[]
                    for _ in range(B):
                        if q: m_,sd=gp_posterior(F[q],F,np.array(ys),mu0[q],mu0)
                        else: m_,sd=mu0.copy(),np.full(len(arms),SIGMA)
                        acq=m_+BETA*sd
                        for i in q: acq[i]=-1e9
                        nx=int(np.argmax(acq)); q.append(nx); ys.append(cal[t][arms[nx]])
                per[(mode,B)][t].append(tst[t][arms[max(q,key=lambda i:cal[t][arms[i]])]])
    M={k:{t:float(np.mean(v)) for t,v in per[k].items()} for k in K}
    fv={k:{f:float(np.mean([M[k][t] for t in tags if fam(t)==f])) for f in fams} for k in K}
    om={f:float(np.mean([float(np.mean(orc[t])) for t in tags if fam(t)==f])) for f in fams}
    print(f"\n### {name}, 격자 40 arm만, 7계열 매크로 (격자 오라클 {np.mean(list(om.values())):.3f})")
    for k in K: print(f"  {k[0]}@{k[1]:<2d} {np.mean(list(fv[k].values())):.3f}")
    for a,b in ((('ours',3),('rand',3)),(('ours',8),('rand',8)),(('ours',3),('unin',3)),(('ours',8),('unin',8))):
        dv=np.array([fv[a][f]-fv[b][f] for f in fams])
        rng=np.random.default_rng(0); bs=dv[rng.integers(0,len(dv),size=(20000,len(dv)))].mean(1)
        print(f"   {a[0]}@{a[1]} vs {b[0]}@{b[1]}: {dv.mean():+.3f} CI [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}] 승 {int((dv>0).sum())}/7 p={signflip(dv):.4f}")
