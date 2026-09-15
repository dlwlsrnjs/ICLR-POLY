#!/usr/bin/env python3
"""Does search ever start contributing once the four published attacks are already spent?"""
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
BUD=[4,8,12,16,24]
for MX,name in (('item_matrix_mj.json','MultiJail'),('item_matrix.json','Lingua')):
    d=json.loads(Path(MX).read_text()); tags,items,matrix=d['tags'],d['items'],d['matrix']
    allarms=sorted(matrix[tags[0]])
    grid=[a for a in allarms if G.match(a) and int(G.match(a).group(1))==3]
    arms=sorted(grid+[a for a in allarms if a in CLEAR]); F=np.array([features(a) for a in arms])
    priors={t:json.loads((BEN/f'{t}.json').read_text())['prior'] for t in tags}
    fams=sorted({fam(t) for t in tags})
    acc={(m,b):{t:[] for t in tags} for m in ('pf_gp','gp','rand') for b in BUD}
    orc={t:[] for t in tags}
    for s in range(20):
        rng=random.Random(1000+s); idx=list(range(len(items))); rng.shuffle(idx)
        cb,te=idx[:len(items)//2],idx[len(items)//2:]
        cal={t:{a:float(np.mean([matrix[t][a][i] for i in cb])) for a in arms} for t in tags}
        tst={t:{a:float(np.mean([matrix[t][a][i] for i in te])) for a in arms} for t in tags}
        for t in tags:
            orc[t].append(max(tst[t].values()))
            pr=np.array([priors[t].get(a,0.) for a in arms],float); mu0=U0*pr
            P=[arms.index(a) for a in sorted(CLEAR)]
            for B in BUD:
                r=random.Random(s*17+B)
                q=r.sample(range(len(arms)),min(B,len(arms)))
                acc[('rand',B)][t].append(tst[t][arms[max(q,key=lambda i:cal[t][arms[i]])]])
                for start,key in ((list(P),'pf_gp'),([],'gp')):
                    q=list(start)[:B]; ys=[cal[t][arms[i]] for i in q]
                    while len(q)<min(B,len(arms)):
                        if q: m_,sd=gp_posterior(F[q],F,np.array(ys),mu0[q],mu0)
                        else: m_,sd=mu0.copy(),np.full(len(arms),SIGMA)
                        acq=m_+BETA*sd
                        for i in q: acq[i]=-1e9
                        nx=int(np.argmax(acq)); q.append(nx); ys.append(cal[t][arms[nx]])
                    acc[(key,B)][t].append(tst[t][arms[max(q,key=lambda i:cal[t][arms[i]])]])
    def fm(x):
        M={t:float(np.mean(v)) for t,v in x.items()}
        return float(np.mean([np.mean([M[t] for t in tags if fam(t)==f]) for f in fams]))
    print(f"\n### {name}, 7계열 매크로, 44 arm (오라클 {fm(orc):.3f})")
    print(f"  {'예산':>4} {'공개4+GP':>9} {'GP만':>7} {'무작위':>7}")
    for B in BUD:
        print(f"  {B:4d} {fm(acc[('pf_gp',B)]):9.3f} {fm(acc[('gp',B)]):7.3f} {fm(acc[('rand',B)]):7.3f}")
