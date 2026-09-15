#!/usr/bin/env python3
"""Published attacks first, then spend the rest of the budget searching the grid.
Does a wider grid pay off once the four strong baselines are already measured?"""
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
LAD=[('32격자',{'plain'}),('64격자',{'plain','persona'}),
     ('96격자',{'plain','persona','fiction'}),('160격자',{'plain','persona','fiction','pap','persona+fiction'})]
BUD=[8,12,16]
for MX,name in (('item_matrix_mj.json','MultiJail'),('item_matrix.json','Lingua')):
    d=json.loads(Path(MX).read_text()); tags,items,matrix=d['tags'],d['items'],d['matrix']
    allarms=sorted(matrix[tags[0]]); fams=sorted({fam(t) for t in tags})
    priors={t:json.loads((BEN/f'{t}.json').read_text())['prior'] for t in tags}
    print(f"\n######## {name}: 공개공격 4개 선행 + 격자 탐색 (7계열 매크로)")
    print(f"{'격자':>8} {'arm':>4} " + " ".join(f"{'예산'+str(b):>8}" for b in BUD) + f" {'오라클':>7}  (공개4전수 기준선)")
    for lab,ws in LAD:
        arms=sorted([a for a in allarms if G.match(a) and a.split('__',1)[1] in ws]+[a for a in allarms if a in CLEAR])
        F=np.array([features(a) for a in arms]); P=[arms.index(a) for a in sorted(CLEAR)]
        per={b:{t:[] for t in tags} for b in BUD}; orc={t:[] for t in tags}; base={t:[] for t in tags}
        for s in range(30):
            rng=random.Random(1000+s); idx=list(range(len(items))); rng.shuffle(idx)
            cb,te=idx[:len(items)//2],idx[len(items)//2:]
            cal={t:{a:float(np.mean([matrix[t][a][i] for i in cb])) for a in arms} for t in tags}
            tst={t:{a:float(np.mean([matrix[t][a][i] for i in te])) for a in arms} for t in tags}
            for t in tags:
                orc[t].append(max(tst[t].values()))
                base[t].append(tst[t][arms[max(P,key=lambda i:cal[t][arms[i]])]])
                pr=np.array([priors[t].get(a,0.) for a in arms],float); mu0=U0*pr
                for B in BUD:
                    q=list(P); ys=[cal[t][arms[i]] for i in q]
                    while len(q)<B:
                        m_,sd=gp_posterior(F[q],F,np.array(ys),mu0[q],mu0)
                        acq=m_+BETA*sd
                        for i in q: acq[i]=-1e9
                        nx=int(np.argmax(acq)); q.append(nx); ys.append(cal[t][arms[nx]])
                    per[B][t].append(tst[t][arms[max(q,key=lambda i:cal[t][arms[i]])]])
        def fm(x):
            M={t:float(np.mean(v)) for t,v in x.items()}
            return {f:float(np.mean([M[t] for t in tags if fam(t)==f])) for f in fams}
        fb=fm(base); fo=fm(orc)
        cells=[]
        for B in BUD:
            fv=fm(per[B]); dv=np.array([fv[f]-fb[f] for f in fams])
            cells.append(f"{np.mean(list(fv.values())):.3f}{'*' if signflip(dv)<0.05 and dv.mean()>0 else ' '}")
        print(f"{lab:>8} {len(arms):4d} " + " ".join(f"{c:>8}" for c in cells) + f" {np.mean(list(fo.values())):7.3f}  {np.mean(list(fb.values())):.3f}")
