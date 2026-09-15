#!/usr/bin/env python3
"""Is 'published attacks first' really the selector's win, or just the four baselines?
Controls at matched budget: 4 plain only, 4 plain + 4 random, 4 plain + 4 GP, 8 GP, 8 random."""
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
for MX,name in (('item_matrix_mj.json','MultiJail'),('item_matrix.json','Lingua')):
    d=json.loads(Path(MX).read_text()); tags,items,matrix=d['tags'],d['items'],d['matrix']
    allarms=sorted(matrix[tags[0]])
    grid=[a for a in allarms if G.match(a) and int(G.match(a).group(1))==3]
    arms=sorted(grid+[a for a in allarms if a in CLEAR]); F=np.array([features(a) for a in arms])
    priors={t:json.loads((BEN/f'{t}.json').read_text())['prior'] for t in tags}
    fams=sorted({fam(t) for t in tags})
    K=['plain4','plain4_rand4','plain4_gp4','gp8','rand8','oracle']
    per={k:{t:[] for t in tags} for k in K}
    for s in range(30):
        rng=random.Random(1000+s); idx=list(range(len(items))); rng.shuffle(idx)
        cb,te=idx[:len(items)//2],idx[len(items)//2:]
        cal={t:{a:float(np.mean([matrix[t][a][i] for i in cb])) for a in arms} for t in tags}
        tst={t:{a:float(np.mean([matrix[t][a][i] for i in te])) for a in arms} for t in tags}
        for t in tags:
            per['oracle'][t].append(max(tst[t].values()))
            pr=np.array([priors[t].get(a,0.) for a in arms],float); mu0=U0*pr
            P=[arms.index(a) for a in sorted(CLEAR)]
            per['plain4'][t].append(tst[t][arms[max(P,key=lambda i:cal[t][arms[i]])]])
            r=random.Random(s*13+7)
            extra=r.sample([i for i in range(len(arms)) if i not in P],4)
            q=P+extra; per['plain4_rand4'][t].append(tst[t][arms[max(q,key=lambda i:cal[t][arms[i]])]])
            r2=random.Random(s*29+3)
            q=r2.sample(range(len(arms)),8); per['rand8'][t].append(tst[t][arms[max(q,key=lambda i:cal[t][arms[i]])]])
            for start,key in ((list(P),'plain4_gp4'),([],'gp8')):
                q=list(start); ys=[cal[t][arms[i]] for i in q]
                while len(q)<8:
                    if q: m_,sd=gp_posterior(F[q],F,np.array(ys),mu0[q],mu0)
                    else: m_,sd=mu0.copy(),np.full(len(arms),SIGMA)
                    acq=m_+BETA*sd
                    for i in q: acq[i]=-1e9
                    nx=int(np.argmax(acq)); q.append(nx); ys.append(cal[t][arms[nx]])
                per[key][t].append(tst[t][arms[max(q,key=lambda i:cal[t][arms[i]])]])
    M={k:{t:float(np.mean(v)) for t,v in per[k].items()} for k in K}
    fv={k:float(np.mean([np.mean([M[k][t] for t in tags if fam(t)==f]) for f in fams])) for k in K}
    print(f"\n### {name}, 7계열 매크로")
    lab={'plain4':'공개공격 4개만 (예산4)','plain4_rand4':'공개공격4 + 무작위4 (예산8)',
         'plain4_gp4':'공개공격4 + prior-GP 4 (예산8)','gp8':'prior-GP 8 (기존, 예산8)',
         'rand8':'무작위 8 (예산8)','oracle':'오라클'}
    for k in K: print(f"  {lab[k]:32s} {fv[k]:.3f}")
