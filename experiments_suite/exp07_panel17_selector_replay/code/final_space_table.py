#!/usr/bin/env python3
"""Matched-budget comparison of the candidate spaces, family-macro, both benchmarks."""
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
C_WILL={'plain','persona','fiction','pap','persona+fiction'}
def space(allarms,key):
    g=[a for a in allarms if G.match(a)]
    base=[a for a in allarms if a in CLEAR]
    if key=='36':  keep=[a for a in g if a.endswith('__plain')]
    elif key=='84':keep=[a for a in g if a.split('__',1)[1] in C_WILL and int(G.match(a).group(1)) in (3,8)]
    elif key=='164':keep=[a for a in g if a.split('__',1)[1] in C_WILL]
    return sorted(keep+base)
for MX,name in (('item_matrix_mj.json','MultiJail'),('item_matrix.json','Lingua')):
    d=json.loads(Path(MX).read_text()); tags,items,matrix=d['tags'],d['items'],d['matrix']
    allarms=sorted(matrix[tags[0]]); fams=sorted({fam(t) for t in tags})
    priors={t:json.loads((BEN/f'{t}.json').read_text())['prior'] for t in tags}
    print(f"\n######## {name} (7계열 매크로, 문항 절반 held-out)")
    print(f"{'공간':>6} {'arm':>4} {'ours@4':>7} {'ours@8':>7} {'공개4전수':>9} {'rand@8':>7} {'오라클':>7}  ours8-공개4")
    for key in ('36','84','164'):
        arms=space(allarms,key); F=np.array([features(a) for a in arms])
        P=[arms.index(a) for a in sorted(CLEAR)]
        K=['ours4','ours8','plain4','rand8','oracle']
        per={k:{t:[] for t in tags} for k in K}
        for s in range(30):
            rng=random.Random(1000+s); idx=list(range(len(items))); rng.shuffle(idx)
            cb,te=idx[:len(items)//2],idx[len(items)//2:]
            cal={t:{a:float(np.mean([matrix[t][a][i] for i in cb])) for a in arms} for t in tags}
            tst={t:{a:float(np.mean([matrix[t][a][i] for i in te])) for a in arms} for t in tags}
            for t in tags:
                per['oracle'][t].append(max(tst[t].values()))
                per['plain4'][t].append(tst[t][arms[max(P,key=lambda i:cal[t][arms[i]])]])
                r=random.Random(s*5+8); q=r.sample(range(len(arms)),8)
                per['rand8'][t].append(tst[t][arms[max(q,key=lambda i:cal[t][arms[i]])]])
                pr=np.array([priors[t].get(a,0.) for a in arms],float); mu0=U0*pr
                for B,kk in ((4,'ours4'),(8,'ours8')):
                    q,ys=[],[]
                    for _ in range(B):
                        if q: m_,sd=gp_posterior(F[q],F,np.array(ys),mu0[q],mu0)
                        else: m_,sd=mu0.copy(),np.full(len(arms),SIGMA)
                        acq=m_+BETA*sd
                        for i in q: acq[i]=-1e9
                        nx=int(np.argmax(acq)); q.append(nx); ys.append(cal[t][arms[nx]])
                    per[kk][t].append(tst[t][arms[max(q,key=lambda i:cal[t][arms[i]])]])
        M={k:{t:float(np.mean(v)) for t,v in per[k].items()} for k in K}
        fvm={k:{f:float(np.mean([M[k][t] for t in tags if fam(t)==f])) for f in fams} for k in K}
        fv={k:float(np.mean(list(fvm[k].values()))) for k in K}
        dv=np.array([fvm['ours8'][f]-fvm['plain4'][f] for f in fams])
        print(f"{key:>6} {len(arms):4d} {fv['ours4']:7.3f} {fv['ours8']:7.3f} {fv['plain4']:9.3f} {fv['rand8']:7.3f} {fv['oracle']:7.3f}  {dv.mean():+.3f} 승{int((dv>0).sum())}/7 p={signflip(dv):.3f}")
