#!/usr/bin/env python3
"""17-model panel with the family as the held-out unit, which is what the panel now supports:
five of the old held-out models are inside the panel, so leaving one model out leaks its family."""
import json,random,sys,collections
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
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
for MX,name in (('item_matrix.json','Lingua'),('item_matrix_mj.json','MultiJail')):
    d=json.loads(Path(MX).read_text()); tags,items,matrix=d['tags'],d['items'],d['matrix']
    allarms=sorted(matrix[tags[0]])
    arms=sorted([a for a in allarms if G.match(a) and int(G.match(a).group(1))==3]+[a for a in allarms if a in CLEAR])
    F=np.array([features(a) for a in arms])
    priors={t:json.loads((BEN/f'{t}.json').read_text())['prior'] for t in tags}
    fams=sorted({fam(t) for t in tags})
    per={k:{t:[] for t in tags} for k in ('fixed_famloo','ours8','oracle')}
    for s in range(30):
        rng=random.Random(1000+s); idx=list(range(len(items))); rng.shuffle(idx)
        cb,te=idx[:len(items)//2],idx[len(items)//2:]
        cal={t:{a:float(np.mean([matrix[t][a][i] for i in cb])) for a in arms} for t in tags}
        tst={t:{a:float(np.mean([matrix[t][a][i] for i in te])) for a in arms} for t in tags}
        for t in tags:
            out=[o for o in tags if fam(o)!=fam(t)]          # 계열 전체를 뺀다
            mn={a:float(np.mean([cal[o][a] for o in out])) for a in arms}
            per['fixed_famloo'][t].append(tst[t][max(mn,key=mn.get)])
            per['oracle'][t].append(max(tst[t].values()))
            pr=np.array([priors[t].get(a,0.0) for a in arms],float)
            mu0=U0*pr; q,ys=[],[]
            for _ in range(8):
                if q: m_,sd=gp_posterior(F[q],F,np.array(ys),mu0[q],mu0)
                else: m_,sd=mu0.copy(),np.full(len(arms),SIGMA)
                acq=m_+BETA*sd
                for i in q: acq[i]=-1e9
                nx=int(np.argmax(acq)); q.append(nx); ys.append(cal[t][arms[nx]])
            per['ours8'][t].append(tst[t][arms[max(q,key=lambda i:cal[t][arms[i]])]])
    M={k:{t:float(np.mean(v)) for t,v in per[k].items()} for k in per}
    # 계열 매크로 평균
    famval={k:{f:float(np.mean([M[k][t] for t in tags if fam(t)==f])) for f in fams} for k in M}
    dv=np.array([famval['ours8'][f]-famval['fixed_famloo'][f] for f in fams])
    rng=np.random.default_rng(0); bs=dv[rng.integers(0,len(dv),size=(20000,len(dv)))].mean(1)
    print(f"\n### {name}, 44 arm, 계열 단위 (7계열)")
    print(f"  고정최선(계열LOO) {np.mean(list(famval['fixed_famloo'].values())):.3f} | ours@8 {np.mean(list(famval['ours8'].values())):.3f} | 오라클 {np.mean(list(famval['oracle'].values())):.3f}")
    print(f"  delta {dv.mean():+.3f} CI [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}] 승 {int((dv>0).sum())}/7 p={signflip(dv):.4f}")
    for f in fams:
        print(f"    {f:8s} fixed {famval['fixed_famloo'][f]:.3f}  ours {famval['ours8'][f]:.3f}  oracle {famval['oracle'][f]:.3f}")
