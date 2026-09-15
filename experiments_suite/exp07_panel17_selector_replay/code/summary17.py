#!/usr/bin/env python3
"""17-model headline table, family-macro, both benchmarks, 44-arm space."""
import json,random,sys
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
ROWS=['pap','translated','deepinception','aim','fixed','prior_only','ours3','ours8','oracle']
res={}
for MX,name in (('item_matrix_mj.json','MultiJail'),('item_matrix.json','Lingua')):
    d=json.loads(Path(MX).read_text()); tags,items,matrix=d['tags'],d['items'],d['matrix']
    allarms=sorted(matrix[tags[0]])
    arms=sorted([a for a in allarms if G.match(a) and int(G.match(a).group(1))==3]+[a for a in allarms if a in CLEAR])
    F=np.array([features(a) for a in arms]); fams=sorted({fam(t) for t in tags})
    priors={t:json.loads((BEN/f'{t}.json').read_text())['prior'] for t in tags}
    per={k:{t:[] for t in tags} for k in ROWS}
    for s in range(30):
        rng=random.Random(1000+s); idx=list(range(len(items))); rng.shuffle(idx)
        cb,te=idx[:len(items)//2],idx[len(items)//2:]
        cal={t:{a:float(np.mean([matrix[t][a][i] for i in cb])) for a in arms} for t in tags}
        tst={t:{a:float(np.mean([matrix[t][a][i] for i in te])) for a in arms} for t in tags}
        for t in tags:
            out=[o for o in tags if fam(o)!=fam(t)]
            mn={a:float(np.mean([cal[o][a] for o in out])) for a in arms}
            per['fixed'][t].append(tst[t][max(mn,key=mn.get)])
            per['oracle'][t].append(max(tst[t].values()))
            for k,a in (('pap','m_pap'),('translated','m_translated'),('deepinception','m_deepinception'),('aim','m_aim')):
                per[k][t].append(tst[t][a])
            pr=np.array([priors[t].get(a,0.0) for a in arms],float)
            per['prior_only'][t].append(tst[t][arms[int(np.argmax(pr))]])
            mu0=U0*pr
            for B,key in ((3,'ours3'),(8,'ours8')):
                q,ys=[],[]
                for _ in range(B):
                    if q: m_,sd=gp_posterior(F[q],F,np.array(ys),mu0[q],mu0)
                    else: m_,sd=mu0.copy(),np.full(len(arms),SIGMA)
                    acq=m_+BETA*sd
                    for i in q: acq[i]=-1e9
                    nx=int(np.argmax(acq)); q.append(nx); ys.append(cal[t][arms[nx]])
                per[key][t].append(tst[t][arms[max(q,key=lambda i:cal[t][arms[i]])]])
    M={k:{t:float(np.mean(v)) for t,v in per[k].items()} for k in ROWS}
    fv={k:{f:float(np.mean([M[k][t] for t in tags if fam(t)==f])) for f in fams} for k in ROWS}
    res[name]=fv
    print(f"\n### {name}, 17모델 7계열 매크로, 44 arm, verified, 문항 절반 held-out")
    for k in ROWS:
        print(f"  {k:14s} {np.mean(list(fv[k].values())):.3f}")
    for a,b in (('ours8','fixed'),('ours8','deepinception'),('ours8','aim'),('ours3','fixed')):
        dv=np.array([fv[a][f]-fv[b][f] for f in fams])
        rng=np.random.default_rng(0); bs=dv[rng.integers(0,len(dv),size=(20000,len(dv)))].mean(1)
        print(f"   {a} vs {b}: {dv.mean():+.3f} CI [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}] 승 {int((dv>0).sum())}/7 p={signflip(dv):.4f}")
json.dump({k:{kk:vv for kk,vv in v.items()} for k,v in res.items()},open('summary17.json','w'),indent=1)
print("\n=== 두 벤치마크 평균 ===")
for k in ROWS:
    m=np.mean([np.mean(list(res[n][k].values())) for n in res])
    print(f"  {k:14s} {m:.3f}")
