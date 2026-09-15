#!/usr/bin/env python3
"""If a couple of harmful queries are allowed, does a setting hold under both metrics?"""
import json,sys,random,re,itertools
from pathlib import Path
import numpy as np
sys.path.insert(0,'.')
from run_lg_selector import features, gp_posterior, U0, BETA, SIGMA
BEN=Path('/home/ubuntu/342/jinkwon/poly/bucket_0913/L40S-only/experiments_suite/exp02_panel_collect/results/benign')
P=re.compile(r'^g(\d+)_(ordered|shuffled)_n(\d+)__(.+)$')
def fam(t):
    b=t.rsplit('_',1)[0]
    for f in ('qwen','llama','gemma','mistral','phi','falcon','glm'):
        if b.startswith(f): return f
def signflip(dv):
    k=len(dv); m=np.arange(1<<k,dtype=np.int64)[:,None]
    s=1-2*((m>>np.arange(k))&1).astype(np.int8)
    return float((np.abs((s*dv).mean(1))>=abs(dv.mean())-1e-12).mean())
MATS={'verified':{'lg':'item_matrix.json','mj':'item_matrix_mj.json'},
      'asr':{'lg':'item_matrix_unsafe.json','mj':'item_matrix_mj_unsafe.json'}}
DATA={}
for met,mm in MATS.items():
    for ds,f in mm.items():
        d=json.loads(Path(f).read_text()); DATA[(met,ds)]=(d['tags'],d['items'],d['matrix'])
tags=DATA[('verified','lg')][0]; fams=sorted({fam(t) for t in tags})
import glob as _g
priors={Path(x).stem:json.loads(Path(x).read_text())['prior'] for x in _g.glob(str(BEN/'*.json'))}
SPLITS={}
for ds in ('lg','mj'):
    items=DATA[('verified',ds)][1]; sp=[]
    for s in range(30):
        rng=random.Random(1000+s); idx=list(range(len(items))); rng.shuffle(idx)
        sp.append((idx[:len(items)//2],idx[len(items)//2:]))
    SPLITS[ds]=sp
SPACES=[('64 plain+per',{'plain','persona'},(3,5,8,12)),
        ('96 plain+per+fic',{'plain','persona','fiction'},(3,5,8,12)),
        ('64 plain+스택',{'plain','persona+fiction'},(3,5,8,12)),
        ('48 조각3,8 3프레임',{'plain','persona','fiction'},(3,8)),
        ('32 조각3,8 plain+per',{'plain','persona'},(3,8))]
def ev(met,ds,frs,fp,B):
    tg,items,matrix=DATA[(met,ds)]
    arms=sorted([a for a in matrix[tg[0]] if P.match(a) and P.match(a).group(4) in frs
                 and int(P.match(a).group(1)) in fp])
    F=np.array([features(a) for a in arms])
    our={t:[] for t in tg}; fix={t:[] for t in tg}
    for cb,te in SPLITS[ds]:
        cal={t:{a:float(np.mean([matrix[t][a][i] for i in cb])) for a in arms} for t in tg}
        tst={t:{a:float(np.mean([matrix[t][a][i] for i in te])) for a in arms} for t in tg}
        for t in tg:
            out=[o for o in tg if fam(o)!=fam(t)]
            mn={a:float(np.mean([cal[o][a] for o in out])) for a in arms}
            fix[t].append(tst[t][max(mn,key=mn.get)])
            pr=np.array([priors[t].get(a,0.) for a in arms],float); mu0=U0*pr
            q,ys=[],[]
            for _ in range(B):
                if q: m_,sd=gp_posterior(F[q],F,np.array(ys),mu0[q],mu0)
                else: m_,sd=mu0.copy(),np.full(len(arms),SIGMA)
                acq=m_+BETA*sd
                for i in q: acq[i]=-1e9
                nx=int(np.argmax(acq)); q.append(nx); ys.append(cal[t][arms[nx]])
            our[t].append(tst[t][arms[max(q,key=lambda i:cal[t][arms[i]])]])
    Mb={t:float(np.mean(v)) for t,v in our.items()}; Mf={t:float(np.mean(v)) for t,v in fix.items()}
    fb={f:float(np.mean([Mb[t] for t in tg if fam(t)==f])) for f in fams}
    fx={f:float(np.mean([Mf[t] for t in tg if fam(t)==f])) for f in fams}
    dv=np.array([fb[f]-fx[f] for f in fams])
    return dv.mean(),int((dv>0).sum()),signflip(dv),float(np.mean(list(fb.values())))
print("=== 예산을 2~4회 허용: 네 조건 중 몇 개에서 유의한가 ===")
res=[]
for lab,frs,fp in SPACES:
    for B in (2,3,4):
        row={}
        for met in ('verified','asr'):
            for ds in ('lg','mj'):
                row[(met,ds)]=ev(met,ds,frs,fp,B)
        nsig=sum(1 for v in row.values() if v[2]<0.05 and v[0]>0)
        res.append((nsig,sum(v[0] for v in row.values()),lab,B,row))
res.sort(key=lambda x:(-x[0],-x[1]))
for nsig,tot,lab,B,row in res[:6]:
    print(f"\n유의 {nsig}/4 | {lab} 예산{B}")
    for (met,ds),(dm,w,p,absv) in sorted(row.items()):
        star='*' if p<0.05 and dm>0 else ' '
        print(f"    {met:8s} {ds}: {absv:.3f} 차 {dm:+.3f}{star} 승 {w}/7 p={p:.4f}")
