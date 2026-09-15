#!/usr/bin/env python3
"""How often does a screen of this size produce 3-of-4 significant by chance?
Null: shuffle the family labels, rerun the same screen, record the best nsig."""
import json,random,re,itertools
from pathlib import Path
import numpy as np
BEN=Path('/home/ubuntu/342/jinkwon/poly/bucket_0913/L40S-only/experiments_suite/exp02_panel_collect/results/benign')
P=re.compile(r'^g(\d+)_(ordered|shuffled)_n(\d+)__(.+)$')
def famof(t):
    b=t.rsplit('_',1)[0]
    for f in ('qwen','llama','gemma','mistral','phi','falcon','glm'):
        if b.startswith(f): return f
def signflip(dv):
    k=len(dv); m=np.arange(1<<k,dtype=np.int64)[:,None]
    s=1-2*((m>>np.arange(k))&1).astype(np.int8)
    return float((np.abs((s*dv).mean(1))>=abs(dv.mean())-1e-12).mean())
MATS={('verified','lg'):'item_matrix.json',('verified','mj'):'item_matrix_mj.json',
      ('asr','lg'):'item_matrix_unsafe.json',('asr','mj'):'item_matrix_mj_unsafe.json'}
DATA={k:json.loads(Path(v).read_text()) for k,v in MATS.items()}
import glob as _g
BJ={Path(x).stem:json.loads(Path(x).read_text()) for x in _g.glob(str(BEN/'*.json'))}
tags0=DATA[('verified','lg')]['tags']; fams=sorted({famof(t) for t in tags0})
SPL={}
for ds in ('lg','mj'):
    n=len(DATA[('verified',ds)]['items']); s=[]
    for i in range(8):
        rng=random.Random(1000+i); idx=list(range(n)); rng.shuffle(idx)
        s.append((idx[:n//2],idx[n//2:]))
    SPL[ds]=s
# 후보를 줄여서(예산8 상위 40) 귀무분포 추정
CANDS=[(8,tuple(r['fs']),tuple(r['ns']),tuple(r['ar']),tuple(r['ex']))
       for r in json.loads(Path('sweep_budget8.json').read_text())]
def cell_scores(met,ds,B,fs,ns,ar,ex):
    D=DATA[(met,ds)]; tg,matrix=D['tags'],D['matrix']
    frames=set(('plain',)+ex)
    arms=sorted([a for a in matrix[tg[0]] if (m:=P.match(a)) and m.group(4) in frames
                 and int(m.group(1)) in fs and int(m.group(3)) in ns and m.group(2) in ar])
    if len(arms)<max(4,2*B): return None
    pr={t:{a:BJ[t]['prior'].get(a,0.) for a in arms} for t in tg}
    order={t:sorted(arms,key=lambda a:-pr[t][a]) for t in tg}
    ben={t:[] for t in tg}; fix={t:[] for t in tg}
    for cb,te in SPL[ds]:
        cal={t:{a:float(np.mean([matrix[t][a][i] for i in cb])) for a in arms} for t in tg}
        tst={t:{a:float(np.mean([matrix[t][a][i] for i in te])) for a in arms} for t in tg}
        for t in tg:
            out=[o for o in tg if famof(o)!=famof(t)]
            mn={a:float(np.mean([cal[o][a] for o in out])) for a in arms}
            fix[t].append(tst[t][max(mn,key=mn.get)])
            ben[t].append(tst[t][max(order[t][:B],key=lambda a:cal[t][a])])
    return ({t:float(np.mean(v)) for t,v in ben.items()},{t:float(np.mean(v)) for t,v in fix.items()})
RAW={}
for c in CANDS:
    d={}
    ok=True
    for key in MATS:
        v=cell_scores(key[0],key[1],*c)
        if v is None: ok=False;break
        d[key]=v
    if ok: RAW[c]=d
print(f"귀무분포 추정용 후보 {len(RAW)}개")
def nsig_for(perm):
    best=0
    for c,d in RAW.items():
        cnt=0
        for key,(mb,mf) in d.items():
            tg=DATA[key]['tags']
            fmap={t:perm[famof(t)] for t in tg}
            fb={f:np.mean([mb[t] for t in tg if fmap[t]==f]) for f in fams}
            fx={f:np.mean([mf[t] for t in tg if fmap[t]==f]) for f in fams}
            dv=np.array([fb[f]-fx[f] for f in fams])
            if dv.mean()>0 and signflip(dv)<0.05: cnt+=1
        best=max(best,cnt)
    return best
obs=nsig_for({f:f for f in fams})
print(f"실제 최대 nsig = {obs}/4")
rng=random.Random(0); dist=[]
for i in range(20):
    sh=fams[:]; rng.shuffle(sh)
    dist.append(nsig_for(dict(zip(fams,sh))))
print(f"계열 라벨 무작위 20회의 최대 nsig 분포: {sorted(dist)}")
print(f"  >= 실제({obs}) 인 비율: {sum(1 for x in dist if x>=obs)}/20")
