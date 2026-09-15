#!/usr/bin/env python3
"""Exhaustive arm-space sweep. Budget mode passed as argv[1]: 0 or 2.
WARNING: this screens >1000 combinations, so any p<0.05 here is a screening hit, not a result.
Top candidates must be re-validated with family-LOO rule selection."""
import json,sys,random,re,itertools
from pathlib import Path
import numpy as np
BEN=Path('/home/ubuntu/342/jinkwon/poly/bucket_0913/L40S-only/experiments_suite/exp02_panel_collect/results/benign')
P=re.compile(r'^g(\d+)_(ordered|shuffled)_n(\d+)__(.+)$')
BUDGET=int(sys.argv[1])
def fam(t):
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
fams=sorted({fam(t) for t in DATA[('verified','lg')]['tags']})
SPL={}
for ds in ('lg','mj'):
    n=len(DATA[('verified',ds)]['items']); s=[]
    for i in range(12):
        rng=random.Random(1000+i); idx=list(range(n)); rng.shuffle(idx)
        s.append((idx[:n//2],idx[n//2:]))
    SPL[ds]=s
FSETS=[(3,),(5,),(3,5),(3,8),(3,12),(5,8),(8,12),(3,5,8),(3,8,12),(3,5,8,12)]
NSETS=[(2,),(2,4),(2,8),(4,8),(2,4,6),(4,6,8),(2,4,6,8)]
ARRS=[('ordered',),('ordered','shuffled')]
EXTRA=[(),('persona',),('fiction',),('pap',),('persona+fiction',),
       ('persona','fiction'),('persona','persona+fiction'),('persona','fiction','persona+fiction')]
def run(met,ds,fs,ns,ar,ex):
    D=DATA[(met,ds)]; tg,matrix=D['tags'],D['matrix']
    frames=set(('plain',)+ex)
    arms=sorted([a for a in matrix[tg[0]] if (m:=P.match(a)) and m.group(4) in frames
                 and int(m.group(1)) in fs and int(m.group(3)) in ns and m.group(2) in ar])
    if len(arms) < max(4, 2*BUDGET): return None   # 예산이 arm 수에 근접하면 사실상 전수 시도
    pr={t:{a:BJ[t]['prior'].get(a,0.) for a in arms} for t in tg}
    order={t:sorted(arms,key=lambda a:-pr[t][a]) for t in tg}
    ben={t:[] for t in tg}; fix={t:[] for t in tg}
    for cb,te in SPL[ds]:
        cal={t:{a:float(np.mean([matrix[t][a][i] for i in cb])) for a in arms} for t in tg}
        tst={t:{a:float(np.mean([matrix[t][a][i] for i in te])) for a in arms} for t in tg}
        for t in tg:
            out=[o for o in tg if fam(o)!=fam(t)]
            mn={a:float(np.mean([cal[o][a] for o in out])) for a in arms}
            fix[t].append(tst[t][max(mn,key=mn.get)])
            if BUDGET==0: pick=order[t][0]
            else:
                cand=order[t][:BUDGET]
                pick=max(cand,key=lambda a:cal[t][a])
            ben[t].append(tst[t][pick])
    Mb={t:float(np.mean(v)) for t,v in ben.items()}; Mf={t:float(np.mean(v)) for t,v in fix.items()}
    fb={f:float(np.mean([Mb[t] for t in tg if fam(t)==f])) for f in fams}
    fx={f:float(np.mean([Mf[t] for t in tg if fam(t)==f])) for f in fams}
    dv=np.array([fb[f]-fx[f] for f in fams])
    return dv.mean(),int((dv>0).sum()),signflip(dv),float(np.mean(list(fb.values()))),len(arms)
rows=[]
for fs,ns,ar,ex in itertools.product(FSETS,NSETS,ARRS,EXTRA):
    r={}
    ok=True
    for key in MATS:
        v=run(key[0],key[1],fs,ns,ar,ex)
        if v is None: ok=False;break
        r[key]=v
    if not ok: continue
    nsig=sum(1 for v in r.values() if v[2]<0.05 and v[0]>0)
    npos=sum(1 for v in r.values() if v[0]>0)
    minw=min(v[1] for v in r.values())
    rows.append((nsig,npos,minw,sum(v[0] for v in r.values()),fs,ns,ar,ex,r))
rows.sort(key=lambda x:(-x[0],-x[1],-x[2],-x[3]))
print(f"### 예산 {BUDGET}: 총 {len(rows)}개 조합 스크리닝")
print(f"{'유의':>4} {'양수':>4} {'최소승':>6} {'arm':>4}  조각/언어/배열/프레임")
for nsig,npos,minw,tot,fs,ns,ar,ex,r in rows[:10]:
    n=list(r.values())[0][4]
    print(f"{nsig:>4}/4 {npos:>4}/4 {minw:>5}/7 {n:4d}  F{fs} n{ns} {'ord' if len(ar)==1 else 'both'} +{list(ex) if ex else '없음'}")
    for k in sorted(r):
        dm,w,p,absv,_=r[k]
        star='*' if p<0.05 and dm>0 else ' '
        print(f"         {k[0]:8s}{k[1]}: {absv:.3f} 차 {dm:+.3f}{star} 승 {w}/7 p={p:.4f}")
json.dump([{'nsig':a,'npos':b,'minw':c,'fs':e,'ns':f,'ar':g,'ex':h,
            'res':{f"{k[0]}_{k[1]}":list(v) for k,v in i.items()}} for a,b,c,d,e,f,g,h,i in rows[:40]],
          open(f'sweep_budget{BUDGET}.json','w'),indent=1)
