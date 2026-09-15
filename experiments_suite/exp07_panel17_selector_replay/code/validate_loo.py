#!/usr/bin/env python3
"""Pick the arm space itself with leave-one-family-out, then score on the held-out family.
This is the only honest read of the screening above."""
import json,sys,random,re,itertools
from pathlib import Path
import numpy as np
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
CANDS=[]
for B in (2,4,8):
    for r in json.loads(Path(f'sweep_budget{B}.json').read_text())[:15]:
        CANDS.append((B,tuple(r['fs']),tuple(r['ns']),tuple(r['ar']),tuple(r['ex'])))
print(f"후보 {len(CANDS)}개 (예산 2/4/8 각 상위 15)")
def per_family(met,ds,B,fs,ns,ar,ex):
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
            out=[o for o in tg if fam(o)!=fam(t)]
            mn={a:float(np.mean([cal[o][a] for o in out])) for a in arms}
            fix[t].append(tst[t][max(mn,key=mn.get)])
            pick=order[t][0] if B==0 else max(order[t][:B],key=lambda a:cal[t][a])
            ben[t].append(tst[t][pick])
    Mb={t:float(np.mean(v)) for t,v in ben.items()}; Mf={t:float(np.mean(v)) for t,v in fix.items()}
    return ({f:float(np.mean([Mb[t] for t in tg if fam(t)==f])) for f in fams},
            {f:float(np.mean([Mf[t] for t in tg if fam(t)==f])) for f in fams})
CACHE={}
for c in CANDS:
    ok=True; d={}
    for key in MATS:
        v=per_family(key[0],key[1],*c)
        if v is None: ok=False;break
        d[key]=v
    if ok: CACHE[c]=d
print(f"유효 후보 {len(CACHE)}개")
held={f:{} for f in fams}; chosen={}
for h in fams:
    tr=[f for f in fams if f!=h]
    best=None
    for c,d in CACHE.items():
        sc=0
        for key,(fb,fx) in d.items():
            sc+=float(np.mean([fb[f]-fx[f] for f in tr]))
        if best is None or sc>best[0]: best=(sc,c)
    c=best[1]; chosen[h]=c
    for key,(fb,fx) in CACHE[c].items():
        held[h][key]=(fb[h],fx[h])
print("\n=== 공간 선택까지 계열 LOO: held-out 계열 성능 ===")
for key in sorted(MATS):
    dv=np.array([held[f][key][0]-held[f][key][1] for f in fams])
    ab=float(np.mean([held[f][key][0] for f in fams]))
    star='*' if signflip(dv)<0.05 and dv.mean()>0 else ' '
    print(f"  {key[0]:8s}{key[1]}: 우리 {ab:.3f} 차 {dv.mean():+.3f}{star} 승 {int((dv>0).sum())}/7 p={signflip(dv):.4f}")
print("\n계열별로 고른 공간:")
for h in fams:
    B,fs,ns,ar,ex=chosen[h]
    print(f"  {h:8s} 예산{B} F{fs} n{ns} {'ord' if len(ar)==1 else 'both'} +{list(ex) if ex else '없음'}")
