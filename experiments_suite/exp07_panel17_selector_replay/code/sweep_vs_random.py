#!/usr/bin/env python3
"""Same sweep, but the baseline is random search at the SAME budget, so calibration access is
matched. Also runs the shuffled-family null to see whether hits survive as anything but chance."""
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
fams=sorted({famof(t) for t in DATA[('verified','lg')]['tags']})
SPL={}
for ds in ('lg','mj'):
    n=len(DATA[('verified',ds)]['items']); s=[]
    for i in range(12):
        rng=random.Random(1000+i); idx=list(range(n)); rng.shuffle(idx)
        s.append((idx[:n//2],idx[n//2:]))
    SPL[ds]=s
FSETS=[(3,),(3,5),(3,8),(3,12),(5,8),(3,5,8),(3,5,8,12)]
NSETS=[(2,),(2,4),(2,8),(2,4,6,8)]
ARRS=[('ordered',),('ordered','shuffled')]
EXTRA=[(),('persona',),('fiction',),('persona','fiction'),('persona','fiction','persona+fiction')]
BUDS=[2,4,8]
def cells(met,ds,B,fs,ns,ar,ex,nrand=24):
    D=DATA[(met,ds)]; tg,matrix=D['tags'],D['matrix']
    frames=set(('plain',)+ex)
    arms=sorted([a for a in matrix[tg[0]] if (m:=P.match(a)) and m.group(4) in frames
                 and int(m.group(1)) in fs and int(m.group(3)) in ns and m.group(2) in ar])
    if len(arms)<max(6,2*B): return None
    pr={t:{a:BJ[t]['prior'].get(a,0.) for a in arms} for t in tg}
    order={t:sorted(arms,key=lambda a:-pr[t][a]) for t in tg}
    ben={t:[] for t in tg}; rnd={t:[] for t in tg}
    for si,(cb,te) in enumerate(SPL[ds]):
        cal={t:{a:float(np.mean([matrix[t][a][i] for i in cb])) for a in arms} for t in tg}
        tst={t:{a:float(np.mean([matrix[t][a][i] for i in te])) for a in arms} for t in tg}
        for t in tg:
            ben[t].append(tst[t][max(order[t][:B],key=lambda a:cal[t][a])])
            rr=random.Random(si*131+hash(t)%1000)
            vals=[tst[t][max(rr.sample(arms,B),key=lambda a:cal[t][a])] for _ in range(nrand)]
            rnd[t].append(float(np.mean(vals)))
    return ({t:float(np.mean(v)) for t,v in ben.items()},{t:float(np.mean(v)) for t,v in rnd.items()},len(arms))
RAW={}
for B,fs,ns,ar,ex in itertools.product(BUDS,FSETS,NSETS,ARRS,EXTRA):
    d={}; ok=True
    for key in MATS:
        v=cells(key[0],key[1],B,fs,ns,ar,ex)
        if v is None: ok=False;break
        d[key]=v
    if ok: RAW[(B,fs,ns,ar,ex)]=d
print(f"유효 조합 {len(RAW)}개 (기준선 = 동일 예산 무작위 탐색)")
def summary(perm):
    out=[]
    for c,d in RAW.items():
        cnt=0; tot=0; detail={}
        for key,(mb,mr,na) in d.items():
            tg=DATA[key]['tags']; fmap={t:perm[famof(t)] for t in tg}
            fb={f:np.mean([mb[t] for t in tg if fmap[t]==f]) for f in fams}
            fr={f:np.mean([mr[t] for t in tg if fmap[t]==f]) for f in fams}
            dv=np.array([fb[f]-fr[f] for f in fams])
            p=signflip(dv); detail[key]=(dv.mean(),int((dv>0).sum()),p,float(np.mean(list(fb.values()))),na)
            if dv.mean()>0 and p<0.05: cnt+=1
            tot+=dv.mean()
        out.append((cnt,tot,c,detail))
    out.sort(key=lambda x:(-x[0],-x[1]))
    return out
real=summary({f:f for f in fams})
print(f"실제 최대 유의 수 = {real[0][0]}/4")
print("\n=== 상위 5 (무작위 탐색 대비) ===")
for cnt,tot,c,det in real[:5]:
    B,fs,ns,ar,ex=c; na=list(det.values())[0][4]
    print(f"\n유의 {cnt}/4 | 예산{B} F{fs} n{ns} {'ord' if len(ar)==1 else 'both'} +{list(ex) if ex else '없음'} ({na} arm)")
    for key in sorted(det):
        dm,w,p,absv,_=det[key]
        star='*' if p<0.05 and dm>0 else ' '
        print(f"     {key[0]:8s}{key[1]}: {absv:.3f} 차 {dm:+.3f}{star} 승 {w}/7 p={p:.4f}")
rng=random.Random(0); dist=[]
for i in range(20):
    sh=fams[:]; rng.shuffle(sh)
    dist.append(summary(dict(zip(fams,sh)))[0][0])
print(f"\n계열 라벨 무작위 20회 최대 유의 수: {sorted(dist)}")
print(f"  실제({real[0][0]}) 이상인 비율: {sum(1 for x in dist if x>=real[0][0])}/20")
json.dump([{'nsig':c,'cfg':[B,list(fs),list(ns),list(ar),list(ex)]} for c,_,(B,fs,ns,ar,ex),_ in real[:20]],
          open('sweep_vs_random_top.json','w'),indent=1)
