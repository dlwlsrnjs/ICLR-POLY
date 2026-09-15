#!/usr/bin/env python3
"""Find a zero-query setting that holds under BOTH verified and plain ASR."""
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
def fsig(bj,fr,sig):
    if fr=='plain': return None
    fs=bj['frame_signals']; parts=[fs[x][sig] for x in fr.split('+') if x in fs]
    return float(np.prod(parts)) if parts else None
MATS={'verified':{'lg':'item_matrix.json','mj':'item_matrix_mj.json'},
      'asr':{'lg':'item_matrix_unsafe.json','mj':'item_matrix_mj_unsafe.json'}}
DATA={}
for met,mm in MATS.items():
    for ds,f in mm.items():
        d=json.loads(Path(f).read_text()); DATA[(met,ds)]=(d['tags'],d['items'],d['matrix'])
tags=DATA[('verified','lg')][0]; fams=sorted({fam(t) for t in tags})
import glob as _g
bjs={Path(x).stem:json.loads(Path(x).read_text()) for x in _g.glob(str(BEN/'*.json'))}
SPLITS={}
for ds in ('lg','mj'):
    items=DATA[('verified',ds)][1]; sp=[]
    for s in range(30):
        rng=random.Random(1000+s); idx=list(range(len(items))); rng.shuffle(idx)
        sp.append((idx[:len(items)//2],idx[len(items)//2:]))
    SPLITS[ds]=sp
FRS=[frozenset(x) for x in ({'plain','persona'},{'plain','fiction'},{'plain','persona','fiction'},
                            {'plain','persona+fiction'},{'plain','persona','persona+fiction'})]
FPS=[(3,),(3,5),(3,8),(3,12),(5,8),(3,5,8,12)]
SIGS=('nonrefusal','detail','fiction_hold')
CELLS=('argmax','panelbest')
def evaluate(met,ds,frs,fp,sig,cellrule):
    tg,items,matrix=DATA[(met,ds)]
    arms=sorted([a for a in matrix[tg[0]] if P.match(a) and P.match(a).group(4) in frs
                 and int(P.match(a).group(1)) in fp])
    if not arms: return None
    cells=sorted({a.split('__',1)[0] for a in arms})
    popf={f:float(np.mean([fsig(bjs[t],f,sig) or 1.0 for t in tg])) for f in frs if f!='plain'}
    popc={c:float(np.mean([bjs[t]['benign_recon_by_cell'].get(c,0.) for t in tg])) for c in cells}
    pick={}
    for t in tg:
        rec=bjs[t]['benign_recon_by_cell']
        c=max(cells,key=lambda x: rec.get(x,0.)) if cellrule=='argmax' else max(cells,key=lambda x: popc[x])
        best,bv='plain',1.0
        for f in frs:
            if f=='plain': continue
            v=fsig(bjs[t],f,sig)
            if v is None: continue
            r=v/max(popf[f],1e-9)
            if r>bv: best,bv=f,r
        pick[t]=f"{c}__{best}"
    if any(p not in matrix[t] for t,p in pick.items()): return None
    ben={t:[] for t in tg}; fix={t:[] for t in tg}
    for cb,te in SPLITS[ds]:
        cal={t:{a:float(np.mean([matrix[t][a][i] for i in cb])) for a in arms} for t in tg}
        tst={t:{a:float(np.mean([matrix[t][a][i] for i in te])) for a in arms} for t in tg}
        for t in tg:
            out=[o for o in tg if fam(o)!=fam(t)]
            mn={a:float(np.mean([cal[o][a] for o in out])) for a in arms}
            fix[t].append(tst[t][max(mn,key=mn.get)]); ben[t].append(tst[t][pick[t]])
    Mb={t:float(np.mean(v)) for t,v in ben.items()}; Mf={t:float(np.mean(v)) for t,v in fix.items()}
    fb={f:float(np.mean([Mb[t] for t in tg if fam(t)==f])) for f in fams}
    fx={f:float(np.mean([Mf[t] for t in tg if fam(t)==f])) for f in fams}
    dv=np.array([fb[f]-fx[f] for f in fams])
    return dv.mean(),int((dv>0).sum()),signflip(dv),float(np.mean(list(fb.values())))
hits=[]
for frs,fp,sig,cr in itertools.product(FRS,FPS,SIGS,CELLS):
    row={}
    ok=True
    for met in ('verified','asr'):
        for ds in ('lg','mj'):
            r=evaluate(met,ds,frs,fp,sig,cr)
            if r is None: ok=False; break
            row[(met,ds)]=r
        if not ok: break
    if not ok: continue
    nsig=sum(1 for k,v in row.items() if v[2]<0.05 and v[0]>0)
    hits.append((nsig,sum(v[0] for v in row.values()),frs,fp,sig,cr,row))
hits.sort(key=lambda x:(-x[0],-x[1]))
print("=== 0쿼리: 네 조건(지표2 × 벤치마크2) 중 몇 개에서 유의한가 ===")
for nsig,tot,frs,fp,sig,cr,row in hits[:8]:
    print(f"\n유의 {nsig}/4 | 프레임{sorted(frs)} 조각{fp} {sig} 셀={cr}")
    for (met,ds),(dm,w,p,absv) in sorted(row.items()):
        star='*' if p<0.05 and dm>0 else ' '
        print(f"    {met:8s} {ds}: 0쿼리 {absv:.3f} 차 {dm:+.3f}{star} 승 {w}/7 p={p:.4f}")
