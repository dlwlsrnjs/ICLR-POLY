#!/usr/bin/env python3
"""Zero harmful queries in the same 24-arm space, with the relative-frame rule."""
import json,random,re
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
import glob as _g
BJ={Path(x).stem:json.loads(Path(x).read_text()) for x in _g.glob(str(BEN/'*.json'))}
FS=(3,5,8,12); NS=(2,); AR=('ordered','shuffled'); EX=('persona','fiction')
def fsig(bj,fr,sig):
    if fr=='plain': return None
    fs=bj['frame_signals']; parts=[fs[x][sig] for x in fr.split('+') if x in fs]
    return float(np.prod(parts)) if parts else None
print("24 arm 공간, 유해 질의 0회 (셀=무해 재구성, 프레임=패널 상대값)")
for sig in ('nonrefusal','detail'):
    print(f"\n-- 신호 {sig}")
    for key,f in sorted(MATS.items()):
        D=json.loads(Path(f).read_text()); tg,items,matrix=D['tags'],D['items'],D['matrix']
        fams=sorted({famof(t) for t in tg}); frames=set(('plain',)+EX)
        arms=sorted([a for a in matrix[tg[0]] if (m:=P.match(a)) and m.group(4) in frames
                     and int(m.group(1)) in FS and int(m.group(3)) in NS and m.group(2) in AR])
        cells=sorted({a.split('__',1)[0] for a in arms})
        popf={fr:float(np.mean([fsig(BJ[t],fr,sig) or 1.0 for t in tg])) for fr in frames if fr!='plain'}
        popc={c:float(np.mean([BJ[t]['benign_recon_by_cell'].get(c,0.) for t in tg])) for c in cells}
        pick={}
        for t in tg:
            c=max(cells,key=lambda x: popc[x])
            best,bv='plain',1.0
            for fr in frames:
                if fr=='plain': continue
                v=fsig(BJ[t],fr,sig)
                if v is None: continue
                r=v/max(popf[fr],1e-9)
                if r>bv: best,bv=fr,r
            pick[t]=f"{c}__{best}"
        ben={t:[] for t in tg}; fix={t:[] for t in tg}; rnd={t:[] for t in tg}
        for s in range(30):
            rng=random.Random(1000+s); idx=list(range(len(items))); rng.shuffle(idx)
            cb,te=idx[:len(items)//2],idx[len(items)//2:]
            cal={t:{a:float(np.mean([matrix[t][a][i] for i in cb])) for a in arms} for t in tg}
            tst={t:{a:float(np.mean([matrix[t][a][i] for i in te])) for a in arms} for t in tg}
            for t in tg:
                out=[o for o in tg if famof(o)!=famof(t)]
                mn={a:float(np.mean([cal[o][a] for o in out])) for a in arms}
                fix[t].append(tst[t][max(mn,key=mn.get)]); ben[t].append(tst[t][pick[t]])
                rr=random.Random(s*55+hash(t)%991)
                rnd[t].append(float(np.mean([tst[t][rr.choice(arms)] for _ in range(24)])))
        M={k:{t:float(np.mean(v)) for t,v in d.items()} for k,d in (('ben',ben),('fix',fix),('rnd',rnd))}
        fv={k:{f2:float(np.mean([M[k][t] for t in tg if famof(t)==f2])) for f2 in fams} for k in M}
        d1=np.array([fv['ben'][f2]-fv['fix'][f2] for f2 in fams])
        d2=np.array([fv['ben'][f2]-fv['rnd'][f2] for f2 in fams])
        s1='*' if signflip(d1)<0.05 and d1.mean()>0 else ' '
        s2='*' if signflip(d2)<0.05 and d2.mean()>0 else ' '
        print(f"   {key[0]:8s}{key[1]}: 0쿼리 {np.mean(list(fv['ben'].values())):.3f} | vs고정 {d1.mean():+.3f}{s1} {int((d1>0).sum())}/7 p={signflip(d1):.4f} | vs무작위1회 {d2.mean():+.3f}{s2} {int((d2>0).sum())}/7 p={signflip(d2):.4f}")
