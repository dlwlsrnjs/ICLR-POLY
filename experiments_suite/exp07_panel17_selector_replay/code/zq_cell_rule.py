#!/usr/bin/env python3
"""Improve the cell half of the zero-query rule (frame half fixed to relative-nonrefusal)."""
import json,sys,random,re
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
def fsig(bj,fr,sig='nonrefusal'):
    if fr=='plain': return None
    fs=bj['frame_signals']; parts=[fs[x][sig] for x in fr.split('+') if x in fs]
    return float(np.prod(parts)) if parts else None
FRS={'plain','persona','fiction'}
for MX,name in (('item_matrix_mj.json','MultiJail'),('item_matrix.json','Lingua')):
    d=json.loads(Path(MX).read_text()); tags,items,matrix=d['tags'],d['items'],d['matrix']
    allarms=sorted(matrix[tags[0]]); fams=sorted({fam(t) for t in tags})
    bjs={t:json.loads((BEN/f'{t}.json').read_text()) for t in tags}
    arms=sorted([a for a in allarms if P.match(a) and P.match(a).group(4) in FRS])
    cells=sorted({a.split('__',1)[0] for a in arms})
    popcell={c:float(np.mean([bjs[t]['benign_recon_by_cell'].get(c,0.) for t in tags])) for c in cells}
    pop={f:float(np.mean([fsig(bjs[t],f) or 1.0 for t in tags])) for f in FRS if f!='plain'}
    splits=[]
    for s in range(30):
        rng=random.Random(1000+s); idx=list(range(len(items))); rng.shuffle(idx)
        splits.append((idx[:len(items)//2],idx[len(items)//2:]))
    fix={t:[] for t in tags}
    for cb,te in splits:
        cal={t:{a:float(np.mean([matrix[t][a][i] for i in cb])) for a in arms} for t in tags}
        tst={t:{a:float(np.mean([matrix[t][a][i] for i in te])) for a in arms} for t in tags}
        for t in tags:
            out=[o for o in tags if fam(o)!=fam(t)]
            mn={a:float(np.mean([cal[o][a] for o in out])) for a in arms}
            fix[t].append(tst[t][max(mn,key=mn.get)])
    Mf={t:float(np.mean(v)) for t,v in fix.items()}
    fx={f:float(np.mean([Mf[t] for t in tags if fam(t)==f])) for f in fams}
    def frame_for(t):
        best,bv='plain',1.0
        for f in FRS:
            if f=='plain': continue
            v=fsig(bjs[t],f)
            if v is None: continue
            r=v/max(pop[f],1e-9)
            if r>bv: best,bv=f,r
        return best
    RULES={
      'recon argmax':        lambda t: max(cells,key=lambda c: bjs[t]['benign_recon_by_cell'].get(c,0.)),
      'recon 패널상대 argmax': lambda t: max(cells,key=lambda c: bjs[t]['benign_recon_by_cell'].get(c,0.)/max(popcell[c],1e-9)),
      'recon 최고난도 통과':    lambda t: max([c for c in cells if bjs[t]['benign_recon_by_cell'].get(c,0.)>=0.75] or cells,
                                        key=lambda c: -popcell[c]),
      '패널 최고 recon 셀 고정': lambda t: max(cells,key=lambda c: popcell[c]),
    }
    print(f"\n######## {name} (96 arm, 프레임=상대 nonrefusal)")
    print(f"{'셀 규칙':>22} {'0쿼리':>7} {'고정':>7} {'차이':>8} {'승':>4} {'p':>7}")
    for lab,fn in RULES.items():
        pick={t:f"{fn(t)}__{frame_for(t)}" for t in tags}
        if any(p not in matrix[t] for t,p in pick.items()): 
            print(f"{lab:>22} (arm 없음)"); continue
        ben={t:[] for t in tags}
        for cb,te in splits:
            for t in tags: ben[t].append(float(np.mean([matrix[t][pick[t]][i] for i in te])))
        Mb={t:float(np.mean(v)) for t,v in ben.items()}
        fb={f:float(np.mean([Mb[t] for t in tags if fam(t)==f])) for f in fams}
        dv=np.array([fb[f]-fx[f] for f in fams])
        star='*' if signflip(dv)<0.05 and dv.mean()>0 else ' '
        print(f"{lab:>22} {np.mean(list(fb.values())):7.3f} {np.mean(list(fx.values())):7.3f} {dv.mean():+8.3f}{star} {int((dv>0).sum()):>2}/7 {signflip(dv):7.4f}")
