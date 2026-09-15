#!/usr/bin/env python3
"""Select the zero-query rule itself with leave-one-family-out, then score on the held-out family.
Everything above chose the rule after seeing all seven families; this removes that selection bias."""
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
def fsig(bj,fr,sig):
    if fr=='plain': return None
    fs=bj['frame_signals']; parts=[fs[x][sig] for x in fr.split('+') if x in fs]
    return float(np.prod(parts)) if parts else None
# 후보 규칙: (프레임집합, 조각쌍, 신호, 셀규칙)
CANDS=[]
for frs in ({'plain','persona'},{'plain','persona','fiction'}):
    for fp in [(3,),(3,8),(3,5,8,12)]:
        for sig in ('nonrefusal','detail'):
            for cell in ('argmax','panelbest'):
                CANDS.append((frozenset(frs),fp,sig,cell))
def run(matrix,bjs,tags,arms,cells,popframe,popcell,frs,sig,cellrule):
    pick={}
    for t in tags:
        rec=bjs[t]['benign_recon_by_cell']
        c = max(cells,key=lambda c: rec.get(c,0.)) if cellrule=='argmax' else max(cells,key=lambda c: popcell[c])
        best,bv='plain',1.0
        for f in frs:
            if f=='plain': continue
            v=fsig(bjs[t],f,sig)
            if v is None: continue
            r=v/max(popframe.get(f,1e-9),1e-9)
            if r>bv: best,bv=f,r
        pick[t]=f"{c}__{best}"
    return pick
for MX,name in (('item_matrix_mj_unsafe.json','MultiJail'),('item_matrix_unsafe.json','Lingua')):
    d=json.loads(Path(MX).read_text()); tags,items,matrix=d['tags'],d['items'],d['matrix']
    allarms=sorted(matrix[tags[0]]); fams=sorted({fam(t) for t in tags})
    bjs={t:json.loads((BEN/f'{t}.json').read_text()) for t in tags}
    splits=[]
    for s in range(30):
        rng=random.Random(1000+s); idx=list(range(len(items))); rng.shuffle(idx)
        splits.append((idx[:len(items)//2],idx[len(items)//2:]))
    # 각 후보의 모델별 성능과 고정 기준선을 미리 계산
    perf={}; fixv={}
    for frs,fp,sig,cellrule in CANDS:
        arms=sorted([a for a in allarms if P.match(a) and P.match(a).group(4) in frs
                     and int(P.match(a).group(1)) in fp])
        cells=sorted({a.split('__',1)[0] for a in arms})
        popframe={f:float(np.mean([fsig(bjs[t],f,sig) or 1.0 for t in tags])) for f in frs if f!='plain'}
        popcell={c:float(np.mean([bjs[t]['benign_recon_by_cell'].get(c,0.) for t in tags])) for c in cells}
        pick=run(matrix,bjs,tags,arms,cells,popframe,popcell,frs,sig,cellrule)
        if any(p not in matrix[t] for t,p in pick.items()): continue
        key=(frs,fp,sig,cellrule)
        perf[key]={t:float(np.mean([np.mean([matrix[t][pick[t]][i] for i in te]) for cb,te in splits])) for t in tags}
        if fp not in fixv:
            fv={t:[] for t in tags}
            for cb,te in splits:
                cal={t:{a:float(np.mean([matrix[t][a][i] for i in cb])) for a in arms} for t in tags}
                tst={t:{a:float(np.mean([matrix[t][a][i] for i in te])) for a in arms} for t in tags}
                for t in tags:
                    out=[o for o in tags if fam(o)!=fam(t)]
                    mn={a:float(np.mean([cal[o][a] for o in out])) for a in arms}
                    fv[t].append(tst[t][max(mn,key=mn.get)])
            fixv[(frs,fp)]={t:float(np.mean(v)) for t,v in fv.items()}
    # 계열 LOO 로 규칙 선택
    chosen={}; ben={}; fx={}
    for held in fams:
        tr=[t for t in tags if fam(t)!=held]
        best=None
        for key,pm in perf.items():
            base=fixv.get((key[0],key[1]))
            if base is None: continue
            score=float(np.mean([pm[t]-base[t] for t in tr]))
            if best is None or score>best[0]: best=(score,key)
        key=best[1]; chosen[held]=key
        te_tags=[t for t in tags if fam(t)==held]
        ben[held]=float(np.mean([perf[key][t] for t in te_tags]))
        fx[held]=float(np.mean([fixv[(key[0],key[1])][t] for t in te_tags]))
    dv=np.array([ben[f]-fx[f] for f in fams])
    print(f"\n######## {name}: 규칙을 계열 LOO 로 고른 뒤 held-out 계열에서 평가")
    print(f"  0쿼리 {np.mean(list(ben.values())):.3f} | 계열LOO 고정 {np.mean(list(fx.values())):.3f} | 차이 {dv.mean():+.3f} 승 {int((dv>0).sum())}/7 p={signflip(dv):.4f}")
    for f in fams:
        frs,fp,sig,cr=chosen[f]
        print(f"    {f:8s} 고른규칙: 프레임{sorted(frs)} 조각{fp} {sig} 셀={cr}  0쿼리 {ben[f]:.3f} vs 고정 {fx[f]:.3f}")
