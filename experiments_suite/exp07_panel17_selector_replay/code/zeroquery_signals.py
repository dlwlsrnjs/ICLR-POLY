#!/usr/bin/env python3
"""Widen the harmless-signal search for a strictly zero-harmful-query selector."""
import json,sys,random,itertools
from pathlib import Path
import numpy as np
sys.path.insert(0,'.')
from run_lg_selector import G
BEN=Path('/home/ubuntu/342/jinkwon/poly/bucket_0913/L40S-only/experiments_suite/exp02_panel_collect/results/benign')
def fam(t):
    b=t.rsplit('_',1)[0]
    for f in ('qwen','llama','gemma','mistral','phi','falcon','glm'):
        if b.startswith(f): return f
def signflip(dv):
    k=len(dv); m=np.arange(1<<k,dtype=np.int64)[:,None]
    s=1-2*((m>>np.arange(k))&1).astype(np.int8)
    return float((np.abs((s*dv).mean(1))>=abs(dv.mean())-1e-12).mean())
SIG=('nonrefusal','fiction_hold','detail')
def frame_val(fs,fr,sig):
    if fr=='plain': return 1.0
    parts=[fs[x][sig] for x in fr.split('+') if x in fs]
    return float(np.prod(parts)) if parts else 1.0
def score(bj,arms,recon_pow,sig,sig_pow):
    rec=bj['benign_recon_by_cell']; fs=bj['frame_signals']
    return {a: (rec.get(a.split('__',1)[0],0.)**recon_pow) * (frame_val(fs,a.split('__',1)[1],sig)**sig_pow) for a in arms}
CAND=[('32 plain',{'plain'}),('64 +persona',{'plain','persona'}),('96 +per,fic',{'plain','persona','fiction'})]
GRID=[(rp,sig,sp) for rp in (0.5,1.0,2.0) for sig in SIG for sp in (1.0,2.0,4.0)]
for MX,name in (('item_matrix_mj.json','MultiJail'),('item_matrix.json','Lingua')):
    d=json.loads(Path(MX).read_text()); tags,items,matrix=d['tags'],d['items'],d['matrix']
    allarms=sorted(matrix[tags[0]]); fams=sorted({fam(t) for t in tags})
    bjs={t:json.loads((BEN/f'{t}.json').read_text()) for t in tags}
    print(f"\n######## {name}: 0쿼리 신호 조합 탐색 (상위 5개만)")
    for lab,ws in CAND:
        arms=sorted([a for a in allarms if G.match(a) and a.split('__',1)[1] in ws])
        # 고정 기준선 한 번만
        fixv={t:[] for t in tags}; res=[]
        splits=[]
        for s in range(30):
            rng=random.Random(1000+s); idx=list(range(len(items))); rng.shuffle(idx)
            cb,te=idx[:len(items)//2],idx[len(items)//2:]
            cal={t:{a:float(np.mean([matrix[t][a][i] for i in cb])) for a in arms} for t in tags}
            tst={t:{a:float(np.mean([matrix[t][a][i] for i in te])) for a in arms} for t in tags}
            splits.append((cal,tst))
            for t in tags:
                out=[o for o in tags if fam(o)!=fam(t)]
                mn={a:float(np.mean([cal[o][a] for o in out])) for a in arms}
                fixv[t].append(tst[t][max(mn,key=mn.get)])
        Mf={t:float(np.mean(v)) for t,v in fixv.items()}
        ff={f:float(np.mean([Mf[t] for t in tags if fam(t)==f])) for f in fams}
        for rp,sig,spw in GRID:
            ben={t:[] for t in tags}
            sc={t:score(bjs[t],arms,rp,sig,spw) for t in tags}
            pick={t:max(arms,key=lambda a:sc[t][a]) for t in tags}
            for cal,tst in splits:
                for t in tags: ben[t].append(tst[t][pick[t]])
            Mb={t:float(np.mean(v)) for t,v in ben.items()}
            fb={f:float(np.mean([Mb[t] for t in tags if fam(t)==f])) for f in fams}
            dv=np.array([fb[f]-ff[f] for f in fams])
            res.append((dv.mean(),np.mean(list(fb.values())),int((dv>0).sum()),signflip(dv),f"recon^{rp} x {sig}^{spw}"))
        res.sort(reverse=True)
        print(f"  -- {lab} (고정 {np.mean(list(ff.values())):.3f})")
        for dmean,absv,wins,p,desc in res[:5]:
            star='*' if p<0.05 and dmean>0 else ' '
            print(f"     {desc:28s} 0쿼리 {absv:.3f}  차 {dmean:+.3f}{star} 승 {wins}/7 p={p:.4f}")
