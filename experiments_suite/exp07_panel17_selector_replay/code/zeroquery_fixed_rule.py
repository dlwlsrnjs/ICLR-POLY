#!/usr/bin/env python3
"""The prior multiplies a frame signal that is always <=1, so plain always wins and MultiJail's
zero-query selector never picks a willingness frame even though persona is worth +0.127 there.
Try selection rules that separate the comprehension cell from the willingness frame."""
import json,sys,random,re
from pathlib import Path
import numpy as np
sys.path.insert(0,'.')
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

def frame_sig(bj,fr,sig):
    if fr=='plain': return None
    fs=bj['frame_signals']
    parts=[fs[x][sig] for x in fr.split('+') if x in fs]
    return float(np.prod(parts)) if parts else None

def pick_arm(bj, arms, rule, sig, pop_mean=None):
    rec=bj['benign_recon_by_cell']
    frames=sorted({a.split('__',1)[1] for a in arms})
    if rule=='raw':                       # 현재 공식
        return max(arms,key=lambda a: rec.get(a.split('__',1)[0],0.)*(frame_sig(bj,a.split('__',1)[1],sig) or 1.0))
    if rule=='split_max':                 # 셀은 recon, 프레임은 신호 argmax (plain 제외)
        cell=max({a.split('__',1)[0] for a in arms}, key=lambda c: rec.get(c,0.))
        cand=[f for f in frames if f!='plain']
        fr=max(cand,key=lambda f: frame_sig(bj,f,sig) or 0.) if cand else 'plain'
        return f"{cell}__{fr}"
    if rule=='split_rel':                 # 프레임 신호를 패널 평균 대비 상대값으로, 1 넘으면 채택
        cell=max({a.split('__',1)[0] for a in arms}, key=lambda c: rec.get(c,0.))
        best,bv='plain',1.0
        for f in frames:
            if f=='plain': continue
            v=frame_sig(bj,f,sig)
            if v is None or pop_mean is None: continue
            r=v/max(pop_mean.get(f,1e-9),1e-9)
            if r>bv: best,bv=f,r
        return f"{cell}__{best}"
    if rule=='always_frame':              # 셀은 recon, 프레임은 고정 persona(있으면)
        cell=max({a.split('__',1)[0] for a in arms}, key=lambda c: rec.get(c,0.))
        fr='persona' if 'persona' in frames else frames[0]
        return f"{cell}__{fr}"
    raise ValueError(rule)

RULES=['raw','split_max','split_rel','always_frame']
for MX,name in (('item_matrix_mj.json','MultiJail'),('item_matrix.json','Lingua')):
    d=json.loads(Path(MX).read_text()); tags,items,matrix=d['tags'],d['items'],d['matrix']
    allarms=sorted(matrix[tags[0]]); fams=sorted({fam(t) for t in tags})
    bjs={t:json.loads((BEN/f'{t}.json').read_text()) for t in tags}
    arms=sorted([a for a in allarms if P.match(a) and P.match(a).group(4) in ('plain','persona','fiction')])
    frames=sorted({a.split('__',1)[1] for a in arms})
    splits=[]
    for s in range(30):
        rng=random.Random(1000+s); idx=list(range(len(items))); rng.shuffle(idx)
        splits.append((idx[:len(items)//2],idx[len(items)//2:]))
    print(f"\n######## {name} (96 arm: plain/persona/fiction)")
    print(f"{'규칙':>14} {'신호':>14} {'0쿼리':>7} {'고정':>7} {'차이':>8} {'승':>4} {'p':>7} {'고른 프레임'}")
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
    for sig in ('nonrefusal','detail','fiction_hold'):
        pop={f:float(np.mean([frame_sig(bjs[t],f,sig) or 1.0 for t in tags])) for f in frames if f!='plain'}
        for rule in RULES:
            pick={t:pick_arm(bjs[t],arms,rule,sig,pop) for t in tags}
            if any(p not in matrix[t] for t,p in pick.items()): continue
            ben={t:[] for t in tags}
            for cb,te in splits:
                for t in tags:
                    ben[t].append(float(np.mean([matrix[t][pick[t]][i] for i in te])))
            Mb={t:float(np.mean(v)) for t,v in ben.items()}
            fb={f:float(np.mean([Mb[t] for t in tags if fam(t)==f])) for f in fams}
            dv=np.array([fb[f]-fx[f] for f in fams])
            star='*' if signflip(dv)<0.05 and dv.mean()>0 else ' '
            frdist={}
            for p in pick.values(): frdist[p.split('__',1)[1]]=frdist.get(p.split('__',1)[1],0)+1
            print(f"{rule:>14} {sig:>14} {np.mean(list(fb.values())):7.3f} {np.mean(list(fx.values())):7.3f} {dv.mean():+8.3f}{star} {int((dv>0).sum()):>2}/7 {signflip(dv):7.4f} {frdist}")
