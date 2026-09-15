#!/usr/bin/env python3
"""The 24-arm space at budgets 2, 4, 8, 12 with the real GP, against random at matched budget."""
import json,random,re,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'.')
from run_lg_selector import features, gp_posterior, U0, BETA, SIGMA
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
print("24 arm 공간, 예산별 (GP vs 동일예산 무작위 vs 계열LOO 고정)")
for key,f in sorted(MATS.items()):
    D=json.loads(Path(f).read_text()); tg,items,matrix=D['tags'],D['items'],D['matrix']
    fams=sorted({famof(t) for t in tg}); frames=set(('plain',)+EX)
    arms=sorted([a for a in matrix[tg[0]] if (m:=P.match(a)) and m.group(4) in frames
                 and int(m.group(1)) in FS and int(m.group(3)) in NS and m.group(2) in AR])
    Fm=np.array([features(a) for a in arms])
    print(f"\n### {key[0]} {key[1]} ({len(arms)} arm)")
    for B in (2,4,8,12):
        per={k:{t:[] for t in tg} for k in ('gp','rand','fixed')}
        for s in range(30):
            rng=random.Random(1000+s); idx=list(range(len(items))); rng.shuffle(idx)
            cb,te=idx[:len(items)//2],idx[len(items)//2:]
            cal={t:{a:float(np.mean([matrix[t][a][i] for i in cb])) for a in arms} for t in tg}
            tst={t:{a:float(np.mean([matrix[t][a][i] for i in te])) for a in arms} for t in tg}
            for t in tg:
                out=[o for o in tg if famof(o)!=famof(t)]
                mn={a:float(np.mean([cal[o][a] for o in out])) for a in arms}
                per['fixed'][t].append(tst[t][max(mn,key=mn.get)])
                rr=random.Random(s*77+hash(t)%997)
                per['rand'][t].append(float(np.mean([tst[t][max(rr.sample(arms,B),key=lambda a:cal[t][a])] for _ in range(24)])))
                pr=np.array([BJ[t]['prior'].get(a,0.) for a in arms],float); mu0=U0*pr
                q,ys=[],[]
                for _ in range(B):
                    if q: m_,sd=gp_posterior(Fm[q],Fm,np.array(ys),mu0[q],mu0)
                    else: m_,sd=mu0.copy(),np.full(len(arms),SIGMA)
                    acq=m_+BETA*sd
                    for i in q: acq[i]=-1e9
                    nx=int(np.argmax(acq)); q.append(nx); ys.append(cal[t][arms[nx]])
                per['gp'][t].append(tst[t][arms[max(q,key=lambda i:cal[t][arms[i]])]])
        M={k:{t:float(np.mean(v)) for t,v in per[k].items()} for k in per}
        fv={k:{f2:float(np.mean([M[k][t] for t in tg if famof(t)==f2])) for f2 in fams} for k in per}
        dr=np.array([fv['gp'][f2]-fv['rand'][f2] for f2 in fams])
        df=np.array([fv['gp'][f2]-fv['fixed'][f2] for f2 in fams])
        s1='*' if signflip(dr)<0.05 and dr.mean()>0 else ' '
        s2='*' if signflip(df)<0.05 and df.mean()>0 else ' '
        print(f"  예산{B:>2}: GP {np.mean(list(fv['gp'].values())):.3f} | vs무작위 {dr.mean():+.3f}{s1} {int((dr>0).sum())}/7 p={signflip(dr):.4f} | vs고정 {df.mean():+.3f}{s2} {int((df>0).sum())}/7 p={signflip(df):.4f}")
