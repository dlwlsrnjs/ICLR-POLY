#!/usr/bin/env python3
"""Adaptive vs best fixed (family-LOO) in each candidate space, family-clustered."""
import json,sys,random
from pathlib import Path
import numpy as np
sys.path.insert(0,'.')
from run_lg_selector import features, gp_posterior, U0, BETA, SIGMA, G
BEN=Path('/home/ubuntu/342/jinkwon/poly/bucket_0913/L40S-only/experiments_suite/exp02_panel_collect/results/benign')
CAND=[('32 plain',{'plain'}),('64 +persona',{'plain','persona'}),
      ('64 +스택',{'plain','persona+fiction'}),
      ('96 +per,fic',{'plain','persona','fiction'}),
      ('160 C공간',{'plain','persona','fiction','pap','persona+fiction'})]
def fam(t):
    b=t.rsplit('_',1)[0]
    for f in ('qwen','llama','gemma','mistral','phi','falcon','glm'):
        if b.startswith(f): return f
def signflip(dv):
    k=len(dv); m=np.arange(1<<k,dtype=np.int64)[:,None]
    s=1-2*((m>>np.arange(k))&1).astype(np.int8)
    return float((np.abs((s*dv).mean(1))>=abs(dv.mean())-1e-12).mean())
for MX,name in (('item_matrix_mj.json','MultiJail'),('item_matrix.json','Lingua')):
    d=json.loads(Path(MX).read_text()); tags,items,matrix=d['tags'],d['items'],d['matrix']
    allarms=sorted(matrix[tags[0]]); fams=sorted({fam(t) for t in tags})
    priors={t:json.loads((BEN/f'{t}.json').read_text())['prior'] for t in tags}
    print(f"\n######## {name} (7계열 매크로, 문항 절반 held-out, 계열 LOO 고정최선)")
    print(f"{'공간':>14} {'arm':>4} {'고정':>7} {'ours@3':>7} {'ours@8':>7} {'ours8-고정':>10} {'승':>4} {'p':>7} {'오라클회수':>9}")
    for lab,ws in CAND:
        arms=sorted([a for a in allarms if G.match(a) and a.split('__',1)[1] in ws])
        F=np.array([features(a) for a in arms])
        K=['fixed','ours3','ours8','oracle']; per={k:{t:[] for t in tags} for k in K}
        for s in range(30):
            rng=random.Random(1000+s); idx=list(range(len(items))); rng.shuffle(idx)
            cb,te=idx[:len(items)//2],idx[len(items)//2:]
            cal={t:{a:float(np.mean([matrix[t][a][i] for i in cb])) for a in arms} for t in tags}
            tst={t:{a:float(np.mean([matrix[t][a][i] for i in te])) for a in arms} for t in tags}
            for t in tags:
                out=[o for o in tags if fam(o)!=fam(t)]
                mn={a:float(np.mean([cal[o][a] for o in out])) for a in arms}
                per['fixed'][t].append(tst[t][max(mn,key=mn.get)])
                per['oracle'][t].append(max(tst[t].values()))
                pr=np.array([priors[t].get(a,0.) for a in arms],float); mu0=U0*pr
                for B,kk in ((3,'ours3'),(8,'ours8')):
                    q,ys=[],[]
                    for _ in range(B):
                        if q: m_,sd=gp_posterior(F[q],F,np.array(ys),mu0[q],mu0)
                        else: m_,sd=mu0.copy(),np.full(len(arms),SIGMA)
                        acq=m_+BETA*sd
                        for i in q: acq[i]=-1e9
                        nx=int(np.argmax(acq)); q.append(nx); ys.append(cal[t][arms[nx]])
                    per[kk][t].append(tst[t][arms[max(q,key=lambda i:cal[t][arms[i]])]])
        M={k:{t:float(np.mean(v)) for t,v in per[k].items()} for k in K}
        fvm={k:{f:float(np.mean([M[k][t] for t in tags if fam(t)==f])) for f in fams} for k in K}
        fv={k:float(np.mean(list(fvm[k].values()))) for k in K}
        dv=np.array([fvm['ours8'][f]-fvm['fixed'][f] for f in fams])
        print(f"{lab:>14} {len(arms):4d} {fv['fixed']:7.3f} {fv['ours3']:7.3f} {fv['ours8']:7.3f} {dv.mean():+10.3f} {int((dv>0).sum()):>2}/7 {signflip(dv):7.4f} {fv['ours8']/fv['oracle']:9.1%}")
