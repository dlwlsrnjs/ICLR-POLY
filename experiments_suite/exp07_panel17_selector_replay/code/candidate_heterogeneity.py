#!/usr/bin/env python3
"""Does the 'no fixed arm is best' claim hold in each candidate space?"""
import json,sys,random,collections
from pathlib import Path
import numpy as np
sys.path.insert(0,'.')
from run_lg_selector import G, CLEAR
CAND=[('32 plain',{'plain'}),('64 +persona',{'plain','persona'}),
      ('64 +스택',{'plain','persona+fiction'}),
      ('96 +per,fic',{'plain','persona','fiction'}),
      ('160 C공간',{'plain','persona','fiction','pap','persona+fiction'})]
def fam(t):
    b=t.rsplit('_',1)[0]
    for f in ('qwen','llama','gemma','mistral','phi','falcon','glm'):
        if b.startswith(f): return f
for MX,name in (('item_matrix_mj.json','MultiJail'),('item_matrix.json','Lingua')):
    d=json.loads(Path(MX).read_text()); tags,items,matrix=d['tags'],d['items'],d['matrix']
    allarms=sorted(matrix[tags[0]])
    print(f"\n######## {name} (전체 문항, 17모델)")
    print(f"{'공간':>14} {'arm':>4} {'고유승자':>7} {'최빈승자':>8} {'오라클':>7} {'고정최선':>8} {'오라클-고정':>10} {'계열순열p':>9}")
    for lab,ws in CAND:
        arms=sorted([a for a in allarms if G.match(a) and a.split('__',1)[1] in ws])
        val={t:{a:float(np.mean(matrix[t][a])) for a in arms} for t in tags}
        win={t:max(arms,key=lambda a:val[t][a]) for t in tags}
        cnt=collections.Counter(win.values())
        mean={a:float(np.mean([val[t][a] for t in tags])) for a in arms}
        fixed=max(mean,key=mean.get)
        orc=float(np.mean([val[t][win[t]] for t in tags]))
        fx=float(np.mean([val[t][fixed] for t in tags]))
        byfam=collections.defaultdict(list)
        for t in tags: byfam[fam(t)].append(win[t])
        obs=sum(collections.Counter(v).most_common(1)[0][1] for v in byfam.values())
        rng=random.Random(0); labels=list(win.values()); null=0; N=20000
        for _ in range(N):
            rng.shuffle(labels); i=0; s=0
            for v in byfam.values():
                s+=collections.Counter(labels[i:i+len(v)]).most_common(1)[0][1]; i+=len(v)
            null+= (s>=obs)
        print(f"{lab:>14} {len(arms):4d} {len(cnt):7d} {cnt.most_common(1)[0][1]:>6}/17 {orc:7.3f} {fx:8.3f} {orc-fx:10.3f} {null/N:9.4f}")
