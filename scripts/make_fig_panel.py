import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=json.load(open("results/paper_panel_theory.json"))
C=d["C"]; comply=d["comply"]; loo=d["loo"]
order=sorted(C,key=lambda t:C[t])
fig,ax=plt.subplots(1,2,figsize=(11,4))
# left: b(C) relationship
cs=[C[t] for t in order]; bs=[comply[t][1] for t in order]
ax[0].scatter(cs,bs,s=70,color="#b2182b",zorder=5)
for t in order: ax[0].annotate(t.replace("-mini","\n-mini").replace("2.5-7B","2.5").replace("3-8B","3").replace("3.5-mini","3.5"),(C[t],comply[t][1]),fontsize=8,xytext=(4,4),textcoords="offset points")
z=np.polyfit(cs,bs,1); xs=np.linspace(min(cs)-3,max(cs)+3,50)
ax[0].plot(xs,z[0]*xs+z[1],"--",color="#888",label=f"corr={d['b_of_C_corr']:.2f}")
ax[0].set_xlabel("capability $C$ (from benign probe)"); ax[0].set_ylabel("compliance slope $b$ (comply vs difficulty)")
ax[0].set_title("Compliance response rises with capability",fontsize=10.5); ax[0].legend(fontsize=9); ax[0].grid(alpha=.25)
ax[0].axhline(0,color="#ccc",lw=.8)
# right: BO query efficiency
tags=[r[0] for r in loo]; warm=[r[5] for r in loo]; van=[r[6] for r in loo]
x=np.arange(len(tags)); w=0.35
ax[1].bar(x-w/2,warm,w,label="theory-warm BO",color="#1a9850")
ax[1].bar(x+w/2,van,w,label="vanilla BO",color="#999")
ax[1].axhline(10,color="#b2182b",ls=":",lw=1.2,label="full grid (oracle brute force)")
ax[1].set_xticks(x); ax[1].set_xticklabels([t.replace("-mini","-mini").replace("2.5-7B","2.5").replace("3-8B","3").replace("3.5-mini","3.5") for t in tags],fontsize=8,rotation=15)
ax[1].set_ylabel("queries to reach oracle"); ax[1].set_title("Theory-warm Bayesian optimization is query-efficient",fontsize=10.5)
ax[1].legend(fontsize=8.5); ax[1].grid(alpha=.25,axis="y")
plt.tight_layout(); plt.savefig("paper/figures/fig_panel.pdf"); plt.savefig("paper/figures/fig_panel.png",dpi=150)
print("wrote fig_panel")
