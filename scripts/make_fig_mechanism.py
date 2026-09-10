import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
data=json.load(open("/tmp/mech_curve.json"))["data"]
ns=[d[0] for d in data]; recon=[d[1] for d in data]; gated=[d[2] for d in data]; comply=[d[3] for d in data]
eng=0.2525
fig,ax=plt.subplots(figsize=(6.2,4.0))
ax.plot(ns,recon,'-o',color="#2166ac",lw=2.2,ms=6,label=r"recon$(n)$: faithful de-interleaving")
ax.plot(ns,comply,'-s',color="#b2182b",lw=2.2,ms=6,label=r"comply$(n)$: $P(\mathrm{unsafe}\mid\mathrm{reconstructed})$")
ax.plot(ns,gated,'-^',color="#1a9850",lw=2.6,ms=7,label=r"gated ASR $=$ recon$\times$comply")
ax.axhline(eng,ls="--",color="#777777",lw=1.4)
ax.text(6.0,eng+0.012,"direct-English compliance (0.25)",ha="center",va="bottom",fontsize=8.5,color="#555555")
# mark peak
pn=ns[gated.index(max(gated))]; pg=max(gated)
ax.annotate("sweet spot",xy=(pn,pg),xytext=(pn+1.1,pg+0.075),fontsize=9.5,
            arrowprops=dict(arrowstyle="->",color="#1a9850",lw=1.6),color="#1a7a3f")
# shaded regions
ax.axvspan(1.5,4,color="#fdae61",alpha=0.10)
ax.axvspan(4,10.5,color="#74add1",alpha=0.10)
ax.text(2.55,0.45,"dispersion\nevades alignment",ha="center",fontsize=8.5,color="#9a6a20")
ax.text(8.2,0.45,"over-dispersion\nbreaks reconstruction",ha="center",fontsize=8.5,color="#2c6a9a")
ax.set_xlabel("language load $n$ (number of interleaved languages)")
ax.set_ylabel("rate")
ax.set_xlim(1.5,10.5); ax.set_ylim(0.20,1.02); ax.set_xticks(ns)
ax.legend(loc="lower left",fontsize=8.4,framealpha=0.95)
ax.grid(alpha=0.25)
ax.set_title("Where the jailbreak happens: two competing factors set the sweet spot",fontsize=10.5)
plt.tight_layout()
plt.savefig("paper/figures/fig_mechanism.pdf")
plt.savefig("paper/figures/fig_mechanism.png",dpi=150)
print("wrote fig_mechanism.pdf/png; peak n=",pn)
