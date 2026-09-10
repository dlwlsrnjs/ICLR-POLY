import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
qw=json.load(open("results/paper_benign_probe_qwen.json"))
gp=json.load(open("results/paper_benign_probe_gpt4o.json"))
order=[f"interleave_ordered_n{n}" for n in [2,4,6,8,10]]+[f"interleave_shuffled_n{n}" for n in [2,4,6,8,10]]
xlab=[f"o{n}" for n in [2,4,6,8,10]]+[f"s{n}" for n in [2,4,6,8,10]]
x=range(len(order))
fig,ax=plt.subplots(figsize=(7.5,3.8))
ax.plot(x,[qw[c] for c in order],'-o',color="#2166ac",lw=2,label="Qwen2.5-7B (weak): recon falls with difficulty")
ax.plot(x,[gp[c] for c in order],'-s',color="#b2182b",lw=2,label="GPT-4o (strong): recon saturated")
ax.axvline(4.5,color="#999",ls=":",lw=1); ax.text(2,0.62,"ordered",ha="center",color="#555"); ax.text(7,0.62,"shuffled",ha="center",color="#555")
ax.set_xticks(list(x)); ax.set_xticklabels(xlab,fontsize=8)
ax.set_xlabel("difficulty setting  (o=ordered / s=shuffled, number = language load $n$)")
ax.set_ylabel("benign reconstruction rate"); ax.set_ylim(0.6,1.03)
ax.set_title("Benign capability probe: the reconstruction ceiling is model-specific\nand predicts how far difficulty can be pushed (no harmful queries)",fontsize=10)
ax.legend(fontsize=8.5,loc="lower left"); ax.grid(alpha=.25)
plt.tight_layout(); plt.savefig("paper/figures/fig_probe.pdf"); plt.savefig("paper/figures/fig_probe.png",dpi=150)
print("wrote fig_probe")
