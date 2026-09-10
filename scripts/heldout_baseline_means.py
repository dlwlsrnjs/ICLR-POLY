#!/usr/bin/env python3
"""Per-technique held-out means for the headline table (fills the AIM/DeepInception/PAP held-out cells).
Reads the held-out method result files (restricted, so run as jinkwon), takes the ungated `unsafe` rate
(the baseline-favourable rule for clear-text attacks, matching the held-in column), and means over EXACTLY
the held-out models the selector actually scored (results/heldout_selector_20260908.json per_model keys),
so the baseline rows and the strategy rows always share the same n per collection. Emits macros ->
results/heldout_baseline_numbers.tex (copy into paper/). Run heldout_selector.py FIRST, then this.
Read-only over results/."""
import json, os
# valid held-out = models used in NONE of the panel. n is driven by the selector's scored set, so a
# model whose warm-start probe has not landed yet (e.g. yi34b's 34B probe waiting on a free GPU) is
# excluded from BOTH the strategy columns and these baseline columns, keeping the caption's "same n" true.
HELDOUT = ["phi35_mini","mistral7b","falcon3_7b","glm4_9b","mistral24b","olmo2_7b","zephyr7b"]
SEL = "results/heldout_selector_20260908.json"
MJ = "results/mj_method_20260906"; LG = "results/method_baselines_v2_20260906"
sel = json.load(open(SEL))
SCORED = {"MJ": [t for t in HELDOUT if t in sel["MultiJail"]["per_model"]],
          "LG": [t for t in HELDOUT if t in sel["Lingua-SafetyBench"]["per_model"]]}
def mean(coll, dirp, meth):
    vs = []
    for t in SCORED[coll]:
        f = f"{dirp}/{t}.json"
        if not os.path.exists(f): continue
        m = json.load(open(f)).get("methods", {}).get(meth)
        if m and m.get("unsafe") is not None: vs.append(m["unsafe"])
    return (sum(vs)/len(vs) if vs else None), len(vs)
lines, nmj = [], 0
for coll, dirp in (("MJ", MJ), ("LG", LG)):
    for meth in ("aim","deepinception","pap"):
        v, n = mean(coll, dirp, meth)
        if coll == "MJ" and meth == "aim": nmj = n
        val = f"{v:.3f}" if v is not None else "--"
        lines.append(f"\\newcommand{{\\pjho{coll}{meth}}}{{{val}}}")
lines.append(f"\\newcommand{{\\pjhobasen}}{{{nmj}}}")
os.makedirs("results", exist_ok=True)
open("results/heldout_baseline_numbers.tex","w").write("\n".join(lines)+"\n")
print("\n".join(lines)); print(f"# held-out baseline n (MJ)={len(SCORED['MJ'])}  (LG)={len(SCORED['LG'])}")
