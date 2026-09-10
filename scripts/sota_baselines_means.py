#!/usr/bin/env python3
"""Aggregate DrAttack/FlipAttack verified (gated) ASR into held-in (panel, 9) and held-out (10) means
for the headline table. Reconstruction-gated (both attacks hide the request), matching tab:main.
Run as jinkwon (restricted results). Skips models not yet done; reports n for each group.
Emits macros -> results/sota_baseline_numbers.tex (copy into paper/)."""
import json, os
PANEL = ["qwen25_3b","qwen25_7b","qwen25_14b","qwen25_32b","llama32_3b_it","llama31_8b_it",
         "gemma2_2b_it","gemma2_9b_it","gemma2_27b"]
HELDOUT = ["phi35_mini","mistral7b","falcon3_7b","glm4_9b","mistral24b",
           "olmo2_7b","zephyr7b"]
MJ = "results/sota_mj_20260909"; LG = "results/sota_lg_20260909"
# keep held-out in lock-step with the selector's scored set so every held-out column shares one n
import json as _json
_sel = _json.load(open("results/heldout_selector_20260908.json"))
_scored = {"MJ":[t for t in HELDOUT if t in _sel["MultiJail"]["per_model"]],
           "LG":[t for t in HELDOUT if t in _sel["Lingua-SafetyBench"]["per_model"]]}
def mean(tags, dirp, cond):
    vs = []
    for t in tags:
        f = f"{dirp}/{t}.json"
        if not os.path.exists(f): continue
        c = json.load(open(f)).get("conds", {}).get(cond)
        if c and c.get("gated") is not None: vs.append(c["gated"])
    return (sum(vs)/len(vs) if vs else None), len(vs)
lines = []
for cond, mac in (("drattack","drattack"), ("flipattack_fcs","flip")):
    for coll, dirp in (("MJ", MJ), ("LG", LG)):
        v, n = mean(PANEL, dirp, cond)
        val = f"{v:.3f}" if v is not None else "--"
        lines.append(f"\\newcommand{{\\pjHI{mac}{coll}}}{{{val}}}")
        v, n = mean(_scored[coll], dirp, cond)
        val = f"{v:.3f}" if v is not None else "--"
        lines.append(f"\\newcommand{{\\pjHO{mac}{coll}}}{{{val}}}")
nhi = mean(PANEL, MJ, "drattack")[1]; nho = mean(_scored["MJ"], MJ, "drattack")[1]
lines.append(f"\\newcommand{{\\pjsotaHIn}}{{{nhi}}}")
lines.append(f"\\newcommand{{\\pjsotaHOn}}{{{nho}}}")
open("results/sota_baseline_numbers.tex","w").write("\n".join(lines)+"\n")
print("\n".join(lines)); print(f"# held-in n={nhi}  held-out n={nho}")
